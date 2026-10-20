"""Candidate clips for the silver consensus set (Phase 2).

The silver set is 300 to 500 clips of 2 to 4 s that a video VLM (two prompts)
and the rules labeller all agree on. Agreement is easy on the common case (two
fighters standing apart), so a plain random sample would be mostly that. Clips
are therefore stratified by a rough position proxy that needs no labels:

- ground: a fighter's torso is far from vertical, or a fighter's box is wider
  than it is tall;
- merged: the two fighters are rarely boxed separately (the detector drew one
  box around both, which happens mostly in clinches and on the ground, or a
  close-up shows only one fighter);
- close: the fighters' boxes overlap or their centres are less than about half a
  body height apart (clinch, cage pressure, exchanges);
- apart: everything else.

The proxy only spreads the sample; it is not a label. Distance to the fence
would help separate cage pressure, but there is no fence geometry yet.

Each clip lies in fighting time inside one shot, so no cut or replay starts
mid-clip.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cornerman.contracts.perception import fighter_boxes, load_roles

STRATA = ("apart", "close", "ground", "merged")
GROUND_ANGLE = 55.0
"""Median torso angle from vertical (degrees) above which a fighter counts as down."""
MERGED_BOTH = 0.5
"""Below this share of frames with both fighters boxed, the clip is 'merged'."""
CLOSE_GAP = 0.45
"""Centre distance in body heights below which the fighters are close. Over 5 fights
the median clip has 0.66 and the lowest quarter is below about 0.48."""
CLOSE_IOU = 0.1
SHOULDERS_HIPS = (5, 6, 11, 12)
KP_CONF = 0.3


def torso_angles(poses: pd.DataFrame) -> pd.DataFrame:
    """Per box (frame, track_id): degrees between the torso and vertical, NaN if unsure."""
    p = poses[poses["kp"].isin(SHOULDERS_HIPS)]
    wide = p.pivot_table(index=["frame", "track_id"], columns="kp", values=["x", "y", "conf"])
    if wide.empty:
        return pd.DataFrame(columns=["frame", "track_id", "angle"])
    ok = (wide["conf"].to_numpy(float) > KP_CONF).all(axis=1)
    dx = (wide["x"][5] + wide["x"][6] - wide["x"][11] - wide["x"][12]) / 2
    dy = (wide["y"][5] + wide["y"][6] - wide["y"][11] - wide["y"][12]) / 2
    angle = np.degrees(np.arctan2(dx.abs(), -dy)).where(ok)
    return angle.rename("angle").reset_index()


def _iou(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    def col(d: pd.DataFrame, c: str) -> np.ndarray:
        return d[c].to_numpy(float)

    w = np.clip(
        np.minimum(col(a, "x2"), col(b, "x2")) - np.maximum(col(a, "x1"), col(b, "x1")), 0, None
    )
    h = np.clip(
        np.minimum(col(a, "y2"), col(b, "y2")) - np.maximum(col(a, "y1"), col(b, "y1")), 0, None
    )
    inter = w * h
    return inter / (_area(a) + _area(b) - inter)


def _area(d: pd.DataFrame) -> np.ndarray:
    return ((d["x2"] - d["x1"]) * (d["y2"] - d["y1"])).to_numpy(float)


def windows(
    frames: pd.DataFrame,
    fighting: pd.DataFrame,
    boxes: pd.DataFrame,
    poses: pd.DataFrame,
    clip_s: float = 3.0,
    min_cover: float = 0.8,
) -> pd.DataFrame:
    """Non-overlapping clips of `clip_s` seconds in fighting time, with position proxies.

    `frames` is tracking/frames.parquet (frame, t, shot_id), `fighting` is
    identity/fighting.parquet, `boxes` is at most one box per fighter per frame
    (contracts.perception.fighter_boxes). A clip is kept if fighting frames cover at
    least `min_cover` of it.

    Columns: shot_id, t0, t1, n_frames, both (share of frames with both fighters
    boxed), then medians over the clip's frames of gap (centre distance in body
    heights), iou, angle_max (the larger torso angle of the two fighters) and
    aspect_min (the smaller box height/width), and stratum.
    """
    cols = ["shot_id", "t0", "t1", "n_frames", "both", "gap", "iou", "angle_max", "aspect_min"]
    f = frames.merge(fighting.loc[fighting["fighting"], ["frame"]], on="frame")
    f = f[f["shot_id"] >= 0].sort_values("frame")
    if f.empty:
        return pd.DataFrame(columns=[*cols, "stratum"])
    step = float(np.median(np.diff(frames["t"].to_numpy()))) if len(frames) > 1 else 0.1
    # A run is consecutive fighting frames in one shot; clips are cut from the start of each run.
    new_run = f["shot_id"].ne(f["shot_id"].shift()) | (f["t"].diff() > 2.5 * step)
    f = f.assign(run=new_run.cumsum())
    f["clip"] = ((f["t"] - f.groupby("run")["t"].transform("min")) // clip_s).astype(int)

    b = boxes.merge(torso_angles(poses), on=["frame", "track_id"], how="left")
    b = b.assign(h=b["y2"] - b["y1"], w=b["x2"] - b["x1"])
    red = b[b["role"] == "red"].set_index("frame")
    blue = b[b["role"] == "blue"].set_index("frame")
    both_frames = red.index.intersection(blue.index)
    r, u = red.loc[both_frames], blue.loc[both_frames]
    pair = pd.DataFrame(
        {
            "gap": np.hypot(
                (r["x1"] + r["x2"] - u["x1"] - u["x2"]) / 2,
                (r["y1"] + r["y2"] - u["y1"] - u["y2"]) / 2,
            )
            / ((r["h"] + u["h"]) / 2),
            "iou": _iou(r, u),
        },
        index=both_frames,
    )
    per_box = pd.DataFrame(
        {
            "angle": b.groupby("frame")["angle"].max(),
            "aspect": (b["h"] / b["w"]).groupby(b["frame"]).min(),
        }
    )

    g = f.set_index("frame").join(pair).join(per_box)
    g["both"] = g.index.isin(both_frames)
    out = g.groupby(["run", "clip"]).agg(
        shot_id=("shot_id", "first"),
        t0=("t", "min"),
        t1=("t", "max"),
        n_frames=("t", "size"),
        both=("both", "mean"),
        gap=("gap", "median"),
        iou=("iou", "median"),
        angle_max=("angle", "median"),
        aspect_min=("aspect", "median"),
    )
    out["t1"] = out["t0"] + clip_s
    out = out[out["n_frames"] >= min_cover * clip_s / step].reset_index(drop=True)
    out["stratum"] = strata(out)
    return out[[*cols, "stratum"]]


def strata(clips: pd.DataFrame) -> np.ndarray:
    """The rough position proxy per clip (see the module docstring). NaN never matches."""
    ground = (clips["angle_max"] >= GROUND_ANGLE) | (clips["aspect_min"] < 1.0)
    merged = clips["both"] < MERGED_BOTH
    close = (clips["gap"] < CLOSE_GAP) | (clips["iou"] > CLOSE_IOU)
    return np.select([ground, merged, close], ["ground", "merged", "close"], "apart")


def sample(clips: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """About n clips, equal shares per stratum, spread evenly over fights.

    Within a stratum, each fight's clips are shuffled and taken in turns (every
    fight's first clip, then every fight's second...), so no fight dominates. A
    stratum short of its share leaves the rest to the others.
    """
    rng = np.random.default_rng(seed)
    c = clips.assign(_r=rng.random(len(clips)))
    c["_turn"] = c.sort_values("_r").groupby(["stratum", "fight_id"]).cumcount()
    c = c.sort_values(["_turn", "_r"])
    share = n // len(STRATA)
    picked = [c[c["stratum"] == s].head(share) for s in STRATA]
    taken = pd.concat(picked)
    rest = c.drop(taken.index).head(max(0, n - len(taken)))
    out = pd.concat([taken, rest]).drop(columns=["_r", "_turn"])
    return out.sort_values(["fight_id", "t0"]).reset_index(drop=True)


def fight_clips(fight_dir: Path, clip_s: float = 3.0) -> pd.DataFrame:
    """`windows` for one processed fight, with its fight_id."""
    fighting = pd.read_parquet(fight_dir / "identity" / "fighting.parquet")
    boxes = fighter_boxes(
        pd.read_parquet(fight_dir / "tracking" / "tracks.parquet"),
        load_roles(fight_dir),
        pd.read_parquet(fight_dir / "tracking" / "live.parquet"),
        fighting,
    )
    w = windows(
        pd.read_parquet(fight_dir / "tracking" / "frames.parquet"),
        fighting,
        boxes,
        pd.read_parquet(fight_dir / "pose" / "poses.parquet"),
        clip_s,
    )
    w = w.assign(fight_id=fight_dir.name)
    return w.assign(clip_id=[f"{fight_dir.name}_{round(t * 10):06d}" for t in w["t0"]])
