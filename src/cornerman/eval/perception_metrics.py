"""Label-free health numbers for perception, per fight.

These are not accuracy. They are things we can measure without any labels that
move when perception breaks, so they catch obvious failures before the Phase 2
gold set gives real accuracy numbers:
- live_frac: share of sampled frames in live shots;
- both_fighters_frac: in live frames, share where both fighters have a box;
- referee_frac: in live frames, share where the referee has a box;
- role_switches_per_min: how often a fighter's box jumps to where the other
  fighter just was, a sign the identities swapped;
- pose_conf_median: median keypoint confidence over the fighters' poses.
Fighters are the red and blue corners.
"""

import numpy as np
import pandas as pd

from cornerman.contracts.perception import role_keys


def role_switches(boxes: pd.DataFrame) -> int:
    """Count frames where the red corner's box is closer to blue's previous box than to its own.

    `boxes` has one row per frame and fighter role with x1, y1, x2, y2.
    """
    c = boxes.assign(cx=(boxes["x1"] + boxes["x2"]) / 2, cy=(boxes["y1"] + boxes["y2"]) / 2)
    wide = c.pivot_table(index="frame", columns="role", values=["cx", "cy"])
    if ("cx", "red") not in wide.columns or ("cx", "blue") not in wide.columns:
        return 0
    wide = wide.dropna()
    a = wide[[("cx", "red"), ("cy", "red")]].to_numpy()
    b = wide[[("cx", "blue"), ("cy", "blue")]].to_numpy()
    if len(a) < 2:
        return 0
    stay = np.linalg.norm(a[1:] - a[:-1], axis=1) + np.linalg.norm(b[1:] - b[:-1], axis=1)
    swap = np.linalg.norm(a[1:] - b[:-1], axis=1) + np.linalg.norm(b[1:] - a[:-1], axis=1)
    return int((swap < stay).sum())


def fight_metrics(
    frames: pd.DataFrame,
    tracks: pd.DataFrame,
    roles: pd.DataFrame,
    live: pd.DataFrame,
    poses: pd.DataFrame,
    step_seconds: float,
    fighting: pd.DataFrame | None = None,
) -> dict[str, float]:
    """With `fighting`, the per-frame numbers are over fighting frames instead of live ones."""
    live_ids = set(live.loc[live["live"], "shot_id"])
    live_frames = frames[frames["shot_id"].isin(live_ids)]["frame"]
    if fighting is not None:
        live_frames = fighting.loc[fighting["fighting"], "frame"]
    n_live = max(len(live_frames), 1)
    keys = role_keys(roles)
    t = tracks.merge(roles[[*keys, "role"]], on=keys)
    t = t[t["frame"].isin(live_frames)]
    per_frame = t.groupby("frame")["role"].agg(set)
    both = sum(1 for s in per_frame if {"red", "blue"} <= s)
    ref = sum(1 for s in per_frame if "referee" in s)
    fighters = (
        t[t["role"].isin(["red", "blue"])]
        .sort_values("score", ascending=False)
        .drop_duplicates(["frame", "role"])
    )
    live_minutes = n_live * step_seconds / 60
    if "track_id" in poses.columns:
        fighter_boxes = t.loc[t["role"].isin(["red", "blue"]), ["frame", "track_id"]]
        poses = poses.merge(fighter_boxes.drop_duplicates(), on=["frame", "track_id"])
    return {
        "live_frac": len(live_frames) / max(len(frames), 1),
        "live_minutes": live_minutes,
        "fighting_minutes": (
            float(fighting["fighting"].sum()) * step_seconds / 60
            if fighting is not None
            else np.nan
        ),
        "both_fighters_frac": both / n_live,
        "referee_frac": ref / n_live,
        "role_switches_per_min": role_switches(fighters) / max(live_minutes, 1e-9),
        "pose_conf_median": float(poses["conf"].median()) if len(poses) else float("nan"),
        "n_tracks": float(tracks["track_id"].nunique()),
    }
