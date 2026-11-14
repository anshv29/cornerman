"""Strikes thrown, from limb speed in the skeletons (Step 6, rules first).

A strike is a limb moving fast toward the opponent. Pose runs at 10 fps (12.5 in
some fights), so a punch spans one to three pose frames: enough to see the fast
step, not its shape. So the rule is about speed between two pose frames, measured
relative to the fighter's own body so that stepping in doesn't count:

- punch: a wrist, relative to the fighter's own shoulder centre, moves toward the
  opponent (the component along the line from the fighter's shoulders to the
  opponent's) at `strike_hand_speed` body heights per second or more;
- kick: an ankle, relative to the fighter's own hip centre, moves at
  `strike_foot_speed` or more and ends at least `strike_foot_lift` above the other
  ankle.

Speeds above `strike_max_speed` are keypoints jumping (left and right swapped), not
limbs. One fighter's detections closer than `strike_refractory_s` count once. Body
height is the fighter's box height. Every threshold was set by reasoning before
looking at any result (config.py).

Only where skeletons can be trusted (Ansh, 2026-10-27): both fighters boxed in a
live shot, apart (the moment's contact flag is off and the boxes don't overlap
like a clinch), at striking range (`strike_range`), in fighting time. So strikes in
the clinch and on the ground (knees, elbows, ground and pound) are never found
here; they come from the VLM.

This detects strikes thrown, not landed: landed or missed is Tier 2.
"""

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from cornerman.config import PerceptionConfig
from cornerman.contracts.perception import fighter_boxes
from cornerman.gold.rules import CLINCH_IOU, gap_iou

SHOULDERS = (5, 6)
HIPS = (11, 12)
WRISTS = (9, 10)
ANKLES = (15, 16)
MAX_DT = 0.15
"""Two pose frames further apart than this aren't consecutive (10 or 12.5 fps)."""
NEAR_MOMENT_S = 0.3
"""A pose frame takes the fighting and contact flags of the moment this close to it."""
COLUMNS = ["t", "actor", "limb", "speed"]
FIGHTERS = ("red", "blue")


@dataclass(frozen=True)
class _Frame:
    """One pose frame with both fighters boxed."""

    t: float
    ok: bool
    """Apart and at striking range."""
    kp: dict[str, np.ndarray | None]
    h: dict[str, float]
    """Box height per fighter (body height)."""


def _mid(kp: np.ndarray, idx: tuple[int, int], min_conf: float) -> np.ndarray | None:
    if min(kp[idx[0], 2], kp[idx[1], 2]) < min_conf:
        return None
    return (kp[idx[0], :2] + kp[idx[1], :2]) / 2


def limb_strikes(
    prev: np.ndarray,
    cur: np.ndarray,
    opp_prev: np.ndarray,
    dt: float,
    body_h: float,
    p: PerceptionConfig,
) -> list[tuple[str, float]]:
    """(limb, speed) for each strike between two pose frames of one fighter: `prev` and
    `cur` are its (17, 3) keypoints, `opp_prev` the opponent's at the first frame."""
    out: list[tuple[str, float]] = []
    c = p.kp_min_conf
    sh0, sh1 = _mid(prev, SHOULDERS, c), _mid(cur, SHOULDERS, c)
    opp = _mid(opp_prev, SHOULDERS, c)
    if sh0 is not None and sh1 is not None and opp is not None:
        u = opp - sh0
        if np.linalg.norm(u) > 0:
            u = u / np.linalg.norm(u)
            best = 0.0
            for k in WRISTS:
                if min(prev[k, 2], cur[k, 2]) < c:
                    continue
                v = ((cur[k, :2] - sh1) - (prev[k, :2] - sh0)) / dt / body_h
                speed = float(np.linalg.norm(v))
                toward = float(v @ u)
                if toward >= p.strike_hand_speed and speed <= p.strike_max_speed:
                    best = max(best, toward)
            if best:
                out.append(("hand", best))
    h0, h1 = _mid(prev, HIPS, c), _mid(cur, HIPS, c)
    if h0 is not None and h1 is not None:
        best = 0.0
        for k, other in (ANKLES, ANKLES[::-1]):
            if min(prev[k, 2], cur[k, 2], cur[other, 2]) < c:
                continue
            v = ((cur[k, :2] - h1) - (prev[k, :2] - h0)) / dt / body_h
            speed = float(np.linalg.norm(v))
            lifted = (cur[other, 1] - cur[k, 1]) / body_h >= p.strike_foot_lift
            if p.strike_foot_speed <= speed <= p.strike_max_speed and lifted:
                best = max(best, speed)
        if best:
            out.append(("foot", best))
    return out


def _refractory(rows: list[dict[str, object]], gap_s: float) -> list[dict[str, object]]:
    """Per fighter, drop detections within `gap_s` of the last one kept."""
    out: list[dict[str, object]] = []
    last: dict[str, float] = {}
    for r in sorted(rows, key=lambda r: cast(float, r["t"])):
        actor, t = str(r["actor"]), cast(float, r["t"])
        if t - last.get(actor, -1e9) >= gap_s:
            out.append(r)
            last[actor] = t
    return out


def fight_strikes(fight_dir: Path, moments: pd.DataFrame, p: PerceptionConfig) -> pd.DataFrame:
    """Strikes thrown in one fight. `moments` are its rows of the moments table
    (events/moments.py), for the fighting-time, live and contact flags."""
    m = moments[moments["live"].astype(bool) & ~moments["contact"].astype(bool)]
    if m.empty:
        return pd.DataFrame(columns=COLUMNS)
    tracks = pd.read_parquet(fight_dir / "tracking" / "tracks.parquet",
                             columns=["frame", "t", "track_id", "shot_id", "x1", "y1", "x2",
                                      "y2", "score"])  # fmt: skip
    roles = pd.read_parquet(fight_dir / "identity" / "roles.parquet")
    live = pd.read_parquet(fight_dir / "tracking" / "live.parquet")
    boxes = fighter_boxes(tracks, roles, live)
    cols = ["frame", "t", "track_id", "x1", "y1", "x2", "y2"]
    pair = boxes.loc[boxes["role"] == "red", cols].merge(
        boxes.loc[boxes["role"] == "blue", cols], on=["frame", "t"], suffixes=("_red", "_blue")
    )
    if pair.empty:
        return pd.DataFrame(columns=COLUMNS)
    # Times as whole milliseconds: merge_asof wants an integer tolerance.
    pair = pair.assign(ms=(pair["t"] * 1000).round().astype(np.int64)).sort_values("ms")
    marks = pd.DataFrame({"ms": (m["t"] * 1000).round().astype(np.int64), "moment": 1.0})
    near = pd.merge_asof(pair, marks.sort_values("ms"), on="ms", direction="nearest",
                         tolerance=round(NEAR_MOMENT_S * 1000))  # fmt: skip
    pair = near[near["moment"].notna()].reset_index(drop=True)

    poses = pd.read_parquet(fight_dir / "pose" / "poses.parquet")
    poses = poses[poses["frame"].isin(pair["frame"])]
    kps: dict[tuple[int, int], np.ndarray] = {}
    for (f, tid), g in poses.groupby(["frame", "track_id"]):
        kp = np.zeros((17, 3))
        kp[g["kp"].to_numpy(int)] = g[["x", "y", "conf"]].to_numpy(float)
        kps[(int(f), int(tid))] = kp  # pyright: ignore[reportArgumentType]

    frame_ids = pair["frame"].to_numpy(int)
    times = pair["t"].to_numpy(float)
    box = {c: pair[[f"{k}_{c}" for k in ("x1", "y1", "x2", "y2")]].to_numpy(float)
           for c in FIGHTERS}  # fmt: skip
    track = {c: pair[f"track_id_{c}"].to_numpy(int) for c in FIGHTERS}
    frames: list[_Frame] = []
    for i, f in enumerate(frame_ids):
        gap, iou = gap_iou(box["red"][i].tolist(), box["blue"][i].tolist())
        frames.append(
            _Frame(
                t=float(times[i]),
                ok=gap <= p.strike_range and iou <= CLINCH_IOU,
                kp={c: kps.get((int(f), int(track[c][i]))) for c in FIGHTERS},
                h={c: float(box[c][i, 3] - box[c][i, 1]) for c in FIGHTERS},
            )
        )
    rows: list[dict[str, object]] = []
    for a, b in itertools.pairwise(frames):
        dt = b.t - a.t
        if not (a.ok and b.ok and 0 < dt <= MAX_DT):
            continue
        for me, opp in (("red", "blue"), ("blue", "red")):
            k0, k1, o = a.kp[me], b.kp[me], a.kp[opp]
            if k0 is None or k1 is None or o is None:
                continue
            body_h = (a.h[me] + b.h[me]) / 2
            for limb, speed in limb_strikes(k0, k1, o, dt, body_h, p):
                rows.append({"t": b.t, "actor": me, "limb": limb, "speed": speed})
    kept = _refractory(rows, p.strike_refractory_s)
    return pd.DataFrame(kept, columns=COLUMNS)
