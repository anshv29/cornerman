"""Labeller A: rules over pose and box geometry (Phase 2, coarse labels only).

What geometry can tell without knowing where the fence is, and without judging
grips:

- ground: a fighter's torso is far from vertical, or their hips are close to
  their ankles in height (sitting, kneeling, lying, holding guard);
- on the ground, the fighter whose hips are clearly higher in the image is on
  top (checked on ViCoS: the top athlete's hips are higher in 98 to 100% of
  mount, side control and guard frames); hips at about the same height means
  nobody is clearly on top;
- standing, boxes overlapping or centres closer than about a third of a body
  height means clinch, else apart.

So the labels are coarse: standing, clinch, ground-red-top, ground-blue-top,
ground-even. Cage pressure, clinch control and hurt are left to the VLM; the
silver set asks the rules to agree only at this coarse level (`agrees`).

Thresholds are set by reasoning and fixed before measuring them on ViCoS; they
are not tuned on it.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cornerman.taxonomy import State

GROUND_ANGLE = 55.0
"""Torso angle from vertical (degrees) above which a fighter is down."""
LOW_HIPS = 0.6
"""Hips less than this many torso lengths above the ankles: sitting, kneeling, lying."""
TOP_MARGIN = 0.25
"""Hip height difference, in torso lengths, needed to call someone on top."""
CLINCH_GAP = 0.35
CLINCH_IOU = 0.15
KP_CONF = 0.3
SHOULDERS, HIPS, ANKLES = [5, 6], [11, 12], [15, 16]

COARSE = ("standing", "clinch", "ground-red-top", "ground-blue-top", "ground-even")


def _mid(kp: np.ndarray, idx: list[int]) -> np.ndarray | None:
    pts = kp[idx]
    ok = pts[:, 2] > KP_CONF
    return pts[ok, :2].mean(axis=0) if ok.any() else None


def body(kp: np.ndarray | None) -> dict[str, float]:
    """Torso angle, torso length, hip height (image y) and hips-over-ankles in torso lengths."""
    nan = float("nan")
    out = {"angle": nan, "torso": nan, "hip_y": nan, "hip_over_ankle": nan}
    if kp is None:
        return out
    sh, hip, ank = _mid(kp, SHOULDERS), _mid(kp, HIPS), _mid(kp, ANKLES)
    if sh is None or hip is None:
        return out
    v = sh - hip
    torso = float(np.hypot(*v))
    if torso < 1:
        return out
    out["angle"] = float(np.degrees(np.arctan2(abs(v[0]), -v[1])))
    out["torso"] = torso
    out["hip_y"] = float(hip[1])
    if ank is not None:
        out["hip_over_ankle"] = float(ank[1] - hip[1]) / torso
    return out


def is_down(b: dict[str, float]) -> bool:
    return b["angle"] >= GROUND_ANGLE or b["hip_over_ankle"] < LOW_HIPS


def frame_label(
    red_kp: np.ndarray | None,
    blue_kp: np.ndarray | None,
    gap: float,
    iou: float,
) -> str | None:
    """Coarse label for one frame, or None if the poses can't support one.

    `gap` is the distance between box centres in body heights, `iou` the box overlap.
    """
    r, b = body(red_kp), body(blue_kp)
    if np.isnan(r["angle"]) or np.isnan(b["angle"]):
        return None
    if is_down(r) or is_down(b):
        torso = (r["torso"] + b["torso"]) / 2
        diff = (b["hip_y"] - r["hip_y"]) / torso  # positive: red's hips higher
        if diff > TOP_MARGIN:
            return "ground-red-top"
        if diff < -TOP_MARGIN:
            return "ground-blue-top"
        return "ground-even"
    if iou > CLINCH_IOU or gap < CLINCH_GAP:
        return "clinch"
    return "standing"


def clip_label(frame_labels: Sequence[str | None], share: float = 0.6) -> str | None:
    """The clip's coarse label: the most common frame label if it covers `share` of
    the frames that have one, else None."""
    known = pd.Series([x for x in frame_labels if x is not None])
    if known.empty:
        return None
    counts = known.value_counts()
    return str(counts.index[0]) if counts.iloc[0] >= share * len(known) else None


def coarse(red: State, blue: State) -> set[str]:
    """Coarse labels the rules may give for a fine (red, blue) state pair.

    Cage pressure can be at arm's length or chest to chest, so it agrees with
    standing or clinch. Scrambles agree with any ground label, since a scramble has
    moments with someone briefly on top.
    """
    if red == State.GROUND_TOP:
        return {"ground-red-top"}
    if blue == State.GROUND_TOP:
        return {"ground-blue-top"}
    if red == State.GROUND_SCRAMBLING:
        return {"ground-even", "ground-red-top", "ground-blue-top"}
    if red in (State.CLINCH_CONTROLLING, State.CLINCH_CONTROLLED):
        return {"clinch"}
    if {red, blue} & {State.STANDING_CAGE_PRESSURING, State.STANDING_CAGE_PRESSURED}:
        return {"standing", "clinch"}
    return {"standing"}


def agrees(red: State, blue: State, rules: str | None) -> bool:
    """True if the rules' coarse label fits the fine state pair."""
    return rules is not None and rules in coarse(red, blue)


def _keypoints(poses: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    """(frame, track_id) -> (17, 3) array of x, y, conf."""
    out: dict[tuple[int, int], np.ndarray] = {}
    for (f, t), g in poses.groupby(["frame", "track_id"]):
        kp = np.zeros((17, 3))
        kp[g["kp"].to_numpy(int)] = g[["x", "y", "conf"]].to_numpy(float)
        out[(int(f), int(t))] = kp  # pyright: ignore[reportArgumentType]
    return out


def label_clips(
    frames: pd.DataFrame, boxes: pd.DataFrame, poses: pd.DataFrame, clips: pd.DataFrame
) -> pd.DataFrame:
    """Labeller A per clip of one fight: clip_id, rules (coarse label or None),
    n_frames (frames with both fighters posed).

    `boxes` is one box per fighter per frame (contracts.perception.fighter_boxes).
    """
    kps = _keypoints(poses)
    by_role: dict[str, dict[int, tuple[int, float, float, float, float]]] = {"red": {}, "blue": {}}
    for role, f, tid, x1, y1, x2, y2 in zip(
        boxes["role"], boxes["frame"], boxes["track_id"],
        boxes["x1"], boxes["y1"], boxes["x2"], boxes["y2"], strict=True,
    ):  # fmt: skip
        if role in by_role:
            by_role[str(role)][int(f)] = (int(tid), float(x1), float(y1), float(x2), float(y2))
    red, blue = by_role["red"], by_role["blue"]
    rows: list[dict[str, object]] = []
    t = frames["t"].to_numpy(float)
    idx = frames["frame"].to_numpy(int)
    for cid, t0, t1 in zip(clips["clip_id"], clips["t0"], clips["t1"], strict=True):
        labels: list[str | None] = []
        for f in idx[(t >= float(t0)) & (t < float(t1))].tolist():
            if f not in red or f not in blue:
                continue
            (rt, *r), (bt, *b) = red[f], blue[f]
            gap, iou = gap_iou(r, b)
            labels.append(frame_label(kps.get((f, rt)), kps.get((f, bt)), gap, iou))
        known = [x for x in labels if x is not None]
        rows.append({"clip_id": cid, "rules": clip_label(labels), "n_frames": len(known)})
    return pd.DataFrame(rows)


def gap_iou(a: list[float], b: list[float]) -> tuple[float, float]:
    """Centre distance in mean box heights, and IoU, of two x1, y1, x2, y2 boxes."""
    h = ((a[3] - a[1]) + (b[3] - b[1])) / 2
    gap = float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2))
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return gap / max(h, 1.0), iw * ih / max(union, 1.0)
