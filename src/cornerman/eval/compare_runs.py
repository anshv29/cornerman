"""Check that two perception runs of the same fight agree.

Used when we change how perception runs (FP16, batching, a new decoder) but
not what it is supposed to compute. Track IDs are arbitrary numbers, so boxes
are matched frame by frame by overlap (IoU), and then we ask:
- do the same boxes exist in both runs (matched share, median IoU)?
- are they grouped into the same tracks (each run-A track maps to one run-B track)?
- do they get the same role?
- are the keypoints in the same place (pixel differences)?
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MIN_KP_CONF = 0.3


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (n, 4) and b (m, 4), as x1 y1 x2 y2."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_boxes(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """One row per matched box pair: frame, track and role in each run, and IoU."""
    cols = ["x1", "y1", "x2", "y2"]
    rows: list[dict[str, object]] = []
    b_by_frame = dict(tuple(b.groupby("frame")))
    for frame, ga in a.groupby("frame"):
        gb = b_by_frame.get(frame)
        if gb is None:
            continue
        iou = _iou(ga[cols].to_numpy(dtype=float), gb[cols].to_numpy(dtype=float))
        ri, ci = linear_sum_assignment(-iou)
        for i, j in zip(ri, ci, strict=True):
            if iou[i, j] > 0:
                rows.append(
                    {
                        "frame": frame,
                        "track_a": ga["track_id"].iloc[i],
                        "track_b": gb["track_id"].iloc[j],
                        "role_a": ga["role"].iloc[i],
                        "role_b": gb["role"].iloc[j],
                        "iou": float(iou[i, j]),
                    }
                )
    return pd.DataFrame(rows, columns=["frame", "track_a", "track_b", "role_a", "role_b", "iou"])


def _load(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tracks = pd.read_parquet(d / "tracking" / "tracks.parquet")
    roles = pd.read_parquet(d / "identity" / "identity.parquet")
    poses = pd.read_parquet(d / "pose" / "poses.parquet")
    shots = pd.read_parquet(d / "shots" / "shots.parquet")
    tracks = tracks.merge(roles[["track_id", "role"]], on="track_id", how="left")
    return tracks.assign(role=tracks["role"].fillna("other")), poses, shots


def compare_fight(a_dir: Path, b_dir: Path) -> dict[str, float]:
    ta, pa, sa = _load(a_dir)
    tb, pb, sb = _load(b_dir)
    m = match_boxes(ta, tb)
    good = m[m["iou"] >= 0.5]
    # Track agreement: share of matched boxes whose run-B track is the one their
    # run-A track maps to most often.
    top = good.groupby("track_a")["track_b"].agg(lambda s: s.value_counts().index[0])
    same_track = (good["track_b"] == good["track_a"].map(top)).mean() if len(good) else np.nan

    kp = pa.merge(pb, on=["frame", "role", "kp"], suffixes=("_a", "_b"))
    kp = kp[(kp["conf_a"] >= MIN_KP_CONF) & (kp["conf_b"] >= MIN_KP_CONF)]
    dx = kp["x_a"].to_numpy(dtype=float) - kp["x_b"].to_numpy(dtype=float)
    dy = kp["y_a"].to_numpy(dtype=float) - kp["y_b"].to_numpy(dtype=float)
    dist: np.ndarray[tuple[int], np.dtype[np.float64]] = np.hypot(dx, dy)
    return {
        "shots_a": float(len(sa)),
        "shots_b": float(len(sb)),
        "boxes_a": float(len(ta)),
        "boxes_b": float(len(tb)),
        "matched_frac": len(good) / max(len(ta), 1),
        "iou_median": float(good["iou"].median()) if len(good) else np.nan,
        "same_track_frac": float(same_track),
        "same_role_frac": float((good["role_a"] == good["role_b"]).mean()) if len(good) else np.nan,
        "pose_frames_a": float(pa[["frame", "role"]].drop_duplicates().shape[0]),
        "pose_frames_b": float(pb[["frame", "role"]].drop_duplicates().shape[0]),
        "kp_px_median": float(np.median(dist)) if len(dist) else np.nan,
        "kp_px_p95": float(np.percentile(dist, 95)) if len(dist) else np.nan,
        "kp_within_2px": float((dist <= 2).mean()) if len(dist) else np.nan,
    }
