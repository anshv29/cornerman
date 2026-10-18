"""Colour samples at the body parts where corner colours live.

Amateur federations put the red and blue corners in red and blue gloves and
shin guards, often with matching shorts trim, while the rest of the kit (and
the referee's uniform) can be black. Colour over a whole box is dominated by
black kit and canvas, which is why box-level clustering failed on the pilot.
With keypoints we can look exactly where the corner colour is.

Per person and frame, a small patch (about 4% of box height across) is read at:
- glove: each wrist;
- shin: halfway between each knee and ankle;
- shorts: just below the midpoint of the hips.
Canvas-coloured pixels are left out (a glove on the mat would otherwise look
like canvas), and the median Lab colour of the rest is stored, with the number
of pixels it came from. A part is skipped when its keypoints are uncertain.
"""

import cv2
import numpy as np
import pandas as pd

GLOVES = ((9,), (10,))
"""COCO keypoints: left wrist, right wrist."""
SHINS = ((13, 15), (14, 16))
"""Knee and ankle pairs; the patch sits halfway between them."""
HIPS = (11, 12)
PATCH = 0.04
"""Patch width as a share of the box height."""
SHORTS_DROP = 0.08
"""How far below the hip midpoint the shorts patch sits, as a share of box height."""
MIN_PIXELS = 5
CANVAS_DELTA = 18.0
"""Pixels within this Lab chroma distance of the canvas are background."""
COLUMNS = ["frame", "track_id", "part", "L", "a", "b", "n"]


def _patch(
    frame_bgr: np.ndarray, x: float, y: float, r: int, canvas_lab: np.ndarray | None
) -> tuple[np.ndarray, int] | None:
    h, w = frame_bgr.shape[:2]
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    lab = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    if canvas_lab is not None:
        lab = lab[np.linalg.norm(lab[:, 1:] - canvas_lab[1:], axis=1) >= CANVAS_DELTA]
    if len(lab) < MIN_PIXELS:
        return None
    return np.median(lab, axis=0), len(lab)


def sample_person(
    frame_bgr: np.ndarray,
    kps: np.ndarray,
    scores: np.ndarray,
    box_h: float,
    canvas_lab: np.ndarray | None,
    min_conf: float,
) -> list[tuple[str, float, float, float, int]]:
    """(part, L, a, b, n_pixels) for each part that could be read."""
    r = max(2, round(PATCH * box_h / 2))
    points: list[tuple[str, float, float]] = []
    for (k,) in GLOVES:
        if scores[k] >= min_conf:
            points.append(("glove", kps[k, 0], kps[k, 1]))
    for knee, ankle in SHINS:
        if min(scores[knee], scores[ankle]) >= min_conf:
            mid = (kps[knee] + kps[ankle]) / 2
            points.append(("shin", mid[0], mid[1]))
    if min(scores[HIPS[0]], scores[HIPS[1]]) >= min_conf:
        mid = (kps[HIPS[0]] + kps[HIPS[1]]) / 2
        points.append(("shorts", mid[0], mid[1] + SHORTS_DROP * box_h))
    out: list[tuple[str, float, float, float, int]] = []
    for part, x, y in points:
        got = _patch(frame_bgr, float(x), float(y), r, canvas_lab)
        if got is not None:
            lab, n = got
            out.append((part, float(lab[0]), float(lab[1]), float(lab[2]), n))
    return out


def empty_samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": pd.Series(dtype="int64"),
            "track_id": pd.Series(dtype="int64"),
            "part": pd.Series(dtype="str"),
            "L": pd.Series(dtype="float64"),
            "a": pd.Series(dtype="float64"),
            "b": pd.Series(dtype="float64"),
            "n": pd.Series(dtype="int64"),
        }
    )
