"""Colour features of a person box, used to tell fighter A, fighter B and the referee apart.

In amateur MMA the two fighters wear different coloured shorts and gloves
(often red and blue corners), and the referee wears a shirt and trousers. So
we describe a box by two hue-saturation histograms: the upper body (torso and
gloves) and the shorts band. Brightness is left out on purpose, because it
changes with lighting and camera far more than clothing colour does.
"""

import cv2
import numpy as np

HUE_BINS = 8
SAT_BINS = 4
N_FEATURES = 2 * HUE_BINS * SAT_BINS
UPPER = (0.15, 0.45)
"""Vertical band of the box, as fractions of its height, for the upper body."""
SHORTS = (0.45, 0.70)


def _hist(hsv: np.ndarray) -> np.ndarray:
    if hsv.size == 0:
        return np.zeros(HUE_BINS * SAT_BINS)
    h = hsv[..., 0].ravel().astype(np.int64) * HUE_BINS // 180
    s = hsv[..., 1].ravel().astype(np.int64) * SAT_BINS // 256
    counts = np.bincount(h * SAT_BINS + s, minlength=HUE_BINS * SAT_BINS).astype(float)
    return counts / counts.sum()


def box_features(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Concatenated upper-body and shorts histograms, square-rooted and L2-normalised.

    The square root (Hellinger mapping) stops one dominant colour, like skin,
    from swamping the rest, and makes Euclidean distance a sensible comparison.
    """
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    bh = y2 - y1
    parts: list[np.ndarray] = []
    for lo, hi in (UPPER, SHORTS):
        crop = frame_bgr[y1 + int(lo * bh) : y1 + int(hi * bh), x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.size else crop
        parts.append(_hist(hsv))
    feat = np.sqrt(np.concatenate(parts))
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat
