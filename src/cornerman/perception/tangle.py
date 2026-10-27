"""Clinch and ground frames: person masks, skeletons kept only where clean, and who is on top.

Why this exists (DECISIONS.md and LEARNING.md section 22): long clinch and ground
shots fail tracking's two-people live test and never got pose, and no
off-the-shelf pose method separates ground tangles at broadcast resolution (two
clean skeletons on 2 to 7% of ground frames). So the `masks` stage stores, every
0.5 s where the mat shows, each person's mask from RF-DETR Seg and a ViTPose
skeleton found with every other person greyed out. The `tangle` stage then decides
standing or ground and who is on top from the masks, in fighting time only.

Skeletons are never used from a frame where two people touch (Ansh, 2026-10-27).
A rule meant to keep the clean ones there ("keypoints inside its own mask") failed
its blind check: it kept 12 judgeable skeletons and 3 were clean, because in a
tangle the masks bleed onto the other body. Those frames get mask features only.

The rules below were fixed by reasoning before any check against data.
"""

import numpy as np

GREY = 127
"""What the other people's pixels are painted with before pose, so the model can't
take their limbs."""


def grey_others(frame: np.ndarray, masks: np.ndarray, k: int) -> np.ndarray:
    """The frame with every person's pixels except person k's painted grey."""
    out = frame.copy()
    if len(masks) > 1:
        others = np.delete(masks, k, axis=0).any(axis=0) & ~masks[k]
        out[others] = GREY
    return out


def small_mask(mask: np.ndarray, width: int) -> np.ndarray:
    """A full-resolution mask resized to `width` pixels wide (nearest pixel)."""
    import cv2

    h, w = mask.shape
    size = (width, max(1, round(h * width / w)))
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


def pack(masks: list[np.ndarray]) -> np.ndarray:
    """Masks of one shape as packed bits, one row each (8 times smaller than bool)."""
    if not masks:
        return np.zeros((0, 0), np.uint8)
    return np.stack([np.packbits(m.ravel()) for m in masks])


def unpack(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The inverse of `pack`: (n, h, w) bool."""
    n = shape[0] * shape[1]
    return np.unpackbits(packed, axis=1)[:, :n].reshape(-1, *shape).astype(bool)


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2

    if px <= 0:
        return mask
    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def fighter_pair(masks: np.ndarray, contact_px: int) -> tuple[int, int, int] | None:
    """(i, j, contact) for the two people in most contact, or None if no two touch.

    Fighters in a clinch or on the ground touch; the referee mostly doesn't. Contact is
    the overlap, in pixels, of the two masks each grown by `contact_px`. With one mask
    (both bodies merged into one) or none touching, there's no pair to judge.
    """
    grown = [_grow(m, contact_px) for m in masks]
    best: tuple[int, int, int] | None = None
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            c = int((grown[i] & grown[j]).sum())
            if c > 0 and (best is None or c > best[2]):
                best = (i, j, c)
    return best


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def posture(a: np.ndarray, b: np.ndarray, max_ratio: float = 1.0) -> str:
    """Ground if the pair's combined box is wider than `max_ratio` times its height
    (bodies lying down), else "standing". Upright people are two to three times taller
    than wide, and two of them side by side still taller than wide."""
    x1, y1, x2, y2 = _bbox(a | b)
    return "ground" if (y2 - y1) / max(1, x2 - x1) < max_ratio else "standing"


def top_of(a: np.ndarray, b: np.ndarray, margin: float = 0.1) -> int | None:
    """0 if mask a's centre is higher in the image than b's by at least `margin` of the
    pair's height, 1 if b's is, else None (can't tell)."""
    ya = float(np.nonzero(a)[0].mean())
    yb = float(np.nonzero(b)[0].mean())
    _, y1, _, y2 = _bbox(a | b)
    if abs(ya - yb) < margin * (y2 - y1):
        return None
    return 0 if ya < yb else 1
