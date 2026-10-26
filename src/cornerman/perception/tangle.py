"""Clinch and ground frames: person masks, skeletons kept only where clean, and who is on top.

Why this exists (DECISIONS.md and LEARNING.md section 22): long clinch and ground
shots fail tracking's two-people live test and never got pose, and no
off-the-shelf pose method separates ground tangles at broadcast resolution (two
clean skeletons on 2 to 7% of ground frames). So the `masks` stage stores, every
0.5 s where the mat shows, each person's mask from RF-DETR Seg and a ViTPose
skeleton found with every other person greyed out. The `tangle` stage then keeps
a skeleton only where it is clean and decides standing or ground and who is on
top from the masks.

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


def skeleton_clean(
    kx: np.ndarray,
    ky: np.ndarray,
    kc: np.ndarray,
    own: np.ndarray,
    other: np.ndarray,
    scale: float,
    kp_min: float,
    min_kps: int = 8,
    min_inside: float = 0.7,
    max_other: float = 0.15,
) -> bool:
    """Whether a skeleton lies on its own body: at least `min_kps` confident keypoints,
    at least `min_inside` of them in its own mask (grown 2 px), at most `max_other` in
    the other fighter's mask and not its own. `scale` maps pixels to mask pixels."""
    keep = kc >= kp_min
    if keep.sum() < min_kps:
        return False
    h, w = own.shape
    xs = np.clip((kx[keep] * scale).astype(int), 0, w - 1)
    ys = np.clip((ky[keep] * scale).astype(int), 0, h - 1)
    own_g = _grow(own, 2)
    inside = own_g[ys, xs].mean()
    on_other = (other[ys, xs] & ~own_g[ys, xs]).mean()
    return bool(inside >= min_inside and on_other <= max_other)


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
