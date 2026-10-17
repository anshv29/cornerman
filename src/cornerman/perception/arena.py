"""Cage or ring mask: keep only people standing on the fighting surface.

Without this, officials, photographers and spectators outside the cage pass
the box-size filter and end up in the "referee" group. A person belongs to the
fight if their feet are on the mat, so:

1. Mat colour: the most common colour in a band across the lower middle of
   ~60 frames spread over the fight (rows 45 to 75%). On a broadcast camera
   the canvas is the biggest single colour there. The band stops above the
   bottom quarter, where the broadcast's score bar sits, and we take the mode
   of a coarse colour histogram rather than a median, because fighters stand
   in the middle of the band. Only bright pixels count (L above MIN_L): a lit
   canvas is bright, while the dark arena, the crowd and black frames would
   otherwise win the vote.
2. Mat area, per frame, at low resolution: pixels close to that colour, holes
   closed (logos, fighters' legs), the large regions kept, then their convex
   hull, since the canvas is convex in the image even when fighters cover parts
   of it. "Close" depends on the canvas. A coloured canvas (the IMMAF one is
   light blue) is matched on chroma only (the a and b channels), with a loose
   brightness floor, because a cageside camera sees the same canvas much
   darker than the main camera does. A white or grey canvas has no chroma to
   match on, so it falls back to full Lab distance.
3. A box is inside when its bottom-centre (the feet) lies in the hull or within
   `margin` of it (feet against the fence sit right at the edge). In tight
   shots the feet are often below the frame edge or behind the broadcast's
   score bar; for a box reaching into the bottom `cut_frac` of the frame we
   can't see the feet, so it is inside if any point across its width, at the
   top of that bottom band, is on the mat.

Frames with no mat region big enough (crowd shots, close-ups) keep nobody.
"""

from collections.abc import Sequence

import cv2
import numpy as np

SMALL = (160, 90)
"""Working resolution (w, h): plenty for a region the size of the canvas, and cheap."""
REGION = (0.45, 0.75, 0.10, 0.90)
"""Rows and columns (fractions of the frame) used to learn the mat colour."""
BIN = 16
"""Lab histogram bin width for finding the most common colour."""
MIN_L = 110
"""OpenCV Lab lightness runs 0 to 255; the canvas is well above mid-grey."""
COLOURED = 10.0
"""A canvas with chroma above this (distance of a, b from neutral 128) is matched on chroma."""
DARK_FACTOR = 0.3
"""On a coloured canvas, pixels darker than this share of the canvas lightness never match."""


def _lab(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, SMALL, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)


def mat_colour(frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
    """Mean Lab colour of the most common coarse colour bin in the sampling band."""
    w, h = SMALL
    r0, r1, c0, c1 = REGION
    pixels = np.concatenate(
        [
            _lab(f)[int(r0 * h) : int(r1 * h), int(c0 * w) : int(c1 * w)].reshape(-1, 3)
            for f in frames_bgr
        ]
    )
    pixels = pixels[pixels[:, 0] >= MIN_L]
    if len(pixels) == 0:
        return np.full(3, np.nan, dtype=np.float32)
    bins = (pixels // BIN).astype(np.int64)
    keys = (bins[:, 0] * 256 + bins[:, 1]) * 256 + bins[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    top = keys == values[np.argmax(counts)]
    return pixels[top].mean(axis=0)


def shot_colour(
    shot_frames: Sequence[np.ndarray], fight_colour: np.ndarray, max_shift: float
) -> np.ndarray:
    """The canvas colour as this shot's camera sees it.

    Different cameras see the same canvas differently (white balance,
    exposure): on the pilot fight the main and cageside cameras were about 33
    apart in chroma. So each shot learns its own colour from its frames, but
    only if that colour is within `max_shift` chroma of the fight-wide colour;
    otherwise (a close-up where a kit fills the band) the fight-wide colour is used.
    """
    own = mat_colour(shot_frames)
    if np.isnan(own).any():
        return fight_colour
    shift = float(np.linalg.norm(own[1:] - fight_colour[1:]))
    return own if shift <= max_shift else fight_colour


def mat_hull(
    frame_bgr: np.ndarray,
    colour: np.ndarray,
    max_delta_e: float,
    min_area: float,
    close_px: int = 9,
    keep_ratio: float = 0.2,
) -> np.ndarray | None:
    """Convex hull of the mat in full-frame pixel coordinates, or None if there's no mat.

    Logos and fighters can split the canvas into pieces, so besides the largest
    region we keep any region at least `keep_ratio` of its size whose centre is
    in the lower two thirds of the frame (vertical blue pads and banners higher
    up are left out).
    """
    lab = _lab(frame_bgr)
    if float(np.hypot(colour[1] - 128, colour[2] - 128)) >= COLOURED:
        near = np.linalg.norm(lab[..., 1:] - colour[1:], axis=2) < max_delta_e
        mask = (near & (lab[..., 0] >= DARK_FACTOR * colour[0])).astype(np.uint8)
    else:
        mask = (np.linalg.norm(lab - colour, axis=2) < max_delta_e).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    if biggest < min_area * SMALL[0] * SMALL[1]:
        return None
    keep = [
        1 + i
        for i, a in enumerate(areas)
        if a == biggest or (a >= keep_ratio * biggest and centroids[1 + i][1] > SMALL[1] / 3)
    ]
    ys, xs = np.nonzero(np.isin(labels, keep))
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    scale = np.array(
        [frame_bgr.shape[1] / SMALL[0], frame_bgr.shape[0] / SMALL[1]], dtype=np.float32
    )
    return (hull.reshape(-1, 2).astype(np.float32) * scale).reshape(-1, 1, 2)


def on_mat(
    hull: np.ndarray | None,
    boxes: np.ndarray,
    margin_px: float,
    frame_h: float,
    cut_frac: float = 0.15,
) -> np.ndarray:
    """True for each box (x1 y1 x2 y2) standing on the mat."""
    if hull is None:
        return np.zeros(len(boxes), dtype=bool)

    def inside(x: float, y: float) -> bool:
        return cv2.pointPolygonTest(hull, (x, y), True) >= -margin_px

    keep: list[bool] = []
    floor = (1 - cut_frac) * frame_h
    for x1, _, x2, y2 in boxes:
        if y2 < floor:
            keep.append(inside(float((x1 + x2) / 2), float(y2)))
        else:
            # Feet not visible. Test the box's width at the cut line, where the
            # canvas is still visible above the score bar, not at the frame edge.
            y = float(floor)
            xs: list[float] = np.linspace(float(x1), float(x2), 9).tolist()
            keep.append(any(inside(x, y) for x in xs))
    return np.array(keep, dtype=bool)
