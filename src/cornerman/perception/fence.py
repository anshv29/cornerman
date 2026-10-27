"""Cage pressure from fence distance (labeller G).

SPEC Layer 1 step 5: cage-pressure states come from measured distance to the
fence. The VLM is not used for cage pressure: its answer flips with the colour
order 46% of the time (DECISIONS.md, silver second build).

The fence is taken to be where the mat ends: the arena stage stores the mat's
outline (a convex hull) in every sampled frame. Not every edge of that hull is a
fence. Where the canvas runs off the side of the picture, or behind the score
bar at the bottom, the hull's edge is the frame border, and nobody standing
there is near a fence. So only "real" edges count (`real_edges`).

Distances are in image space, in body heights (the mean height of the two
fighters' boxes), so a wide shot and a tight shot give similar numbers. The mat
homography that SPEC asks for is not done in v1: on the far side of the cage
the canvas is foreshortened, so a fighter there looks closer to the fence than
they are. That is a known limitation, logged in DECISIONS.md.

The rule, with thresholds set by reasoning before looking at any data:
- both fighters standing, not touching, at striking range;
- one fighter's feet within `near` body heights of a real edge;
- the other fighter farther from that same edge by at least `margin`, so they
  are between the first fighter and the open mat;
- then the one near the fence is pressured and the other is pressuring.
If nobody is near a fence, the moment is neutral. If someone is near a fence
but the other isn't clearly in front of them (both along the fence, side by
side), the rule abstains.
"""

from collections.abc import Sequence

import numpy as np

Segment = tuple[float, float, float, float]
"""x0, y0, x1, y1 in pixels."""

ANKLES = (15, 16)


def real_edges(
    xs: Sequence[float],
    ys: Sequence[float],
    width: float,
    height: float,
    border: float = 0.02,
    bar: float = 0.85,
) -> list[Segment]:
    """The hull's edges that can be a fence, in pixels.

    `xs` and `ys` are the hull's corners as shares of the frame (arena stage). An
    edge is dropped when both its ends lie within `border` of the same side of the
    frame, or both lie below `bar` (the bottom band where the score bar hides the
    canvas, the same band the cage mask treats as "feet hidden").
    """
    n = len(xs)
    if n < 3:
        return []
    out: list[Segment] = []
    for i in range(n):
        x0, y0 = float(xs[i]), float(ys[i])
        x1, y1 = float(xs[(i + 1) % n]), float(ys[(i + 1) % n])
        on_side = (
            (x0 <= border and x1 <= border)
            or (x0 >= 1 - border and x1 >= 1 - border)
            or (y0 <= border and y1 <= border)
            or (y0 >= 1 - border and y1 >= 1 - border)
        )
        if on_side or (y0 >= bar and y1 >= bar):
            continue
        out.append((x0 * width, y0 * height, x1 * width, y1 * height))
    return out


def point_segment(px: float, py: float, seg: Segment, ky: float = 1.0) -> float:
    """Distance in pixels from a point to a line segment, with vertical image distances
    stretched by `ky` (a perspective correction: the floor's depth is squashed in the
    image, people aren't)."""
    x0, y0, x1, y1 = seg
    py, y0, y1 = py * ky, y0 * ky, y1 * ky
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    u = 0.0 if length2 == 0 else min(1.0, max(0.0, ((px - x0) * dx + (py - y0) * dy) / length2))
    return float(np.hypot(px - (x0 + u * dx), py - (y0 + u * dy)))


def feet(
    box: Sequence[float],
    kp: np.ndarray | None,
    kp_min: float,
    frame_h: float,
    cut: float = 0.15,
) -> tuple[float, float] | None:
    """Where a fighter stands, in pixels: the middle of the confident ankles, else the
    bottom centre of the box. None when the box reaches into the bottom `cut` of the
    frame and no ankle is seen (feet hidden by the score bar or the frame edge)."""
    if kp is not None:
        ank = kp[list(ANKLES)]
        ok = ank[:, 2] >= kp_min
        if ok.any():
            x, y = ank[ok, :2].mean(axis=0)
            return float(x), float(y)
    x1, _, x2, y2 = (float(v) for v in box)
    if y2 >= (1 - cut) * frame_h:
        return None
    return (x1 + x2) / 2, y2


FAR_SLOPE = 30.0
"""An edge flatter than this many degrees from horizontal that lies above a fighter's
feet in the image is the far fence."""


def is_far(seg: Segment, point: tuple[float, float]) -> bool:
    """Whether `seg` is the far fence as seen from `point`: shallow, and above the feet."""
    x0, y0, x1, y1 = seg
    angle = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    angle = min(angle, 180 - angle)
    return bool(angle < FAR_SLOPE and max(y0, y1) < point[1])


def fence_distance(
    point: tuple[float, float], edges: Sequence[Segment], ky: float = 1.0
) -> tuple[float, int]:
    """(distance in pixels to the nearest edge, its index), or (inf, -1) with no edges."""
    if not edges:
        return float("inf"), -1
    d = [point_segment(point[0], point[1], e, ky) for e in edges]
    k = int(np.argmin(d))
    return d[k], k


def cage_call(
    red: tuple[float, float],
    blue: tuple[float, float],
    body_h: float,
    edges: Sequence[Segment],
    near: float,
    margin: float,
    ky: float = 1.0,
    far: bool = True,
) -> str | None:
    """ "red-pressured", "blue-pressured", "neutral", or None (abstain).

    `red` and `blue` are feet points in pixels and `body_h` the mean box height;
    the caller has already checked both are standing apart at striking range. `ky`
    stretches vertical distances (perspective correction); with `far` False, the far
    fence (`is_far`) can't make a call: a fighter near only the far fence is neutral.
    """
    if not edges or body_h <= 0:
        return None
    ok_r = [e for e in edges if far or not is_far(e, red)]
    ok_b = [e for e in edges if far or not is_far(e, blue)]
    dr, er = fence_distance(red, ok_r, ky)
    db, eb = fence_distance(blue, ok_b, ky)
    dr, db = dr / body_h, db / body_h
    if min(dr, db) > near:
        return "neutral"
    if dr <= db:
        other = point_segment(blue[0], blue[1], ok_r[er], ky) / body_h
        return "red-pressured" if other - dr >= margin else None
    other = point_segment(red[0], red[1], ok_b[eb], ky) / body_h
    return "blue-pressured" if other - db >= margin else None
