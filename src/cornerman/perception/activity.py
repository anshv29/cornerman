"""Which sampled frames show live fighting, as opposed to walk-outs, breaks and announcements.

The cue is motion. While a round is on, the two fighters keep moving: their
box centres travel a few tenths of a box height per second even in a slow
exchange. During the minute between rounds they sit on stools, during the
referee's instructions and the result announcement they stand still, and in
walk-outs they're outside the cage (dropped by the cage mask). On the two pilot
fights the median fighter speed was 0.3 to 0.6 box heights per second in rounds
and about 0.1 in breaks.

Steps:
1. per frame, the mean speed of the fighters seen in it (box heights per second);
2. a centred rolling median over `window_s`, so one still moment doesn't count;
3. fighting = median at or above `min_speed`, in a live shot;
4. fill non-fighting gaps shorter than `min_gap_s` (a stoppage, a clinch where
   the detector merges both fighters into one box) and drop fighting runs
   shorter than `min_run_s`.
The label-free check: a fight that goes the distance should come out near
3 rounds x 3 minutes.
"""

import numpy as np
import pandas as pd


def fighter_speed(boxes: pd.DataFrame, max_dt: float) -> pd.Series:
    """Mean fighter speed per frame, in box heights per second.

    `boxes` has one row per frame and fighter with frame, t, x1, y1, x2, y2.
    Steps longer than `max_dt` (a fighter missing for a while) aren't used.
    """
    b = boxes.assign(
        cx=(boxes["x1"] + boxes["x2"]) / 2,
        cy=(boxes["y1"] + boxes["y2"]) / 2,
        h=boxes["y2"] - boxes["y1"],
    ).sort_values(["role", "frame"])
    g = b.groupby("role")
    dt = g["t"].diff()
    dist = (g["cx"].diff() ** 2 + g["cy"].diff() ** 2) ** 0.5
    speed = (dist / b["h"] / dt).where(dt <= max_dt)
    return speed.groupby(b["frame"]).mean()


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """(start, end exclusive, value) for each run of equal values."""
    if len(mask) == 0:
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(mask)]])
    return [(int(s), int(e), bool(mask[s])) for s, e in zip(starts, ends, strict=True)]


def fighting_frames(
    frames: pd.DataFrame,
    boxes: pd.DataFrame,
    live_shots: set[int],
    window_s: float,
    min_speed: float,
    min_gap_s: float,
    min_run_s: float,
) -> pd.DataFrame:
    """frame, t, speed (smoothed) and fighting for every sampled frame."""
    f = frames.sort_values("frame").reset_index(drop=True)
    dt = float(np.median(np.diff(f["t"]))) if len(f) > 1 else 0.1
    speed = fighter_speed(boxes, max_dt=3 * dt).reindex(f["frame"]).to_numpy()
    window = max(1, round(window_s / dt))
    smooth = pd.Series(speed).rolling(window, center=True, min_periods=max(1, window // 4)).median()
    live = f["shot_id"].isin(live_shots).to_numpy()
    fighting = (smooth.to_numpy() >= min_speed) & live

    for target, max_len in ((False, min_gap_s), (True, min_run_s)):
        runs = _runs(fighting)
        for i, (s, e, v) in enumerate(runs):
            inner = 0 < i < len(runs) - 1  # only gaps between fighting runs get filled
            if v == target and (e - s) * dt < max_len and (target or inner):
                fighting[s:e] = not target
    return pd.DataFrame(
        {"frame": f["frame"], "t": f["t"], "speed": smooth.to_numpy(), "fighting": fighting}
    )
