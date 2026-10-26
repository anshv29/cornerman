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


RUNNING, STOPPED, UNKNOWN = "running", "stopped", "unknown"


def clock_state(
    clock: pd.DataFrame, max_dt_s: float, tol_s: float, break_max_s: float
) -> pd.Series:
    """running, stopped or unknown for each clock reading (arena stage's clock table).

    A reading is running when a readable neighbour (the one before or after, at
    most `max_dt_s` away) differs from it by the time between them, give or take
    `tol_s`, in either direction (bars count down or up). It is stopped when a
    neighbour at least 1.5 s away shows the same time: between rounds, before the
    fight and after it the clock is frozen. Otherwise (no readable neighbour, or
    a change that fits neither, usually an OCR slip) it is unknown.

    Some bars count the rest between rounds down in the same place: the clock
    jumps up from the end of a round to about 1:00 and runs down again. So a
    reading that jumps up to at most `break_max_s`, and the readings after it that
    keep following that countdown, are stopped. A single misread can only mark
    itself, since the next true reading won't follow its countdown.
    """
    c = clock.sort_values("t").reset_index(drop=True)
    known = c[c["clock_s"].notna()]
    t = known["t"].to_numpy(dtype=float)
    v = known["clock_s"].to_numpy(dtype=float)
    state = pd.Series(UNKNOWN, index=c.index, dtype=object)
    for k, i in enumerate(known.index):
        votes: set[str] = set()
        for j in (k - 1, k + 1):
            if not 0 <= j < len(t):
                continue
            dt = abs(t[j] - t[k])
            if dt > max_dt_s:
                continue
            dc = abs(v[j] - v[k])
            if abs(dc - dt) <= tol_s:
                votes.add(RUNNING)
            elif dc == 0 and dt >= 1.5:
                votes.add(STOPPED)
        if votes:
            state[i] = RUNNING if RUNNING in votes else STOPPED
    k = 1
    while k < len(t):
        if v[k] <= break_max_s and v[k] - v[k - 1] > t[k] - t[k - 1] + tol_s:
            k0 = k
            while k < len(t) and abs(v[k] - (v[k0] - (t[k] - t[k0]))) <= tol_s:
                state[known.index[k]] = STOPPED
                k += 1
        else:
            k += 1
    return pd.Series(state.to_numpy(), index=c["frame"].to_numpy(), name="clock")


def fighting_v2(
    motion: pd.DataFrame,
    frames: pd.DataFrame,
    live_shots: set[int],
    clock: pd.Series,
    people: pd.Series,
    reach_s: float,
    margin_s: float,
    max_people: int,
    crowd_window_s: float,
    min_gap_s: float,
    min_run_s: float,
) -> pd.DataFrame:
    """Fighting time from the round clock where it can be read, motion and a crowd check
    where it can't.

    `motion` is the motion-only table (`fighting_frames`), `clock` the state per
    clock reading (`clock_state`, indexed by frame), `people` the number of
    people on the canvas per sampled frame. Per frame:
    1. clock running (nearest reading within `reach_s`): fighting, live shot or not
       (a long clinch or ground exchange merges the two fighters into one box, so its
       shot fails the two-people live test; the round is still on);
    2. clock stopped: not fighting;
    3. clock unknown: motion says fighting, at most `max_people` on the canvas
       (median over `crowd_window_s`; walk-outs and celebrations bring cornermen
       and officials in), and within `margin_s` of the first and last running
       reading if the fight has any (walk-out before, celebration after).
    Then, as for motion, short gaps are filled and short runs dropped. `live` says
    whether the frame's shot is live: only live shots have pose and roles, so anything
    per fighter needs `fighting & live`.
    """
    f = motion.sort_values("frame").reset_index(drop=True)
    dt = float(np.median(np.diff(f["t"]))) if len(f) > 1 else 0.1
    readings = pd.DataFrame({"frame": clock.index.to_numpy(dtype=np.int64), "clock": clock.values})
    readings = readings.sort_values("frame")
    step = float(np.median(np.diff(f["frame"]))) if len(f) > 1 else 1.0
    tol = round(reach_s / dt * step)
    near = pd.merge_asof(f[["frame"]], readings, on="frame", direction="nearest", tolerance=tol)[
        "clock"
    ].fillna(UNKNOWN)
    state = near.to_numpy(dtype=object)
    shot = f["frame"].map(frames.set_index("frame")["shot_id"]).to_numpy()
    live = pd.Series(shot).isin(live_shots).to_numpy()
    window = max(1, round(crowd_window_s / dt))
    crowd = f["frame"].map(people).fillna(0).rolling(window, center=True, min_periods=1).median()
    run_t = f["t"].to_numpy()[state == RUNNING]
    inside = (
        (f["t"] >= run_t.min() - margin_s) & (f["t"] <= run_t.max() + margin_s)
        if len(run_t)
        else pd.Series(True, index=f.index)
    ).to_numpy()
    fallback = f["fighting"].to_numpy() & (crowd.to_numpy() <= max_people) & inside
    fighting = np.where(state == RUNNING, True, np.where(state == STOPPED, False, fallback))
    fighting = fighting.astype(bool)
    for target, max_len in ((False, min_gap_s), (True, min_run_s)):
        runs = _runs(fighting)
        for i, (s, e, v) in enumerate(runs):
            inner = 0 < i < len(runs) - 1
            if v == target and (e - s) * dt < max_len and (target or inner):
                fighting[s:e] = not target
    return pd.DataFrame(
        {
            "frame": f["frame"],
            "t": f["t"],
            "speed": f["speed"],
            "clock": state.astype(str),
            "motion": f["fighting"].to_numpy(dtype=bool),
            "fighting": fighting,
            "live": live,
        }
    )
