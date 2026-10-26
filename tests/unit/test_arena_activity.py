"""Cage mask and the fighting (not walk-out or break) filter."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2")

from cornerman.perception.activity import fighting_frames
from cornerman.perception.arena import mat_colour, mat_hull, on_mat, parse_clock, shot_colour

H, W = 360, 640
CANVAS = (235, 200, 140)  # light blue, BGR


def _frame(canvas: tuple[int, int, int] = CANVAS) -> np.ndarray:
    f = np.full((H, W, 3), 25, dtype=np.uint8)  # dark arena
    f[150:330, 60:580] = canvas
    f[320:360, 150:490] = (40, 40, 40)  # broadcast score bar
    return f


def test_mat_colour_is_the_canvas_not_the_dark_or_the_score_bar() -> None:
    lab = mat_colour([_frame()] * 3)
    assert lab[0] > 180  # bright
    assert np.hypot(lab[1] - 128, lab[2] - 128) > 10  # coloured


def test_feet_on_canvas_are_kept_and_outside_are_dropped() -> None:
    f = _frame()
    hull = mat_hull(f, mat_colour([f]), 18, 0.05)
    boxes = np.array(
        [
            [200, 100, 260, 250],  # feet on the canvas
            [600, 20, 630, 120],  # spectator beside and above the canvas
            [300, 200, 380, 359],  # feet cut off by the frame bottom, over the canvas
            [0, 200, 40, 359],  # cut off, but beside the canvas
        ],
        dtype=float,
    )
    assert on_mat(hull, boxes, 0.03 * H, H).tolist() == [True, False, True, False]


def test_no_canvas_keeps_nobody() -> None:
    f = np.full((H, W, 3), 25, dtype=np.uint8)
    colour = mat_colour([_frame()])
    assert mat_hull(f, colour, 18, 0.05) is None
    assert not on_mat(None, np.array([[0, 0, 10, 10]], dtype=float), 5, H).any()


def test_shot_colour_adapts_to_a_camera_but_not_to_a_close_up() -> None:
    fight = mat_colour([_frame()])
    darker_camera = mat_colour([_frame((200, 150, 90))])
    assert np.allclose(shot_colour([_frame((200, 150, 90))], fight, 45), darker_camera)
    red_close_up = np.full((H, W, 3), (40, 40, 230), dtype=np.uint8)
    assert np.allclose(shot_colour([red_close_up], fight, 45), fight)


def _activity(speeds: list[float], fps: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two fighters moving at the given speed (box heights per second), per second of video."""
    per_second = int(fps)
    n = len(speeds) * per_second
    t = np.arange(n) / fps
    v = np.repeat(speeds, per_second)
    x = np.cumsum(v * 100 / fps)  # box height is 100 px
    rows = [
        {"frame": i, "t": t[i], "role": r, "x1": x[i] + off, "y1": 0.0,
         "x2": x[i] + off + 40, "y2": 100.0}
        for i in range(n) for r, off in (("fighter_a", 0), ("fighter_b", 200))
    ]  # fmt: skip
    frames = pd.DataFrame({"frame": np.arange(n), "t": t, "shot_id": 0})
    return frames, pd.DataFrame(rows)


def test_rounds_count_and_breaks_do_not() -> None:
    # 60 s round, 30 s break, 60 s round.
    frames, boxes = _activity([0.4] * 60 + [0.05] * 30 + [0.4] * 60)
    out = fighting_frames(frames, boxes, {0}, 10, 0.2, 8, 15)
    secs = out.groupby(out["t"].astype(int))["fighting"].mean()
    assert secs.iloc[5:55].all() and secs.iloc[95:145].all()
    assert not secs.iloc[66:84].any()
    assert out["fighting"].sum() / 10 == pytest.approx(120, abs=10)


def test_short_pause_is_filled_and_short_burst_is_dropped() -> None:
    frames, boxes = _activity([0.4] * 40 + [0.05] * 6 + [0.4] * 40 + [0.05] * 40 + [0.4] * 5)
    out = fighting_frames(frames, boxes, {0}, 3, 0.2, 8, 15)
    secs = out.groupby(out["t"].astype(int))["fighting"].mean()
    assert secs.iloc[40:46].all()  # a 6 s pause inside a round still counts
    assert not secs.iloc[126:].any()  # a 5 s burst during a break doesn't


def test_not_live_shots_never_count() -> None:
    frames, boxes = _activity([0.4] * 30)
    out = fighting_frames(frames, boxes, set(), 10, 0.2, 8, 15)
    assert not out["fighting"].any()


def test_kit_colour_reads_gloves_shins_and_shorts() -> None:
    from cornerman.perception.kitcolour import sample_person

    frame = np.full((200, 100, 3), 30, dtype=np.uint8)  # dark kit everywhere
    frame[:, :] = (30, 30, 30)
    frame[90:110, 10:30] = (0, 0, 220)  # red glove around the wrist
    frame[150:170, 40:60] = (220, 60, 0)  # blue shin guard
    kps = np.zeros((17, 2))
    kps[9] = (20, 100)  # left wrist
    kps[10] = (90, 190)  # right wrist, over dark
    kps[13], kps[15] = (50, 140), (50, 180)  # knee and ankle: shin at (50, 160)
    kps[11], kps[12] = (40, 100), (60, 100)
    scores = np.full(17, 0.9)
    scores[16] = 0.1  # right ankle uncertain: no right shin sample
    got = sample_person(frame, kps, scores, box_h=200, canvas_lab=None, min_conf=0.3)
    parts = [p for p, *_ in got]
    assert parts.count("glove") == 2 and parts.count("shin") == 1 and "shorts" in parts
    glove_red = next(g for g in got if g[0] == "glove" and g[2] > 160)  # high a = red
    assert glove_red[4] >= 5
    shin = next(g for g in got if g[0] == "shin")
    assert shin[3] < 100  # low b = blue


@pytest.mark.parametrize(
    ("texts", "want"),
    [
        (["01:38", "ROUND 1", "FIGHTWEEK", "IMMAF"], (98.0, 1)),
        (["CEN", "2.43 RD2"], (163.0, 2)),
        (["R3", "00:05"], (5.0, 3)),
        (["2ND", "0:59"], (59.0, 2)),
        (["IMMAF"], (None, None)),
        (["61.2 kg", "12:30"], (None, None)),  # a weight, then a time of day past 10 minutes
        (["03:75", "03:00"], (180.0, None)),  # seconds over 59 can't be a clock
        (["FIN", "0046", "ROUND 2"], (46.0, 2)),  # OCR dropped the colon
        (["246"], (166.0, None)),
        (["2022", "IMMAF"], (None, None)),  # a year is not a clock
    ],
)
def test_parse_clock(texts: list[str], want: tuple[float | None, int | None]) -> None:
    assert parse_clock(texts) == want
