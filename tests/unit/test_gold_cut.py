"""Cutting silver-set clips: frames line up with tracked boxes, corners drawn in."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from cornerman.gold.cut import clip_frames, cut_fight, draw  # noqa: E402

FPS = 30
STEP = 3


def _video(path: Path, n: int) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (320, 240))
    for _ in range(n):
        w.write(np.full((240, 320, 3), 128, np.uint8))
    w.release()


def test_clip_frames_are_tracked_frames_in_range() -> None:
    frames = pd.DataFrame({"frame": np.arange(0, 90, STEP), "t": np.arange(0, 90, STEP) / FPS})
    clips = pd.DataFrame({"clip_id": ["a", "b"], "t0": [0.0, 1.5], "t1": [1.0, 2.5]})
    got = clip_frames(frames, clips)
    assert got["a"] == list(range(0, 30, STEP))
    assert got["b"] == list(range(45, 75, STEP))


def test_draw_uses_corner_colours() -> None:
    img = np.zeros((100, 100, 3), np.uint8)
    boxes = pd.DataFrame({"role": ["red", "blue"], "x1": [10.0, 50.0], "y1": [10.0, 10.0],
                          "x2": [40.0, 90.0], "y2": [90.0, 90.0]})  # fmt: skip
    out = draw(img, boxes)
    assert tuple(out[50, 10]) == (0, 0, 255)  # left edge of the red box (BGR)
    assert tuple(out[50, 50]) == (255, 0, 0)
    assert img.sum() == 0  # input untouched


def test_cut_fight_writes_one_clip_per_row(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    _video(video, 120)
    frames = pd.DataFrame({"frame": np.arange(0, 120, STEP), "t": np.arange(0, 120, STEP) / FPS})
    boxes = pd.DataFrame({"frame": [0, 3], "role": ["red", "red"], "x1": 10.0, "y1": 10.0,
                          "x2": 50.0, "y2": 100.0})  # fmt: skip
    clips = pd.DataFrame({"clip_id": ["c1", "c2"], "t0": [0.0, 2.0], "t1": [1.0, 3.0]})
    paths = cut_fight(video, STEP, frames, boxes, clips, tmp_path / "out", fps=10)
    assert [p.name for p in paths] == ["c1.mp4", "c2.mp4"]
    cap = cv2.VideoCapture(str(paths[0]))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert n == 10 and h == 480


def test_overlapping_clips_both_get_written(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    _video(video, 90)
    frames = pd.DataFrame({"frame": np.arange(0, 90, STEP), "t": np.arange(0, 90, STEP) / FPS})
    boxes = pd.DataFrame(columns=["frame", "role", "x1", "y1", "x2", "y2"])
    clips = pd.DataFrame({"clip_id": ["a", "b"], "t0": [0.0, 0.9], "t1": [1.0, 1.9]})
    paths = cut_fight(video, STEP, frames, boxes, clips, tmp_path / "out", fps=10)
    assert sorted(p.name for p in paths) == ["a.mp4", "b.mp4"]
