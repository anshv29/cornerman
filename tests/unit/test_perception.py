"""Perception logic that runs without models: shots, colour features, live shots, metrics."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2")

from cornerman.contracts.perception import fighter_boxes
from cornerman.eval.perception_metrics import fight_metrics, role_switches
from cornerman.perception.appearance import N_FEATURES, box_features
from cornerman.perception.identity import corner_fighters, live_shots
from cornerman.perception.shots import shot_of_frame, shots_from_probs
from cornerman.perception.video import frame_step


def test_shots_from_probs_splits_at_cuts() -> None:
    probs = np.array([0.0, 0.1, 0.9, 0.0, 0.0, 0.8, 0.7, 0.0])
    shots = shots_from_probs(probs, fps=2.0, threshold=0.5)
    assert shots[["first_frame", "last_frame"]].values.tolist() == [[0, 1], [3, 4], [7, 7]]
    assert shots["start"].tolist() == [0.0, 1.5, 3.5]
    assert shot_of_frame(shots, np.array([0, 2, 4, 6, 7])).tolist() == [0, -1, 1, -1, 2]


def test_frame_step() -> None:
    assert frame_step(50, 10) == 5
    assert frame_step(29.97, 10) == 3
    assert frame_step(5, 10) == 1


def _person(colour_top: tuple[int, int, int], colour_shorts: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((100, 50, 3), dtype=np.uint8)
    frame[:45] = colour_top
    frame[45:] = colour_shorts
    return frame


def test_box_features_separate_shorts_colours() -> None:
    red = box_features(_person((150, 170, 200), (0, 0, 220)), (0, 0, 50, 100))
    red2 = box_features(_person((150, 170, 200), (10, 10, 200)), (0, 0, 50, 100))
    blue = box_features(_person((150, 170, 200), (220, 0, 0)), (0, 0, 50, 100))
    assert red.shape == (N_FEATURES,)
    assert np.isclose(np.linalg.norm(red), 1.0)
    assert np.linalg.norm(red - red2) < np.linalg.norm(red - blue)


def _scene() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Red, blue and referee over 4 shots, tracks breaking mid-shot, and their roles."""
    offset = {"red": 0.0, "blue": 100.0, "referee": 400.0}
    rows: list[dict[str, object]] = []
    roles: list[tuple[int, str]] = []
    track = 0
    for shot in range(4):
        for role, dx in offset.items():
            for piece in range(2):
                track += 1
                roles.append((track, role))
                for f in range(shot * 100 + piece * 50, shot * 100 + piece * 50 + 50):
                    x = 200 + dx + 10 * np.sin(f / 7)
                    rows.append(
                        {"frame": f, "t": f / 10, "shot_id": shot, "track_id": track,
                         "x1": x, "y1": 100.0, "x2": x + 60, "y2": 300.0, "score": 0.9}
                    )  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(roles, columns=["track_id", "role"])


def test_corner_fighters_follows_title_order() -> None:
    assert corner_fighters("F0003", "F0007") == {"red": "F0003", "blue": "F0007"}


def test_live_shots_and_metrics() -> None:
    tracks, roles = _scene()
    frames = pd.DataFrame({"frame": np.arange(500), "shot_id": np.repeat([0, 1, 2, 3, 4], 100)})
    frames["t"] = frames["frame"] / 10
    shots = pd.DataFrame(
        {"shot_id": range(5), "start": [0.0, 10, 20, 30, 40], "end": [10.0, 20, 30, 40, 50]}
    )
    live = live_shots(tracks, shots, frames, min_frac=0.5)
    assert live["live"].tolist() == [True, True, True, True, False]

    for r in (roles, roles.merge(tracks[["frame", "track_id"]])):  # per tracklet, per box
        boxes = fighter_boxes(tracks, r, live)
        assert set(boxes["role"]) == {"red", "blue"}
        assert not boxes.duplicated(["frame", "role"]).any()
        assert role_switches(boxes) == 0

    poses = pd.DataFrame(
        {"frame": [0, 0, 0], "track_id": [1, 1, 5], "conf": [0.5, 0.9, 0.1]}
    )  # track 1 is red, track 5 the referee
    m = fight_metrics(frames, tracks, roles, live, poses, step_seconds=0.1)
    assert m["live_frac"] == pytest.approx(0.8)
    assert m["both_fighters_frac"] == pytest.approx(1.0)
    assert m["role_switches_per_min"] == 0
    assert m["pose_conf_median"] == pytest.approx(0.7)  # the referee's pose is left out


def test_role_switch_is_counted() -> None:
    a = [(0, 0), (1, 0), (100, 0), (101, 0)]  # red jumps to where blue was at frame 2
    b = [(100, 0), (101, 0), (2, 0), (3, 0)]
    rows = [
        {"frame": f, "role": r, "x1": x, "y1": y, "x2": x + 1, "y2": y + 1}
        for r, pts in (("red", a), ("blue", b))
        for f, (x, y) in enumerate(pts)
    ]
    assert role_switches(pd.DataFrame(rows)) == 1


def test_prefetch_keeps_order_and_raises() -> None:
    from cornerman.perception.video import prefetch

    assert list(prefetch(iter(range(100)), depth=4)) == list(range(100))
    assert list(prefetch(iter(range(5)), depth=0)) == list(range(5))

    def broken() -> Iterator[int]:
        yield 1
        raise ValueError("decode failed")

    with pytest.raises(ValueError, match="decode failed"):
        list(prefetch(broken(), depth=2))


def test_ensure_video_links_a_staged_video(tmp_path: Path) -> None:
    import shutil

    from cornerman.config import PathsConfig, Settings
    from cornerman.perception.video import ensure_video, raw_dir

    staged = tmp_path / "input" / "cornerman-videos" / "ft_abc.mp4"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"video")
    settings = Settings(paths=PathsConfig(raw=tmp_path / "raw", video_source=tmp_path / "input"))
    path = ensure_video(settings, "ft_abc")
    assert path.read_bytes() == b"video"
    assert path.is_symlink()
    shutil.rmtree(raw_dir(settings, "ft_abc"))
    assert staged.exists()  # deleting raw/ never touches the read-only source
