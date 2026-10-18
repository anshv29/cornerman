"""Perception logic that runs without models: shots, colour features, identity, metrics."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2")

from cornerman.eval.perception_metrics import fight_metrics, role_switches
from cornerman.perception.appearance import N_FEATURES, box_features
from cornerman.perception.identity import (
    assign_roles,
    cluster,
    corner_features,
    corner_fighters,
    live_shots,
    summarise,
)
from cornerman.perception.shots import shot_of_frame, shots_from_probs
from cornerman.perception.stages import fighter_boxes
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


# Lab chroma (a, b) offsets from neutral per body part, for each synthetic person.
KIT = {
    "R": {"glove": (45, 25), "shin": (45, 25), "shorts": (0, 0)},  # red corner, black shorts
    "B": {"glove": (0, -45), "shin": (0, -45), "shorts": (0, 0)},  # blue corner
    "X": {"glove": (15, 20), "shin": (0, 0), "shorts": (0, 0)},  # referee: bare hands, black
}


def _scene(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three people over 4 shots; tracks restart every shot and break mid-shot.

    Returns tracks and the keypoint colour samples pose would have produced.
    The fighters stay about one box height apart; the referee stands further
    away and is on screen less, since the camera follows the fighters.
    """
    rng = np.random.default_rng(seed)
    offset = {"R": 0.0, "B": 100.0, "X": 400.0}
    rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    track = 0
    for shot in range(4):
        for person in ("R", "B", "X"):
            for piece in range(2):  # each person's shot track breaks in two
                track += 1
                # The referee steps out of shot for part of each piece.
                length = 35 if person == "X" else 50
                for f in range(shot * 100 + piece * 50, shot * 100 + piece * 50 + length):
                    x = 200 + offset[person] + 10 * np.sin(f / 7)
                    rows.append(
                        {
                            "frame": f, "t": f / 10, "shot_id": shot, "track_id": track,
                            "x1": x, "y1": 100.0, "x2": x + 60, "y2": 300.0, "score": 0.9,
                            "truth": person,
                        }
                    )  # fmt: skip
                    for part, (da, db) in KIT[person].items():
                        a, b = rng.normal([128 + da, 128 + db], 6)
                        samples.append(
                            {"frame": f, "track_id": track, "part": part, "L": 120.0,
                             "a": float(a), "b": float(b), "n": 20}
                        )  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(samples)


def test_identity_finds_corners_and_referee_across_cuts() -> None:
    tracks, samples = _scene()
    roles = assign_roles(tracks.drop(columns="truth"), samples, min_frames=5, max_dist=2.0)
    truth = tracks.groupby("track_id")["truth"].first()
    mapping = roles.set_index("track_id").join(truth).groupby("truth")["role"].agg(set)
    assert mapping.to_dict() == {"R": {"red"}, "B": {"blue"}, "X": {"referee"}}


def test_cannot_link_keeps_lookalikes_apart() -> None:
    """Both fighters wear the same colours; co-occurrence still forces two groups."""
    tracks, samples = _scene()
    owner = tracks.groupby("track_id")["truth"].first()
    is_b = samples["track_id"].map(owner) == "B"
    samples.loc[is_b, ["a", "b"]] = [173.0, 153.0]  # blue fighter now looks red too
    tl = summarise(tracks, corner_features(samples), min_frames=5)
    labels = cluster(tl, k=4, max_dist=2.0)
    for i, tid in enumerate(tl.ids):
        for j in tl.conflicts[i]:
            assert labels[i] != labels[j] or labels[i] == -1, (tid, tl.ids[j])


def test_someone_else_on_the_canvas_is_other() -> None:
    """A cornerman on the canvas in green and jeans gets his own group, named other."""
    tracks, samples = _scene()
    extra = tracks[tracks["track_id"] == 1].copy()
    extra["track_id"] = 999
    extra["x1"] += 600
    extra["x2"] += 600
    tracks = pd.concat([tracks, extra])
    kit = pd.DataFrame(
        [{"frame": f, "track_id": 999, "part": p, "L": 120.0, "a": a, "b": b, "n": 20}
         for f in extra["frame"] for p, (a, b) in
         (("glove", (143.0, 148.0)), ("shin", (125.0, 110.0)), ("shorts", (90.0, 140.0)))]
    )  # fmt: skip
    roles = assign_roles(
        tracks.drop(columns="truth"), pd.concat([samples, kit]), min_frames=5, max_dist=2.0
    )
    assert roles.set_index("track_id").loc[999, "role"] == "other"
    assert (roles.set_index("track_id").loc[[1, 2], "role"] == "red").all()


def test_tracklets_without_samples_are_other() -> None:
    tracks, samples = _scene()
    roles = assign_roles(tracks.drop(columns="truth"), samples[samples["track_id"] != 1], 5, 2.0)
    assert roles.set_index("track_id").loc[1, "role"] == "other"


def test_corner_fighters_follows_title_order() -> None:
    assert corner_fighters("F0003", "F0007") == {"red": "F0003", "blue": "F0007"}


def test_live_shots_and_metrics() -> None:
    tracks, samples = _scene()
    tracks = tracks.drop(columns="truth")
    frames = pd.DataFrame({"frame": np.arange(500), "shot_id": np.repeat([0, 1, 2, 3, 4], 100)})
    frames["t"] = frames["frame"] / 10
    shots = pd.DataFrame(
        {"shot_id": range(5), "start": [0.0, 10, 20, 30, 40], "end": [10.0, 20, 30, 40, 50]}
    )
    live = live_shots(tracks, shots, frames, min_frac=0.5)
    assert live["live"].tolist() == [True, True, True, True, False]

    roles = assign_roles(tracks, samples, min_frames=5, max_dist=2.0)
    boxes = fighter_boxes(tracks, roles, live)
    assert set(boxes["role"]) == {"red", "blue"}
    assert not boxes.duplicated(["frame", "role"]).any()
    assert role_switches(boxes) == 0

    fighter_tracks = roles.loc[roles["role"].isin(["red", "blue"]), "track_id"]
    ref_track = int(roles.loc[roles["role"] == "referee", "track_id"].iloc[0])
    poses = pd.DataFrame(
        {"frame": [0, 0, 0], "track_id": [int(fighter_tracks.iloc[0])] * 2 + [ref_track],
         "conf": [0.5, 0.9, 0.1]}
    )  # fmt: skip
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
