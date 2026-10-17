"""Perception logic that runs without models: shots, colour features, identity, metrics."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2")

from cornerman.eval.perception_metrics import fight_metrics, role_switches
from cornerman.perception.appearance import N_FEATURES, box_features
from cornerman.perception.identity import assign_roles, cluster, live_shots, summarise
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


def _scene(seed: int = 0) -> pd.DataFrame:
    """Three people over 4 shots; tracks restart every shot and break mid-shot.

    Fighters stay about one box height apart; the referee stands about three away.
    Each person has a distinct colour signature plus noise.
    """
    rng = np.random.default_rng(seed)
    signature = {p: np.eye(N_FEATURES)[i * 10] for i, p in enumerate(["A", "B", "R"])}
    offset = {"A": 0.0, "B": 100.0, "R": 400.0}
    rows: list[dict[str, object]] = []
    track = 0
    for shot in range(4):
        for person in ("A", "B", "R"):
            for piece in range(2):  # each person's shot track breaks in two
                track += 1
                for f in range(shot * 100 + piece * 50, shot * 100 + piece * 50 + 50):
                    x = 200 + offset[person] + 10 * np.sin(f / 7)
                    noisy = np.asarray(
                        np.abs(signature[person] + rng.normal(0, 0.05, N_FEATURES)),
                        dtype=np.float64,
                    )
                    feat = [float(v) for v in noisy / np.linalg.norm(noisy)]
                    row: dict[str, object] = {
                        "frame": f, "t": f / 10, "shot_id": shot, "track_id": track,
                        "x1": x, "y1": 100.0, "x2": x + 60, "y2": 300.0, "score": 0.9,
                        "truth": person,
                    }  # fmt: skip
                    row.update({f"app_{i}": v for i, v in enumerate(feat)})
                    rows.append(row)
    return pd.DataFrame(rows)


def test_identity_recovers_people_across_cuts() -> None:
    tracks = _scene()
    roles = assign_roles(tracks.drop(columns="truth"), min_frames=5, max_dist=0.8)
    truth = tracks.groupby("track_id")["truth"].first()
    merged = roles.set_index("track_id").join(truth)
    # Each true person maps to exactly one role, and the referee is named correctly.
    mapping = merged.groupby("truth")["role"].agg(set)
    assert all(len(s) == 1 for s in mapping)
    assert mapping["R"] == {"referee"}
    assert {next(iter(mapping["A"])), next(iter(mapping["B"]))} == {"fighter_a", "fighter_b"}
    assert mapping["A"] == {"fighter_a"}  # A starts on the left


def test_cannot_link_keeps_lookalikes_apart() -> None:
    """Both fighters wear the same colours; co-occurrence still forces two groups."""
    tracks = _scene()
    cols = [c for c in tracks.columns if c.startswith("app_")]
    b_rows = tracks["truth"] == "B"
    tracks.loc[b_rows, cols] = tracks.loc[tracks["truth"] == "A", cols].to_numpy()
    tl = summarise(tracks, min_frames=5)
    labels = cluster(tl, k=3, max_dist=0.8)
    for i, tid in enumerate(tl.ids):
        for j in tl.conflicts[i]:
            assert labels[i] != labels[j] or labels[i] == -1, (tid, tl.ids[j])


def test_far_away_appearance_is_other() -> None:
    tracks = _scene()
    stray = tracks[tracks["track_id"] == 1].copy()
    stray["track_id"] = 999
    stray["frame"] += 10_000
    cols = [c for c in tracks.columns if c.startswith("app_")]
    stray[cols] = np.eye(N_FEATURES)[N_FEATURES - 1]
    roles = assign_roles(
        pd.concat([tracks, stray]).drop(columns="truth"), min_frames=5, max_dist=0.8
    )
    assert roles.set_index("track_id").loc[999, "role"] == "other"


def test_live_shots_and_metrics() -> None:
    tracks = _scene().drop(columns="truth")
    frames = pd.DataFrame({"frame": np.arange(500), "shot_id": np.repeat([0, 1, 2, 3, 4], 100)})
    frames["t"] = frames["frame"] / 10
    shots = pd.DataFrame(
        {"shot_id": range(5), "start": [0.0, 10, 20, 30, 40], "end": [10.0, 20, 30, 40, 50]}
    )
    live = live_shots(tracks, shots, frames, min_frac=0.5)
    assert live["live"].tolist() == [True, True, True, True, False]

    roles = assign_roles(tracks, min_frames=5, max_dist=0.8)
    boxes = fighter_boxes(tracks, roles, live)
    assert set(boxes["role"]) == {"fighter_a", "fighter_b"}
    assert not boxes.duplicated(["frame", "role"]).any()
    assert role_switches(boxes) == 0

    poses = pd.DataFrame({"conf": [0.5, 0.7, 0.9]})
    m = fight_metrics(frames, tracks, roles, live, poses, step_seconds=0.1)
    assert m["live_frac"] == pytest.approx(0.8)
    assert m["both_fighters_frac"] == pytest.approx(1.0)
    assert m["role_switches_per_min"] == 0
    assert m["pose_conf_median"] == pytest.approx(0.7)


def test_role_switch_is_counted() -> None:
    a = [(0, 0), (1, 0), (100, 0), (101, 0)]  # A jumps to where B was at frame 2
    b = [(100, 0), (101, 0), (2, 0), (3, 0)]
    rows = [
        {"frame": f, "role": r, "x1": x, "y1": y, "x2": x + 1, "y2": y + 1}
        for r, pts in (("fighter_a", a), ("fighter_b", b))
        for f, (x, y) in enumerate(pts)
    ]
    assert role_switches(pd.DataFrame(rows)) == 1
