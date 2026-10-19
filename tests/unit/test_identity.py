"""Identity (round 3): people from crop appearance, fighters, roles per crop, confidence."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")

from cornerman.perception import crop_features as cf
from cornerman.perception.identity import (
    IdentityParams,
    Probe,
    Tracklets,
    assign,
    behaviour_labels,
    box_roles,
    cluster,
    confidence,
    label_free_checks,
    per_crop_roles,
    tracklet_roles,
    train_probe,
)

DIM = 384
OFFSET = {"R": 0.0, "B": 100.0, "X": 400.0}


def _people(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    v = {p: rng.normal(size=DIM) for p in OFFSET}
    return {p: x / np.linalg.norm(x) for p, x in v.items()}


def _scene(
    swap_track: bool = False, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Probe]:
    """Red (R), blue (B) and the referee (X) over 4 shots; tracks break mid-shot.

    Returns live tracks, the crop table (one crop every 5 frames per track), features,
    hue histograms and a probe that scores R and B as fighters. With `swap_track`,
    in shot 0 the tracker swaps R and B halfway through their tracklets (a tangle).
    """
    rng = np.random.default_rng(seed + 1)
    vec = _people(seed)
    rows: list[dict[str, object]] = []
    crops: list[dict[str, object]] = []
    feats: list[np.ndarray] = []
    track = 0
    for shot in range(4):
        for person in OFFSET:
            for piece in range(2):
                track += 1
                for f in range(shot * 100 + piece * 50, shot * 100 + piece * 50 + 50):
                    who = person
                    if swap_track and shot == 0 and piece == 1 and person in "RB":
                        who = {"R": "B", "B": "R"}[person]
                    x = 200 + OFFSET[who] + 10 * np.sin(f / 7)
                    rows.append(
                        {"frame": f, "t": f / 10, "shot_id": shot, "track_id": track,
                         "x1": x, "y1": 100.0, "x2": x + 60, "y2": 300.0, "score": 0.9,
                         "truth": who}
                    )  # fmt: skip
                    if f % 5 == 0:
                        crops.append(
                            {"track_id": track, "frame": f, "merged": False, "has_mask": True,
                             "angle": 10.0, "red": 50 if who == "R" else 0,
                             "blue": 50 if who == "B" else 0, "npx": 200, "truth": who}
                        )  # fmt: skip
                        noisy: np.ndarray = vec[who] + 0.3 * rng.normal(size=DIM) / np.sqrt(DIM)
                        feats.append(noisy / float(np.linalg.norm(noisy)))
    tracks = pd.DataFrame(rows)
    table = pd.DataFrame(crops)
    probe = Probe(w=4 * (vec["R"] + vec["B"] - vec["X"]), b=0.0)
    return tracks, table, np.array(feats), np.zeros((len(table), 18), np.float32), probe


def _accuracy(crop_roles: pd.DataFrame, table: pd.DataFrame) -> float:
    name = {"R": "red", "B": "blue", "X": "referee"}
    return float((crop_roles["role"].to_numpy() == table["truth"].map(name).to_numpy()).mean())


def test_identity_finds_corners_and_referee_across_cuts() -> None:
    tracks, table, feats, hue, probe = _scene()
    result = assign(tracks, table.drop(columns="truth"), feats, hue, probe, IdentityParams())
    assert _accuracy(result.crop_roles, table) == 1.0
    assert result.stats["fighter_prob_min"] > 0.9
    assert result.stats["colour_margin"] > 0.4


def test_identity_follows_an_id_switch_inside_a_tracklet() -> None:
    """The tracker swaps the fighters halfway through two tracklets; roles per crop follow."""
    tracks, table, feats, hue, probe = _scene(swap_track=True)
    result = assign(tracks, table.drop(columns="truth"), feats, hue, probe, IdentityParams())
    assert _accuracy(result.crop_roles, table) > 0.95
    swapped = table["frame"].between(50, 99) & table["truth"].isin(["R", "B"])
    assert _accuracy(result.crop_roles[swapped.to_numpy()], table[swapped]) > 0.9


def test_per_crop_roles_ignore_one_odd_crop() -> None:
    """One crop that looks like blue inside a red tracklet stays red (switch cost)."""
    red, blue, ref = np.eye(3)
    emb = np.array([red, red, 0.55 * red + 0.84 * blue, red, red])
    crops = pd.DataFrame({"track_id": [1] * 5, "frame": [0, 10, 20, 30, 40]})
    out = per_crop_roles(crops, emb, {"red": red, "blue": blue}, {7: ref}, 7, switch=2.0)
    assert out["role"].tolist() == ["red"] * 5
    out = per_crop_roles(crops, emb, {"red": red, "blue": blue}, {7: ref}, 7, switch=0.0)
    assert out["role"].tolist()[2] == "blue"


def test_other_crops_near_the_referee_are_named_referee() -> None:
    red, blue, ref = np.eye(3)
    crops = pd.DataFrame({"track_id": [1, 2], "frame": [0, 0]})
    out = per_crop_roles(crops, np.array([ref, ref]), {"red": red, "blue": blue}, {7: ref}, 7, 2.0)
    assert out["role"].tolist() == ["referee", "referee"]
    out = per_crop_roles(
        crops, np.array([ref, ref]), {"red": red, "blue": blue}, {7: ref}, None, 2.0
    )
    assert out["role"].tolist() == ["other", "other"]


def test_cannot_link_keeps_lookalikes_apart() -> None:
    """Two tracklets on screen together never share a group, even if they look the same."""
    feats = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.01], [0.0, 1.0]])
    tl = Tracklets(
        ids=np.arange(4),
        features=feats,
        weights=np.array([50.0, 50.0, 40.0, 30.0]),
        conflicts=[{1}, {0}, {3}, {2}],
    )
    labels = cluster(tl, k=3, max_dist=10.0)
    assert labels[0] != labels[1]
    assert labels[2] != labels[3]


def test_box_roles_take_the_nearest_crop_of_the_tracklet() -> None:
    tracks = pd.DataFrame({"frame": [0, 1, 2, 3, 4, 0], "track_id": [1, 1, 1, 1, 1, 2]})
    crops = pd.DataFrame({"frame": [0, 4], "track_id": [1, 1], "role": ["red", "blue"]})
    boxes = box_roles(tracks, crops)
    red = boxes[boxes["track_id"] == 1].set_index("frame")["role"]
    assert red.tolist() == ["red", "red", "red", "blue", "blue"]
    assert boxes.set_index("track_id").loc[2, "role"] == "other"  # no crop at all
    assert len(box_roles(tracks, crops, fill=False)) == 5
    summary = tracklet_roles(boxes).set_index("track_id")
    assert summary.loc[1, "role"] == "red"
    assert summary.loc[1, "distance"] == pytest.approx(0.4)


def test_label_free_checks_catch_one_corner_on_two_people() -> None:
    tracks, table, feats, hue, probe = _scene()
    result = assign(tracks, table.drop(columns="truth"), feats, hue, probe, IdentityParams())
    fighting = set(tracks["frame"].tolist())
    good = label_free_checks(
        tracks, box_roles(tracks, result.crop_roles), result.crop_roles, fighting
    )
    assert good["dup"] == 0.0
    assert good["both"] == pytest.approx(1.0)
    assert good["closest"] == pytest.approx(1.0)  # the fighters are always closest
    # Call the referee red too: one corner on two boxes in every frame.
    bad_roles = result.crop_roles.replace({"role": {"referee": "red"}})
    bad = label_free_checks(tracks, box_roles(tracks, bad_roles), bad_roles, fighting)
    assert bad["dup"] == pytest.approx(1.0)
    assert confidence({**result.stats, **good}) > confidence({**result.stats, **bad})


def test_confidence_is_bounded() -> None:
    best = {"dup": 0.0, "fighter_prob_min": 0.9, "rb_share": 0.8, "closest": 0.7}
    worst = {"dup": 0.5, "fighter_prob_min": 0.1, "rb_share": 0.0, "closest": 0.0}
    assert confidence(best) == 1.0
    assert confidence(worst) == 0.0
    assert 0.0 < confidence({**best, "closest": float("nan")}) < 1.0


def test_behaviour_labels() -> None:
    """On the ground or kicking = fighter; a tracklet that never moves = official."""
    frames = np.arange(40)
    tracks = pd.DataFrame(
        {"frame": np.tile(frames, 2), "t": np.tile(frames / 10, 2),
         "track_id": np.repeat([1, 2], 40),
         "x1": np.r_[frames * 5.0, np.full(40, 500.0)], "y1": 0.0,
         "x2": np.r_[frames * 5.0 + 50, np.full(40, 550.0)], "y2": 200.0}
    )  # fmt: skip
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    kp[[13, 14], 1] = 150.0  # knees
    kp[[15, 16], 1] = 190.0  # ankles below knees
    kick = kp.copy()
    kick[15, 1] = 100.0  # left ankle above the right knee
    poses = pd.DataFrame(
        [{"frame": f, "track_id": t, "kp": k, "x": 0.0, "y": float(p[k, 1]), "conf": 0.9}
         for f, t, p in ((0, 1, kick), (1, 1, kp), (0, 2, kp)) for k in range(17)]
    )  # fmt: skip
    crops = pd.DataFrame(
        {"track_id": [1, 1, 1, 2], "frame": [0, 1, 2, 0], "angle": [5.0, 80.0, 5.0, 5.0]}
    )
    assert behaviour_labels(tracks, poses, crops).tolist() == [1, 1, -1, 0]


def test_train_probe_separates_classes() -> None:
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(1, 0.3, (100, 5)), rng.normal(-1, 0.3, (30, 5))]
    y = np.r_[np.ones(100, int), np.zeros(30, int)]
    probe = train_probe(x, y)
    assert (probe.score(x[:100]) > 0.5).mean() > 0.95
    assert (probe.score(x[100:]) < 0.5).mean() > 0.95


def test_probe_round_trips(tmp_path: pytest.TempPathFactory) -> None:
    from pathlib import Path

    p = Probe(w=np.array([1.0, -2.0]), b=0.5)
    path = Path(str(tmp_path)) / "probe.npz"
    p.save(path)
    q = Probe.load(path)
    assert np.allclose(q.w, p.w) and q.b == 0.5


def test_crop_geometry_and_colour() -> None:
    upright = np.zeros((17, 3))
    upright[:, 2] = 0.9
    upright[[5, 6], :2] = [[40, 30], [60, 30]]  # shoulders
    upright[[11, 12], :2] = [[42, 90], [58, 90]]  # hips
    assert cf.torso_angle(upright) == pytest.approx(0.0, abs=1e-6)
    lying = upright.copy()
    lying[[5, 6], :2] = [[10, 60], [10, 70]]
    lying[[11, 12], :2] = [[70, 60], [70, 70]]
    assert cf.torso_angle(lying) == pytest.approx(90.0)
    assert np.isnan(cf.torso_angle(None))

    img = np.zeros((160, 80, 3), np.uint8)
    img[:80] = (0, 0, 200)  # red top half (BGR)
    img[80:] = (235, 200, 140)  # light canvas blue
    mask = cf.body_mask(upright, (0.0, 0.0, 80.0, 160.0), (160, 80))
    assert mask.sum() > 0
    assert cf.body_mask(None, (0.0, 0.0, 80.0, 160.0), (160, 80)).sum() == 0
    red, blue, n = cf.colour_counts(img, np.ones((160, 80), np.uint8))
    assert red == 80 * 80 and blue == 0 and n == 160 * 80
    x, grid = cf.model_input(img, mask)
    assert x.shape == (3, 224, 224) and grid.shape == (256,)
