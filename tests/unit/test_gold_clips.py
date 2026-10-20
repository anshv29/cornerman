"""Silver-set candidate clips: windows in fighting time, position proxy, stratified sample."""

import numpy as np
import pandas as pd

from cornerman.gold.clips import STRATA, sample, torso_angles, windows

STEP = 0.1


def _pose(frame: int, track: int, lying: bool) -> list[dict[str, object]]:
    """Shoulders above hips (standing) or level with them (lying)."""
    pts = {5: (40, 30), 6: (60, 30), 11: (42, 90), 12: (58, 90)}
    if lying:
        pts = {5: (10, 60), 6: (10, 70), 11: (70, 60), 12: (70, 70)}
    return [{"frame": frame, "track_id": track, "kp": k, "x": float(x), "y": float(y),
             "conf": 0.9} for k, (x, y) in pts.items()]  # fmt: skip


def _fight() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """12 s in two shots: 0-6 s apart, 6-9 s close, 9-12 s on the ground with one box."""
    n = 120
    t = np.arange(n) * STEP
    frames = pd.DataFrame({"frame": np.arange(n), "t": t, "shot_id": (t >= 6).astype(int)})
    fighting = pd.DataFrame({"frame": np.arange(n), "fighting": True})
    boxes: list[dict[str, object]] = []
    poses: list[dict[str, object]] = []
    for f, s in zip(range(n), t, strict=True):
        if s < 9:
            gap = 300.0 if s < 6 else 60.0
            for role, x, tid in (("red", 100.0, 1), ("blue", 100.0 + gap, 2)):
                boxes.append({"frame": f, "track_id": tid, "role": role,
                              "x1": x, "y1": 0.0, "x2": x + 80, "y2": 200.0})  # fmt: skip
                poses += _pose(f, tid, lying=False)
        else:
            boxes.append({"frame": f, "track_id": 3, "role": "red",
                          "x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 200.0})  # fmt: skip
            poses += _pose(f, 3, lying=True)
    return frames, fighting, pd.DataFrame(boxes), pd.DataFrame(poses)


def test_torso_angles() -> None:
    p = pd.DataFrame(_pose(0, 1, lying=False) + _pose(0, 2, lying=True))
    a = torso_angles(p).set_index("track_id")["angle"]
    assert a[1] < 5 and a[2] > 80


def test_windows_follow_shots_and_strata() -> None:
    w = windows(*_fight(), clip_s=3.0)
    assert w["stratum"].tolist() == ["apart", "apart", "close", "ground"]
    assert (w["t1"] - w["t0"]).between(2.9, 3.0).all()
    assert (w["t0"].iloc[1:].to_numpy() >= w["t1"].iloc[:-1].to_numpy()).all()  # no overlap
    assert w["shot_id"].tolist() == [0, 0, 1, 1]


def test_windows_skip_breaks() -> None:
    frames, fighting, boxes, poses = _fight()
    fighting.loc[fighting["frame"].between(10, 50), "fighting"] = False  # 1 to 5 s is a break
    w = windows(frames, fighting, boxes, poses, clip_s=3.0)
    assert w["t0"].min() >= 5.0


def test_sample_balances_strata_and_fights() -> None:
    rng = np.random.default_rng(0)
    clips = pd.DataFrame(
        {"fight_id": rng.choice(["a", "b", "c"], 600, p=[0.8, 0.1, 0.1]),
         "stratum": rng.choice(STRATA, 600, p=[0.7, 0.1, 0.1, 0.1]),
         "t0": rng.random(600)}
    )  # fmt: skip
    s = sample(clips, 80)
    assert len(s) == 80
    assert s["stratum"].value_counts().eq(20).all()
    # Fight a has 80% of the clips but gets no more than its turn in each stratum.
    assert s["fight_id"].value_counts()["a"] < 0.6 * len(s)
    assert sample(clips, 80).equals(s)  # same seed, same sample


def test_sample_prefers_clips_with_both_fighters_boxed() -> None:
    clips = pd.DataFrame(
        {"fight_id": ["a"] * 8, "stratum": ["ground"] * 8, "t0": np.arange(8.0),
         "both": [0.0, 0.1, 0.9, 0.2, 1.0, 0.0, 0.8, 0.3]}
    )  # fmt: skip
    # Three clips: the three with both fighters boxed come first.
    assert sorted(sample(clips, 3)["both"]) == [0.8, 0.9, 1.0]
