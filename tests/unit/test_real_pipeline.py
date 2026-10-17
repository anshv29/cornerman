"""The real perception pipeline end to end on a tiny synthetic video, on CPU.

The three neural models are replaced by stand-ins (cuts from a known list,
people found by colour, keypoints spread over the box). The tracker, identity,
contracts, caching and overlay run for real.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("trackers")

from cornerman.config import PathsConfig, PerceptionConfig, Settings  # noqa: E402
from cornerman.perception import models  # noqa: E402
from cornerman.perception.stages import REAL_PIPELINE  # noqa: E402
from cornerman.pipeline.runner import Status, run  # noqa: E402
from cornerman.products.overlay import render  # noqa: E402

FPS = 10
N = 60
CUT = 30
SIZE = (320, 240)
# BGR colours: fighter in red shorts, fighter in blue shorts, referee in grey,
# all standing on a light-blue canvas, plus a green spectator off the canvas.
PEOPLE = [((0, 0, 255), 60), ((255, 0, 0), 120), ((128, 128, 128), 240)]
SPECTATOR = (0, 200, 0)
CANVAS = (235, 200, 140)


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, SIZE)
    for f in range(N):
        frame = np.full((SIZE[1], SIZE[0], 3), 30, dtype=np.uint8)
        frame[140:240, :] = CANVAS
        frame[20:100, 280:300] = SPECTATOR  # feet at y=100, above the canvas
        for colour, x in PEOPLE:
            # Fighters move like fighters (well above the fighting speed threshold).
            x0 = x + int(25 * np.sin(f / 3)) + (20 if f >= CUT else 0)
            frame[60:200, max(0, x0) : x0 + 40] = colour
        writer.write(frame)
    writer.release()


def _fake_cuts(path: Path, device: str) -> np.ndarray:
    probs = np.zeros(N)
    probs[CUT] = 0.9
    return probs


def _fake_detect(
    frames: list[np.ndarray], name: str, threshold: float, half: bool = False
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in frames:
        boxes: list[list[float]] = []
        for colour in [c for c, _ in PEOPLE] + [SPECTATOR]:
            mask = np.all(np.abs(frame.astype(int) - colour) < 40, axis=2)
            ys, xs = np.nonzero(mask)
            if len(xs):
                boxes.append([float(xs.min()), float(ys.min()), xs.max() + 1.0, ys.max() + 1.0])
        out.append((np.array(boxes, dtype=float).reshape(-1, 4), np.full(len(boxes), 0.95)))
    return out


def _fake_pose(
    frames: list[np.ndarray],
    boxes: list[np.ndarray],
    name: str,
    revision: str,
    device: str,
    half: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for b in boxes:
        t = np.linspace(0, 1, 17)
        kps = np.stack(
            [b[:, [0]] + t * (b[:, [2]] - b[:, [0]]), b[:, [1]] + t * (b[:, [3]] - b[:, [1]])],
            axis=2,
        )
        out.append((kps, np.full((len(b), 17), 0.8)))
    return out


def test_real_pipeline_on_synthetic_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "cut_probabilities", _fake_cuts)
    monkeypatch.setattr(models, "detect_people", _fake_detect)
    monkeypatch.setattr(models, "estimate_poses", _fake_pose)
    settings = Settings(
        pipeline="real",
        paths=PathsConfig(
            derived=tmp_path / "derived",
            quarantine=tmp_path / "quarantine",
            raw=tmp_path / "raw",
            manifest=tmp_path / "manifest.json",
        ),
        perception=PerceptionConfig(
            fps=FPS,
            device="cpu",
            min_track_frames=3,
            det_batch=7,
            min_height=200,
            fight_window_s=1.0,
            fight_min_gap_s=0.0,
            fight_min_run_s=0.0,
        ),
    )
    _write_video(tmp_path / "raw" / "ft_test" / "video.mp4")

    results = run(REAL_PIPELINE, settings, ["ft_test"])
    assert {r.status for r in results} == {Status.RAN}, results
    d = tmp_path / "derived" / "ft_test"

    roles = pd.read_parquet(d / "identity" / "identity.parquet")
    tracks = pd.read_parquet(d / "tracking" / "tracks.parquet")
    assert tracks["shot_id"].nunique() == 2
    # The spectator off the canvas is dropped by the cage mask, every frame.
    assert (tracks["x1"].between(275, 305) & (tracks["y2"] <= 101)).sum() == 0
    frames = pd.read_parquet(d / "tracking" / "frames.parquet")
    assert (frames.loc[frames["shot_id"] >= 0, "n_dropped"] >= 1).all()
    fighting = pd.read_parquet(d / "identity" / "fighting.parquet")
    assert fighting["fighting"].mean() > 0.8
    assert set(roles["role"]) == {"fighter_a", "fighter_b", "referee"}
    # Every frame of the referee's colour ends up labelled referee, across the cut.
    t = tracks.merge(roles, on="track_id")
    ref_x = t.loc[t["role"] == "referee", "x1"]
    assert (ref_x > 200).all()

    poses = pd.read_parquet(d / "pose" / "poses.parquet")
    assert set(poses["role"]) == {"fighter_a", "fighter_b"}
    assert poses.groupby(["frame", "role"]).size().eq(17).all()

    # Second run is fully cached.
    again = run(REAL_PIPELINE, settings, ["ft_test"])
    assert {r.status for r in again} == {Status.CACHED}

    out = render(tmp_path / "raw" / "ft_test" / "video.mp4", d, tmp_path / "o.mp4", FPS, seconds=2)
    assert out.stat().st_size > 0
