"""The real perception pipeline end to end on a tiny synthetic video, on CPU.

The four neural models are replaced by stand-ins (cuts from a known list,
people found by colour, keypoints spread over the box, appearance = mean body
colour). The tracker, identity, contracts, caching and overlay run for real.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("trackers")

from cornerman.config import PathsConfig, PerceptionConfig, Settings  # noqa: E402
from cornerman.perception import models, stages  # noqa: E402
from cornerman.perception.identity import Probe  # noqa: E402
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


def _fake_segment(
    frames: list[np.ndarray], name: str, threshold: float, half: bool = False
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for frame in frames:
        masks = [
            np.all(np.abs(frame.astype(int) - colour) < 40, axis=2)
            for colour in [c for c, _ in PEOPLE] + [SPECTATOR]
        ]
        masks = [m for m in masks if m.any()]
        boxes = [
            [float(xs.min()), float(ys.min()), xs.max() + 1.0, ys.max() + 1.0]
            for ys, xs in (np.nonzero(m) for m in masks)
        ]
        stack = np.array(masks, dtype=bool).reshape(-1, *frame.shape[:2])
        out.append((np.array(boxes, dtype=float).reshape(-1, 4), np.full(len(boxes), 0.9), stack))
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


def _fake_embed(
    inputs: list[np.ndarray], weights: list[np.ndarray], name: str, revision: str, device: str
) -> np.ndarray:
    """Appearance = mean colour (normalised RGB) over the body mask, in the first 3 of 384 dims."""
    out = np.zeros((len(inputs), 384), np.float32)
    for i, (x, w) in enumerate(zip(inputs, weights, strict=True)):
        cell = np.kron(w.reshape(16, 16), np.ones((14, 14)))
        out[i, :3] = (x * cell).sum((1, 2)) / max(float(cell.sum()), 1e-6)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)


def _fake_bar() -> Callable[[np.ndarray], list[str]]:
    """A round clock counting down one second per reading (one reading a second)."""
    seconds = iter(range(163, 0, -1))

    def read(img: np.ndarray) -> list[str]:
        s = next(seconds)
        return [f"{s // 60:02d}:{s % 60:02d}", "ROUND 2"]

    return read


def test_real_pipeline_on_synthetic_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "cut_probabilities", _fake_cuts)
    monkeypatch.setattr(models, "detect_people", _fake_detect)
    monkeypatch.setattr(models, "estimate_poses", _fake_pose)
    monkeypatch.setattr(models, "segment_people", _fake_segment)
    monkeypatch.setattr(models, "embed_crops", _fake_embed)
    monkeypatch.setattr(stages, "bar_reader", _fake_bar)
    # Fighter score from the green channel: red (255, 0, 0) and blue (0, 0, 255) people
    # have none, the grey referee has some.
    w = np.zeros(384)
    w[1] = -6.0
    Probe(w=w, b=-1.0).save(tmp_path / "probe.npz")
    settings = Settings(
        pipeline="real",
        paths=PathsConfig(
            derived=tmp_path / "derived",
            quarantine=tmp_path / "quarantine",
            raw=tmp_path / "raw",
            crops=tmp_path / "crops",
            thumbs=tmp_path / "thumbs",
            manifest=tmp_path / "manifest.json",
        ),
        perception=PerceptionConfig(
            fps=FPS,
            device="cpu",
            crop_every_s=0.3,
            identity_probe=tmp_path / "probe.npz",
            det_batch=7,
            min_height=200,
            fight_window_s=1.0,
            fight_min_gap_s=0.0,
            fight_min_run_s=0.0,
            arena_clock_every_s=1.0,
            arena_thumb_every_s=2.0,
            masks_every_s=0.5,
            masks_width=80,
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
    assert set(roles["role"]) == {"red", "blue", "referee"}
    # Corners come from colour: the red-shorts person (left) is red, blue-shorts is blue,
    # grey (right) is the referee, on both sides of the cut.
    per_box = pd.read_parquet(d / "identity" / "roles.parquet")
    assert len(per_box) == len(tracks)
    t = tracks.merge(per_box, on=["frame", "track_id"])
    mean_x = t.groupby("role")["x1"].mean()
    assert mean_x["red"] < mean_x["blue"] < mean_x["referee"]

    poses = pd.read_parquet(d / "pose" / "poses.parquet")
    assert set(poses["track_id"]) == set(tracks["track_id"])  # everyone on the canvas
    assert poses.groupby(["frame", "track_id"]).size().eq(17).all()
    samples = pd.read_parquet(d / "pose" / "samples.parquet")
    assert set(samples["part"]) == {"glove", "shin", "shorts"}
    crops = list((tmp_path / "crops" / "ft_test").glob("*.jpg"))
    assert crops, "expected person crops"
    # About one per crop_every_s (0.3 s) per tracked person, over 6 s of video.
    assert len(crops) <= tracks["track_id"].nunique() * 21

    quality = json.loads((d / "identity" / "quality.json").read_text())
    assert 0.0 <= quality["confidence"] <= 1.0
    assert {"dup", "fighter_prob_min", "rb_share", "closest"} <= set(quality["checks"])
    feats = np.load(d / "crop_features" / "features.npy")
    assert len(feats) == len(pd.read_parquet(d / "crop_features" / "crops.parquet"))

    # Arena: a mat outline in every sampled frame of a shot, the clock every second,
    # a thumbnail every 2 s.
    hull = pd.read_parquet(d / "arena" / "hull.parquet")
    assert len(hull) == (frames["shot_id"] >= 0).sum()
    assert all(len(x) >= 3 for x in hull["x"])
    ys = np.concatenate(hull["y"].to_list())
    assert ys.min() == pytest.approx(140 / SIZE[1], abs=0.05)  # the canvas starts at y=140
    clock = pd.read_parquet(d / "arena" / "clock.parquet")
    assert len(clock) == N // FPS
    assert clock["clock_s"].tolist() == [163.0 - k for k in range(N // FPS)]
    assert (clock["round"] == 2).all()
    # The clock runs throughout, so every frame of a live shot is fighting time.
    v2 = pd.read_parquet(d / "fighting" / "fighting.parquet")
    assert (v2["clock"] == "running").mean() > 0.9
    assert v2["fighting"].mean() > 0.8
    assert len(list((tmp_path / "thumbs" / "ft_test").glob("*.jpg"))) == N // (2 * FPS)

    # Masks every 0.5 s where the mat shows: the three people on the canvas, not the
    # spectator, each with a mask and a skeleton.
    from cornerman.perception.tangle import unpack

    inst = pd.read_parquet(d / "masks" / "instances.parquet")
    with_mat = set(hull.loc[hull["x"].map(len) >= 3, "frame"])
    assert set(inst["frame"]) == set(range(0, N, 5)) & with_mat
    assert inst.groupby("frame").size().eq(3).all()
    assert all(len(k) == 17 for k in inst["kx"])
    saved = np.load(d / "masks" / "masks.npz")
    shape = tuple(int(v) for v in saved["shape"])
    assert shape == (60, 80)
    m = unpack(saved["packed"], (shape[0], shape[1]))
    assert len(m) == len(inst) and (m.sum((1, 2)) == inst["area"].to_numpy()).all()

    # Second run is fully cached.
    again = run(REAL_PIPELINE, settings, ["ft_test"])
    assert {r.status for r in again} == {Status.CACHED}

    out = render(tmp_path / "raw" / "ft_test" / "video.mp4", d, tmp_path / "o.mp4", FPS, seconds=2)
    assert out.stat().st_size > 0
