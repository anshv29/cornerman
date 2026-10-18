"""The real perception stages: video in, tracks, poses, identities and fighting time out.

Same runner, caching and quarantine as the stub pipeline. Each video is
decoded once into a frame cache next to the raw video (decode.py); a rerun
after the raw folder was deleted fetches the video and rebuilds the cache.

Order: ingest, shots, tracking (people on the canvas, plus person crops), pose
(everyone tracked in live shots, plus colour samples at gloves, shins and
shorts), identity (red, blue, referee, other, from those samples; no GPU),
and the fighting-time filter, which needs to know who the fighters are.
"""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.contracts.perception import (
    IDENTITY_SCHEMA,
    POSES_SCHEMA,
    TRACKS_SCHEMA,
    validate,
)
from cornerman.perception import models
from cornerman.perception.activity import fighting_frames
from cornerman.perception.appearance import N_FEATURES, box_features
from cornerman.perception.arena import mat_colour, mat_hull, on_mat, shot_colour
from cornerman.perception.decode import (
    FrameCache,
    cached_indices,
    ensure_frames,
    iter_cached,
    small_frames,
)
from cornerman.perception.identity import FIGHTERS, assign_roles, live_shots
from cornerman.perception.kitcolour import COLUMNS as SAMPLE_COLUMNS
from cornerman.perception.kitcolour import empty_samples, sample_person
from cornerman.perception.shots import shot_of_frame, shots_from_probs
from cornerman.perception.video import Frame, ensure_video, frame_step, probe
from cornerman.pipeline.hashing import hash_file
from cornerman.pipeline.stage import QuarantineError, Stage, StageContext

APP_COLS = [f"app_{i}" for i in range(N_FEATURES)]


def _video_json(ctx: StageContext) -> dict[str, float]:
    data: dict[str, float] = json.loads((ctx.inputs["ingest"] / "video.json").read_text())
    return data


def _batched(items: Iterator[tuple[int, Frame]], size: int) -> Iterator[list[tuple[int, Frame]]]:
    batch: list[tuple[int, Frame]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest(ctx: StageContext) -> None:
    """Download the video if needed and record its properties and checksum."""
    path = ensure_video(ctx.settings, ctx.fight_id)
    info = probe(path)
    if info.height < ctx.settings.perception.min_height:
        raise QuarantineError(f"video is only {info.height}p")
    video = {
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "n_frames": info.n_frames,
        "duration_s": info.duration_s,
        "sha256": hash_file(path),
    }
    (ctx.out_dir / "video.json").write_text(json.dumps(video, indent=2, sort_keys=True))


def shots(ctx: StageContext) -> None:
    p = ctx.settings.perception
    fps = float(_video_json(ctx)["fps"])
    cache = ensure_frames(ctx.settings, ctx.fight_id, fps)
    probs = models.cut_probabilities(small_frames(cache), models.pick_device(p.device))
    df = shots_from_probs(probs, fps, p.shot_threshold, stride=cache.shot_step)
    if df.empty:
        raise QuarantineError("no shots found")
    df.to_parquet(ctx.out_dir / "shots.parquet", index=False)


class CanvasColours:
    """The canvas colour for the whole fight and, lazily, as each shot's camera sees it."""

    SAMPLES = 60
    PER_SHOT = 12

    def __init__(self, cache: FrameCache, shot_df: pd.DataFrame, max_shift: float) -> None:
        self.cache = cache
        self.max_shift = max_shift
        idx = np.array(cached_indices(cache))
        self.idx = idx
        self.shot_ids = shot_of_frame(shot_df, idx)
        pick = idx[:: max(1, len(idx) // self.SAMPLES)]
        self.fight = mat_colour([f for _, f in iter_cached(cache, set(pick.tolist()))])
        self.by_shot: dict[int, np.ndarray] = {}

    def of(self, shot_id: int) -> np.ndarray:
        if shot_id not in self.by_shot:
            frames = self.idx[self.shot_ids == shot_id]
            pick = set(frames[:: max(1, len(frames) // self.PER_SHOT)].tolist())
            shot = [f for _, f in iter_cached(self.cache, pick)]
            self.by_shot[shot_id] = shot_colour(shot, self.fight, self.max_shift)
        return self.by_shot[shot_id]


class CropWriter:
    """Saves a small crop of each tracked person about once per `every_s` seconds.

    Crops let identity be redone later with a different method without the GPU
    or the video. They contain fight footage, so they go to `paths.crops`
    (local, gitignored, not in DVC) and never into derived data.
    """

    def __init__(self, folder: Path, every_s: float, height: int) -> None:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        self.folder = folder
        self.every_s = every_s
        self.height = height
        self.last: dict[int, float] = {}

    def maybe_save(
        self, frame: Frame, track_id: int, frame_idx: int, t: float, box: np.ndarray
    ) -> None:
        if t - self.last.get(track_id, -1e9) < self.every_s:
            return
        x1, y1, x2, y2 = (round(float(v)) for v in box)
        crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if crop.size == 0:
            return
        scale = self.height / crop.shape[0]
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), self.height))
        cv2.imwrite(str(self.folder / f"{track_id:06d}_{frame_idx:07d}.jpg"), crop)
        self.last[track_id] = t


def tracking(ctx: StageContext) -> None:
    """Detect people, keep the big ones standing on the canvas, track them within each shot."""
    p = ctx.settings.perception
    video = _video_json(ctx)
    fps, height = float(video["fps"]), float(video["height"])
    step = frame_step(fps, p.fps)
    shot_df = pd.read_parquet(ctx.inputs["shots"] / "shots.parquet")
    cache = ensure_frames(ctx.settings, ctx.fight_id, fps)
    tracker = models.ShotTracker(fps / step)
    canvas = CanvasColours(cache, shot_df, p.mat_shot_shift) if p.mat_mask else None
    crops = CropWriter(ctx.settings.paths.crops / ctx.fight_id, p.crop_every_s, p.crop_height)

    rows: list[dict[str, object]] = []
    feats: list[np.ndarray] = []
    sampled: list[tuple[int, int, int]] = []
    for batch in _batched(iter_cached(cache, depth=p.prefetch), p.det_batch):
        idx = np.array([i for i, _ in batch])
        shot_ids = shot_of_frame(shot_df, idx)
        dets = models.detect_people([f for _, f in batch], p.detector, p.det_threshold, p.half)
        for (i, frame), shot_id, (boxes, conf) in zip(batch, shot_ids, dets, strict=True):
            if shot_id < 0:
                sampled.append((i, int(shot_id), 0))
                continue
            keep = (boxes[:, 3] - boxes[:, 1]) >= p.min_person_height * height
            if canvas is not None:
                hull = mat_hull(frame, canvas.of(int(shot_id)), p.mat_delta, p.mat_min_area)
                keep &= on_mat(hull, boxes, p.mat_margin * height, height, p.mat_cut)
            sampled.append((i, int(shot_id), int((~keep).sum())))
            boxes, conf = boxes[keep], conf[keep]
            ids = tracker.update(int(shot_id), frame, boxes, conf)
            for box, c, tid in zip(boxes, conf, ids, strict=True):
                if tid < 0:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                rows.append(
                    {
                        "frame": i,
                        "t": i / fps,
                        "shot_id": int(shot_id),
                        "track_id": int(tid),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": float(c),
                    }
                )
                feats.append(box_features(frame, (x1, y1, x2, y2)))
                crops.maybe_save(frame, int(tid), i, i / fps, box)

    tracks = pd.DataFrame(rows, columns=list(TRACKS_SCHEMA.columns))
    app = pd.DataFrame(np.array(feats).reshape(-1, N_FEATURES), columns=APP_COLS)
    tracks = pd.concat([tracks.reset_index(drop=True), app], axis=1)
    validate(TRACKS_SCHEMA, tracks, "tracks").to_parquet(
        ctx.out_dir / "tracks.parquet", index=False
    )
    frames = pd.DataFrame(sampled, columns=["frame", "shot_id", "n_dropped"])
    frames.insert(1, "t", frames["frame"] / fps)
    frames.to_parquet(ctx.out_dir / "frames.parquet", index=False)
    live_shots(tracks, shot_df, frames, p.live_min_frac).to_parquet(
        ctx.out_dir / "live.parquet", index=False
    )


def pose(ctx: StageContext) -> None:
    """Pose for everyone tracked in live shots, plus colour samples at gloves, shins, shorts."""
    p = ctx.settings.perception
    fps = float(_video_json(ctx)["fps"])
    tracks = pd.read_parquet(ctx.inputs["tracking"] / "tracks.parquet")
    live = pd.read_parquet(ctx.inputs["tracking"] / "live.parquet")
    shot_df = pd.read_parquet(ctx.inputs["shots"] / "shots.parquet")
    boxes = tracks[tracks["shot_id"].isin(live.loc[live["live"], "shot_id"])]
    by_frame = {int(g["frame"].to_numpy(dtype=np.int64)[0]): g for _, g in boxes.groupby("frame")}
    cache = ensure_frames(ctx.settings, ctx.fight_id, fps)
    canvas = CanvasColours(cache, shot_df, p.mat_shot_shift) if p.mat_mask else None
    device = models.pick_device(p.device)

    wanted = iter_cached(cache, wanted=set(by_frame), depth=p.prefetch)
    rows: list[pd.DataFrame] = []
    samples: list[tuple[int, int, str, float, float, float, int]] = []
    for batch in _batched(wanted, p.pose_batch):
        groups = [by_frame[i] for i, _ in batch]
        xyxy = [g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) for g in groups]
        results = models.estimate_poses(
            [f for _, f in batch], xyxy, p.pose_model, p.pose_revision, device, p.half
        )
        for (i, frame), g, b, (kps, scores) in zip(batch, groups, xyxy, results, strict=True):
            colour = canvas.of(int(g["shot_id"].iloc[0])) if canvas is not None else None
            for tid, box, k, s in zip(g["track_id"], b, kps, scores, strict=True):
                rows.append(
                    pd.DataFrame(
                        {
                            "frame": i,
                            "t": i / fps,
                            "track_id": int(tid),
                            "kp": np.arange(17),
                            "x": k[:, 0].astype(float),
                            "y": k[:, 1].astype(float),
                            "conf": np.clip(s.astype(float), 0.0, 1.0),
                        }
                    )
                )
                for part, lab_l, lab_a, lab_b, n in sample_person(
                    frame, k, s, float(box[3] - box[1]), colour, p.kp_min_conf
                ):
                    samples.append((i, int(tid), part, lab_l, lab_a, lab_b, n))
    poses = pd.concat(rows, ignore_index=True) if rows else _empty_poses()
    validate(POSES_SCHEMA, poses, "poses").to_parquet(ctx.out_dir / "poses.parquet", index=False)
    sample_df = pd.DataFrame(samples, columns=SAMPLE_COLUMNS) if samples else empty_samples()
    sample_df.to_parquet(ctx.out_dir / "samples.parquet", index=False)


def _empty_poses() -> pd.DataFrame:
    """No live frames: an empty table that still has the contract's column types."""
    return pd.DataFrame(
        {
            "frame": pd.Series(dtype="int64"),
            "t": pd.Series(dtype="float64"),
            "track_id": pd.Series(dtype="int64"),
            "kp": pd.Series(dtype="int64"),
            "x": pd.Series(dtype="float64"),
            "y": pd.Series(dtype="float64"),
            "conf": pd.Series(dtype="float64"),
        }
    )


def identity(ctx: StageContext) -> None:
    """Corners and referee from the colour samples, then which frames are fighting."""
    p = ctx.settings.perception
    tracks = pd.read_parquet(ctx.inputs["tracking"] / "tracks.parquet")
    frames = pd.read_parquet(ctx.inputs["tracking"] / "frames.parquet")
    live = pd.read_parquet(ctx.inputs["tracking"] / "live.parquet")
    samples = pd.read_parquet(ctx.inputs["pose"] / "samples.parquet")
    if tracks.empty:
        raise QuarantineError("no people tracked")
    roles = assign_roles(tracks, samples, p.min_track_frames, p.max_role_distance)
    validate(IDENTITY_SCHEMA, roles, "identity").to_parquet(
        ctx.out_dir / "identity.parquet", index=False
    )
    fighting = fighting_frames(
        frames,
        fighter_boxes(tracks, roles, live),
        set(live.loc[live["live"], "shot_id"].tolist()),
        p.fight_window_s,
        p.fight_min_speed,
        p.fight_min_gap_s,
        p.fight_min_run_s,
    )
    fighting.to_parquet(ctx.out_dir / "fighting.parquet", index=False)


def fighter_boxes(
    tracks: pd.DataFrame,
    roles: pd.DataFrame,
    live: pd.DataFrame,
    fighting: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """At most one box per fighter per frame, from live shots (and fighting frames, if given)."""
    t = tracks.merge(roles[["track_id", "role"]], on="track_id")
    t = t[t["role"].isin(FIGHTERS) & t["shot_id"].isin(live.loc[live["live"], "shot_id"])]
    if fighting is not None:
        t = t[t["frame"].isin(fighting.loc[fighting["fighting"], "frame"])]
    t = t.sort_values(["frame", "role", "score"], ascending=[True, True, False])
    return t.drop_duplicates(["frame", "role"]).reset_index(drop=True)


REAL_PIPELINE: list[Stage] = [
    Stage("ingest", ingest, config_keys=("perception.max_height", "perception.min_height")),
    Stage(
        "shots",
        shots,
        deps=("ingest",),
        config_keys=("perception.shot_threshold", "perception.shot_fps", "perception.fps"),
    ),
    Stage(
        "tracking",
        tracking,
        deps=("ingest", "shots"),
        config_keys=(
            "perception.fps",
            "perception.detector",
            "perception.det_threshold",
            "perception.min_person_height",
            "perception.half",
            "perception.mat_mask",
            "perception.mat_delta",
            "perception.mat_min_area",
            "perception.mat_margin",
            "perception.mat_shot_shift",
            "perception.mat_cut",
            "perception.live_min_frac",
            "perception.crop_every_s",
            "perception.crop_height",
        ),
    ),
    Stage(
        "pose",
        pose,
        deps=("ingest", "shots", "tracking"),
        config_keys=(
            "perception.fps",
            "perception.pose_model",
            "perception.pose_revision",
            "perception.half",
            "perception.kp_min_conf",
            "perception.mat_mask",
            "perception.mat_shot_shift",
        ),
    ),
    Stage(
        "identity",
        identity,
        deps=("tracking", "pose"),
        config_keys=(
            "perception.min_track_frames",
            "perception.max_role_distance",
            "perception.fight_window_s",
            "perception.fight_min_speed",
            "perception.fight_min_gap_s",
            "perception.fight_min_run_s",
        ),
    ),
]
