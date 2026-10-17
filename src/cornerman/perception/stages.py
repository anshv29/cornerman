"""The real perception stages: video in, tracks, identities and poses out.

Same runner, caching and quarantine as the stub pipeline. Raw video is read
through `ensure_video`, so a rerun after the video was deleted downloads it again.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from cornerman.contracts.perception import (
    IDENTITY_SCHEMA,
    POSES_SCHEMA,
    TRACKS_SCHEMA,
    validate,
)
from cornerman.perception import models
from cornerman.perception.appearance import N_FEATURES, box_features
from cornerman.perception.identity import ROLES, assign_roles, live_shots
from cornerman.perception.shots import shot_of_frame, shots_from_probs
from cornerman.perception.video import (
    Frame,
    ensure_video,
    frame_step,
    iter_frames,
    prefetch,
    probe,
)
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
    path = ensure_video(ctx.settings, ctx.fight_id)
    probs = models.cut_probabilities(path, models.pick_device(p.device))
    df = shots_from_probs(probs, fps, p.shot_threshold)
    if df.empty:
        raise QuarantineError("no shots found")
    df.to_parquet(ctx.out_dir / "shots.parquet", index=False)


def tracking(ctx: StageContext) -> None:
    """Detect people on sampled frames, keep the big ones, track them within each shot."""
    p = ctx.settings.perception
    video = _video_json(ctx)
    fps, height = float(video["fps"]), float(video["height"])
    step = frame_step(fps, p.fps)
    shot_df = pd.read_parquet(ctx.inputs["shots"] / "shots.parquet")
    path = ensure_video(ctx.settings, ctx.fight_id)
    tracker = models.ShotTracker(fps / step)

    rows: list[dict[str, object]] = []
    feats: list[np.ndarray] = []
    sampled: list[tuple[int, int]] = []
    frames_in = prefetch(iter_frames(path, step=step), p.prefetch)
    for batch in _batched(frames_in, p.det_batch):
        idx = np.array([i for i, _ in batch])
        shot_ids = shot_of_frame(shot_df, idx)
        dets = models.detect_people([f for _, f in batch], p.detector, p.det_threshold, p.half)
        for (i, frame), shot_id, (boxes, conf) in zip(batch, shot_ids, dets, strict=True):
            sampled.append((i, int(shot_id)))
            if shot_id < 0:
                continue
            big = (boxes[:, 3] - boxes[:, 1]) >= p.min_person_height * height
            boxes, conf = boxes[big], conf[big]
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

    tracks = pd.DataFrame(rows, columns=list(TRACKS_SCHEMA.columns))
    app = pd.DataFrame(np.array(feats).reshape(-1, N_FEATURES), columns=APP_COLS)
    tracks = pd.concat([tracks.reset_index(drop=True), app], axis=1)
    validate(TRACKS_SCHEMA, tracks, "tracks").to_parquet(
        ctx.out_dir / "tracks.parquet", index=False
    )
    frames = pd.DataFrame(sampled, columns=["frame", "shot_id"])
    frames.insert(1, "t", frames["frame"] / fps)
    frames.to_parquet(ctx.out_dir / "frames.parquet", index=False)


def identity(ctx: StageContext) -> None:
    p = ctx.settings.perception
    tracks = pd.read_parquet(ctx.inputs["tracking"] / "tracks.parquet")
    frames = pd.read_parquet(ctx.inputs["tracking"] / "frames.parquet")
    shot_df = pd.read_parquet(ctx.inputs["shots"] / "shots.parquet")
    if tracks.empty:
        raise QuarantineError("no people tracked")
    roles = assign_roles(tracks, p.min_track_frames, p.max_role_distance)
    validate(IDENTITY_SCHEMA, roles, "identity").to_parquet(
        ctx.out_dir / "identity.parquet", index=False
    )
    live = live_shots(tracks, shot_df, frames, p.live_min_frac)
    live.to_parquet(ctx.out_dir / "live.parquet", index=False)


def fighter_boxes(tracks: pd.DataFrame, roles: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """At most one box per fighter per frame, from live shots only."""
    t = tracks.merge(roles[["track_id", "role"]], on="track_id")
    t = t[t["role"].isin(ROLES[:2]) & t["shot_id"].isin(live.loc[live["live"], "shot_id"])]
    t = t.sort_values(["frame", "role", "score"], ascending=[True, True, False])
    return t.drop_duplicates(["frame", "role"]).reset_index(drop=True)


def pose(ctx: StageContext) -> None:
    p = ctx.settings.perception
    video = _video_json(ctx)
    fps = float(video["fps"])
    step = frame_step(fps, p.fps)
    tracks = pd.read_parquet(ctx.inputs["tracking"] / "tracks.parquet")
    roles = pd.read_parquet(ctx.inputs["identity"] / "identity.parquet")
    live = pd.read_parquet(ctx.inputs["identity"] / "live.parquet")
    boxes = fighter_boxes(tracks, roles, live)
    by_frame = {int(g["frame"].to_numpy(dtype=np.int64)[0]): g for _, g in boxes.groupby("frame")}
    path = ensure_video(ctx.settings, ctx.fight_id)
    device = models.pick_device(p.device)

    wanted = prefetch(
        (item for item in iter_frames(path, step=step) if item[0] in by_frame), p.prefetch
    )
    rows: list[pd.DataFrame] = []
    for batch in _batched(wanted, p.pose_batch):
        groups = [by_frame[i] for i, _ in batch]
        xyxy = [g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) for g in groups]
        results = models.estimate_poses(
            [f for _, f in batch], xyxy, p.pose_model, p.pose_revision, device, p.half
        )
        for (i, _), g, (kps, scores) in zip(batch, groups, results, strict=True):
            for role, k, s in zip(g["role"], kps, scores, strict=True):
                rows.append(
                    pd.DataFrame(
                        {
                            "frame": i,
                            "t": i / fps,
                            "role": role,
                            "kp": np.arange(17),
                            "x": k[:, 0].astype(float),
                            "y": k[:, 1].astype(float),
                            "conf": np.clip(s.astype(float), 0.0, 1.0),
                        }
                    )
                )
    poses = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=list(POSES_SCHEMA.columns))
    )
    validate(POSES_SCHEMA, poses, "poses").to_parquet(ctx.out_dir / "poses.parquet", index=False)


REAL_PIPELINE: list[Stage] = [
    Stage("ingest", ingest, config_keys=("perception.max_height", "perception.min_height")),
    Stage("shots", shots, deps=("ingest",), config_keys=("perception.shot_threshold",)),
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
        ),
    ),
    Stage(
        "identity",
        identity,
        deps=("tracking", "shots"),
        config_keys=(
            "perception.min_track_frames",
            "perception.max_role_distance",
            "perception.live_min_frac",
        ),
    ),
    Stage(
        "pose",
        pose,
        deps=("ingest", "tracking", "identity"),
        config_keys=(
            "perception.fps",
            "perception.pose_model",
            "perception.pose_revision",
            "perception.half",
        ),
    ),
]


def output_path(derived: Path, fight_id: str, stage: str, name: str) -> Path:
    return derived / fight_id / stage / name
