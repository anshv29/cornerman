"""Tier 0 stub stages.

These have the same names, inputs and output files the real perception stages
will have, but they make up plausible data from the seed instead of reading
video. They exist so the runner, caching, contracts and DVC can be built and
tested before any GPU work. Each one gets replaced by the real thing in Tier 1.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from cornerman.config import Settings
from cornerman.contracts.events import validate_events
from cornerman.pipeline.stage import Stage, StageContext
from cornerman.taxonomy import Action, State, compatible

ROLES = ("fighter_a", "fighter_b", "referee")
FIGHTER_ROLES = ("fighter_a", "fighter_b")


def _meta_source(fight_id: str, settings: Settings) -> Sequence[Path]:
    return [settings.paths.fixtures / fight_id / "meta.json"]


def _read_json(path: Path) -> dict[str, object]:
    data: dict[str, object] = json.loads(path.read_text())
    return data


def ingest(ctx: StageContext) -> None:
    """Copy fight metadata and record (fake) video properties."""
    meta = _read_json(ctx.settings.paths.fixtures / ctx.fight_id / "meta.json")
    rounds = int(str(meta["rounds"]))
    round_seconds = int(str(meta["round_seconds"]))
    video = {"duration_s": rounds * round_seconds, "fps": ctx.settings.stub.fps}
    (ctx.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (ctx.out_dir / "video.json").write_text(json.dumps(video, indent=2, sort_keys=True))


def shots(ctx: StageContext) -> None:
    """Split the fight into shots at random cut points."""
    duration = float(str(_read_json(ctx.inputs["ingest"] / "video.json")["duration_s"]))
    n = ctx.settings.stub.n_shots
    cuts = np.sort(ctx.rng().uniform(0, duration, size=n - 1))
    edges = np.concatenate([[0.0], cuts, [duration]])
    df = pd.DataFrame(
        {"shot_id": np.arange(n), "start": edges[:-1], "end": edges[1:], "kind": "live"}
    )
    df.to_parquet(ctx.out_dir / "shots.parquet", index=False)


def tracking(ctx: StageContext) -> None:
    """Random-walk boxes for both fighters and the referee, in mat coordinates 0..1."""
    video = _read_json(ctx.inputs["ingest"] / "video.json")
    duration, fps = float(str(video["duration_s"])), float(str(video["fps"]))
    shot_df = pd.read_parquet(ctx.inputs["shots"] / "shots.parquet")
    t = np.arange(0, duration, 1 / fps)
    shot_id = np.searchsorted(shot_df["end"].to_numpy(), t, side="right")
    rng = ctx.rng()
    frames: list[pd.DataFrame] = []
    for i, role in enumerate(ROLES):
        steps = rng.normal(0, 0.01, size=(len(t), 2))
        xy = np.clip(0.3 + 0.2 * i + np.cumsum(steps, axis=0), 0, 1)
        frames.append(
            pd.DataFrame(
                {
                    "t": t,
                    "shot_id": np.minimum(shot_id, len(shot_df) - 1),
                    "role": role,
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                }
            )
        )
    tracks = pd.concat(frames).sort_values(["t", "role"], kind="stable")
    tracks.to_parquet(ctx.out_dir / "tracks.parquet", index=False)


def pose(ctx: StageContext) -> None:
    """Keypoints scattered around each fighter's box centre."""
    tracks = pd.read_parquet(ctx.inputs["tracking"] / "tracks.parquet")
    fighters = tracks[tracks["role"].isin(FIGHTER_ROLES)].reset_index(drop=True)
    k = ctx.settings.stub.keypoints
    rng = ctx.rng()
    n = len(fighters)
    df = pd.DataFrame(
        {
            "t": np.repeat(fighters["t"].to_numpy(), k),
            "role": np.repeat(fighters["role"].to_numpy(), k),
            "kp": np.tile(np.arange(k), n),
            "x": np.repeat(fighters["x"].to_numpy(), k) + rng.normal(0, 0.02, n * k),
            "y": np.repeat(fighters["y"].to_numpy(), k) + rng.normal(0, 0.02, n * k),
            "conf": rng.uniform(0.3, 1.0, n * k),
        }
    )
    df.to_parquet(ctx.out_dir / "poses.parquet", index=False)


def geometry(ctx: StageContext) -> None:
    """Distance to the fence, treating the mat as a 9 m square."""
    tracks = pd.read_parquet(ctx.inputs["tracking"] / "tracks.parquet")
    f = tracks[tracks["role"].isin(FIGHTER_ROLES)].reset_index(drop=True)
    x, y = f["x"].to_numpy(), f["y"].to_numpy()
    edge = np.minimum.reduce([x, 1 - x, y, 1 - y])
    out = pd.DataFrame({"t": f["t"], "role": f["role"], "fence_dist_m": 9.0 * edge})
    out.to_parquet(ctx.out_dir / "geometry.parquet", index=False)


_STATE_PAIRS = [(a, b) for a in State for b in State if compatible(a, b)]


def events(ctx: StageContext) -> None:
    """One row per fighter per second from a sticky random walk over legal state pairs."""
    meta = _read_json(ctx.inputs["ingest"] / "meta.json")
    duration = float(str(_read_json(ctx.inputs["ingest"] / "video.json")["duration_s"]))
    ids = (str(meta["fighter_a"]), str(meta["fighter_b"]))
    rng = ctx.rng()
    p_act = ctx.settings.stub.events_per_minute / 60.0 / 2.0
    actions = [a.value for a in Action]

    rows: list[dict[str, object]] = []
    pair = _STATE_PAIRS[0]
    for t in np.arange(0.0, duration, 1.0):
        if rng.random() < 0.2:
            pair = _STATE_PAIRS[rng.integers(len(_STATE_PAIRS))]
        for fighter, state in zip(ids, pair, strict=True):
            action = actions[rng.integers(len(actions))] if rng.random() < p_act else None
            landed = bool(rng.random() < 0.4) if action == Action.STRIKE.value else None
            rows.append(
                {
                    "fight_id": ctx.fight_id,
                    "t": float(t),
                    "fighter": fighter,
                    "state": state.value,
                    "momentum": bool(rng.random() < 0.1),
                    "action": action,
                    "landed": landed,
                    "confidence": float(rng.uniform(0.5, 1.0)),
                    "source": "stub",
                }
            )
    df = pd.DataFrame(rows)
    df["landed"] = df["landed"].astype("boolean")
    validate_events(df).to_parquet(ctx.out_dir / "events.parquet", index=False)


PIPELINE: list[Stage] = [
    Stage("ingest", ingest, config_keys=("stub.fps",), sources=_meta_source),
    Stage("shots", shots, deps=("ingest",), config_keys=("stub.n_shots",)),
    Stage("tracking", tracking, deps=("ingest", "shots")),
    Stage("pose", pose, deps=("tracking",), config_keys=("stub.keypoints",)),
    Stage("geometry", geometry, deps=("tracking",)),
    Stage(
        "events",
        events,
        deps=("ingest", "geometry", "pose"),
        config_keys=("stub.events_per_minute",),
    ),
]
