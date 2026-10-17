"""Run configuration.

Every run is reproducible from one YAML file plus the seed inside it. Values in
the YAML win. Fields the YAML leaves out can come from CORNERMAN_* environment
variables (nested fields use a double underscore, e.g. CORNERMAN_STUB__FPS=30).
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    fixtures: Path = Path("data/fixtures")
    derived: Path = Path("data/derived")
    quarantine: Path = Path("data/quarantine")
    raw: Path = Path("data/raw")
    """Raw video, one folder per fight. Deleted once the fight is processed."""
    manifest: Path = Path("data/manifest.json")
    """fight_id -> video URL for the fights in this run. Never committed."""


class StubConfig(BaseModel):
    """Knobs for the Tier 0 stub stages, which fake perception output."""

    fps: float = Field(default=10.0, gt=0)
    n_shots: int = Field(default=4, ge=1)
    keypoints: int = Field(default=17, ge=1)
    events_per_minute: float = Field(default=12.0, gt=0)


class PerceptionConfig(BaseModel):
    """Knobs for the real perception stages. Models are pinned in DECISIONS.md."""

    max_height: int = Field(default=720, ge=240)
    """Download at most this resolution. 720p is plenty for 2D pose and halves the bytes."""
    min_height: int = Field(default=480, ge=1)
    """Quarantine a fight whose downloaded video is smaller than this (spec: drop under 480p)."""
    fps: float = Field(default=10.0, gt=0)
    """Frames per second sampled for detection, tracking and pose."""
    device: str = "auto"
    half: bool = False
    """Run the detector and pose model in FP16 on the GPU. About twice as fast on a T4;
    outputs differ from FP32 only by rounding (checked with `cornerman compare`)."""
    shot_threshold: float = Field(default=0.5, gt=0, lt=1)
    detector: str = "rfdetr-large"
    det_threshold: float = Field(default=0.4, gt=0, lt=1)
    det_batch: int = Field(default=8, ge=1)
    prefetch: int = Field(default=64, ge=0)
    """Frames decoded ahead on a background thread while the GPU works. 0 turns it off."""
    min_person_height: float = Field(default=0.2, gt=0, lt=1)
    """Boxes shorter than this fraction of the frame are crowd or corner staff."""
    min_track_frames: int = Field(default=5, ge=1)
    max_role_distance: float = Field(default=0.6, gt=0)
    """Tracklets further than this (appearance distance) from every role are 'other'."""
    live_min_frac: float = Field(default=0.5, gt=0, le=1)
    """A shot is live if at least this share of its frames shows 2+ large people."""
    pose_model: str = "usyd-community/vitpose-plus-large"
    pose_revision: str = "e211df377e49"
    pose_batch: int = Field(default=16, ge=1)
    overlay_seconds: float = Field(default=60.0, gt=0)


class SourceConfig(BaseModel):
    """One public channel of amateur fights."""

    name: str
    channel_url: str
    title_contains: str = "FULL FIGHT"
    """Only uploads whose title has this text are single, complete fights."""


class CollectConfig(BaseModel):
    """Footage collection: where fights come from and what gets kept."""

    sources: list[SourceConfig] = Field(default_factory=list[SourceConfig])
    min_height: int = Field(default=480, ge=1)
    meta_dir: Path = Path("data/meta")
    fights_csv: Path = Path("data/fights.csv")
    fighters_csv: Path = Path("data/fighters.csv")
    skipped_csv: Path = Path("data/skipped.csv")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CORNERMAN_", env_nested_delimiter="__")

    seed: int = 0
    pipeline: Literal["stub", "real"] = "stub"
    fights: list[str] = Field(default_factory=list)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    stub: StubConfig = Field(default_factory=StubConfig)
    collect: CollectConfig = Field(default_factory=CollectConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)


def load_settings(path: Path) -> Settings:
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return Settings(**raw)
