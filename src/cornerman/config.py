"""Run configuration.

Every run is reproducible from one YAML file plus the seed inside it. Values in
the YAML win. Fields the YAML leaves out can come from CORNERMAN_* environment
variables (nested fields use a double underscore, e.g. CORNERMAN_STUB__FPS=30).
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    fixtures: Path = Path("data/fixtures")
    derived: Path = Path("data/derived")
    quarantine: Path = Path("data/quarantine")


class StubConfig(BaseModel):
    """Knobs for the Tier 0 stub stages, which fake perception output."""

    fps: float = Field(default=10.0, gt=0)
    n_shots: int = Field(default=4, ge=1)
    keypoints: int = Field(default=17, ge=1)
    events_per_minute: float = Field(default=12.0, gt=0)


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
    fights: list[str] = Field(default_factory=list)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    stub: StubConfig = Field(default_factory=StubConfig)
    collect: CollectConfig = Field(default_factory=CollectConfig)


def load_settings(path: Path) -> Settings:
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return Settings(**raw)
