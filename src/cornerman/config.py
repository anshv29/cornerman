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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CORNERMAN_", env_nested_delimiter="__")

    seed: int = 0
    fights: list[str] = Field(default_factory=list)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    stub: StubConfig = Field(default_factory=StubConfig)


def load_settings(path: Path) -> Settings:
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return Settings(**raw)
