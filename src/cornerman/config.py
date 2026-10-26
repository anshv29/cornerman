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
    crops: Path = Path("data/crops")
    """Small person crops, one folder per fight. Local only: never committed, never in DVC."""
    video_source: Path | None = None
    """A folder holding pre-downloaded videos named {fight_id}.mp4 (searched recursively),
    e.g. the private Kaggle dataset. Used before trying to download."""


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
    shot_fps: float = Field(default=25.0, gt=0)
    """Frame rate fed to shot detection. A 50 fps video is halved; cuts are still
    sharp at 25 fps, which is close to the footage TransNetV2 was trained on."""
    detector: str = "rfdetr-large"
    det_threshold: float = Field(default=0.4, gt=0, lt=1)
    det_batch: int = Field(default=8, ge=1)
    prefetch: int = Field(default=64, ge=0)
    """Frames decoded ahead on a background thread while the GPU works. 0 turns it off."""
    min_person_height: float = Field(default=0.2, gt=0, lt=1)
    """Boxes shorter than this fraction of the frame are crowd or corner staff."""
    mat_mask: bool = True
    """Drop people whose feet aren't on the canvas (spectators, officials outside the cage)."""
    mat_delta: float = Field(default=18.0, gt=0)
    """Colour distance (Lab chroma for a coloured canvas) that still counts as canvas."""
    mat_min_area: float = Field(default=0.05, gt=0, lt=1)
    mat_margin: float = Field(default=0.03, ge=0)
    """How far outside the canvas outline feet may be, as a share of frame height."""
    mat_shot_shift: float = Field(default=45.0, ge=0)
    """How far a shot's own canvas colour may be from the fight's before it's distrusted."""
    mat_cut: float = Field(default=0.15, ge=0, lt=1)
    """Boxes reaching into this bottom share of the frame have their feet hidden."""
    crop_every_s: float = Field(default=1.0, gt=0)
    """Save a crop of each tracked person about this often (for redoing identity later)."""
    crop_height: int = Field(default=160, ge=16)
    kp_min_conf: float = Field(default=0.3, ge=0, le=1)
    """Keypoints below this confidence aren't used for colour samples."""
    live_min_frac: float = Field(default=0.5, gt=0, le=1)
    """A shot is live if at least this share of its frames shows 2+ large people."""
    fight_window_s: float = Field(default=10.0, gt=0)
    fight_min_speed: float = Field(default=0.2, gt=0)
    """Median fighter speed (box heights per second) that counts as fighting. On the
    pilot, rounds ran at 0.3 to 0.6 and breaks near 0.1; 0.2 gave 9.3 and 9.5 minutes."""
    fight_min_gap_s: float = Field(default=8.0, ge=0)
    fight_min_run_s: float = Field(default=15.0, ge=0)
    crop_model: str = "facebook/dinov2-small"
    """Appearance features for person crops (Apache 2.0), used by identity."""
    crop_revision: str = "ed25f3a31f01"
    merged_ratio: float = Field(default=1.6, gt=1)
    """A crop whose box is this many times wider (relative to height) than the fight's
    median box is "merged": in a clinch, one box around both fighters."""
    identity_probe: Path = Path(__file__).parent / "perception" / "identity_probe.npz"
    """Fighter-versus-official classifier weights (`cornerman identity-probe` writes them).
    Shipped inside the package so the Kaggle job, which pip-installs it, has them too."""
    identity_flag_below: float = Field(default=0.5, ge=0, le=1)
    """Fights whose identity confidence is below this are flagged (quality.json). 0.5 was
    chosen after the held-out test (the frozen 0.4 caught too few wrong fights), so it
    must be re-validated on the next fresh fights (DECISIONS.md, 2026-10-19)."""
    identity_segment_min_s: float = Field(default=3.0, ge=0)
    """Shots with less fighting time than this are never usable on their own in a flagged
    fight: too few crops to judge (identity_segments)."""
    identity_segment_min_both: float = Field(default=0.3, ge=0, le=1)
    """A shot of a flagged fight is usable only if both fighters are boxed in at least
    this share of its fighting frames (identity_segments; set from the first by-eye
    check, tested on a second, DECISIONS.md)."""
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
    age_verdicts_csv: Path = Path("data/meta/age_verdicts.csv")
    """Results-check verdicts per video (experiments/age_results_check.py)."""
    hold_csv: Path = Path("data/age_hold.csv")
    """Fights whose age isn't confirmed yet (contracts.hold)."""


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
