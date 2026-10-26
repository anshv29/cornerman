"""Pick a batch of fights for a Kaggle job, and bring its outputs home.

A batch only contains fights whose final stage isn't on the laptop yet, so a
job that died halfway is resumed by simply building the next batch.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cornerman.contracts.hold import HOLD_CSV, held_fights

REPO = "https://github.com/anshv29/cornerman"
FINAL_STAGE = "identity"


def done_fights(derived: Path) -> set[str]:
    return {p.parent.parent.name for p in derived.glob(f"*/{FINAL_STAGE}/manifest.json")}


def select_fights(
    fights: pd.DataFrame, derived: Path, n: int, hold: Path = HOLD_CSV
) -> pd.DataFrame:
    """The next n unprocessed fights, favouring fighters with the most fights.

    Repeat fighters are the most valuable data (SPEC Layer 0), so the pilot
    starts with them. Fights on hold (contracts.hold) are never picked.
    """
    counts = pd.concat([fights["fighter_a_id"], fights["fighter_b_id"]]).value_counts()
    depth = fights["fighter_a_id"].map(counts) + fights["fighter_b_id"].map(counts)
    skip = done_fights(derived) | held_fights(hold)
    todo = fights.assign(depth=depth)[~fights["fight_id"].isin(skip)]
    return todo.sort_values(["depth", "fight_id"], ascending=[False, True]).head(n)


def pushed_commit() -> str:
    """HEAD, after checking it is on GitHub, since the Kaggle job installs from there."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "branch", "-r", "--contains", sha], capture_output=True, text=True, check=True
    ).stdout.strip()
    if not remote:
        raise RuntimeError(f"commit {sha[:8]} is not pushed; the Kaggle job can't install it")
    return sha


def build_values(batch: pd.DataFrame, config: Path, overlays: int) -> dict[str, object]:
    cfg: dict[str, Any] = yaml.safe_load(config.read_text())
    return {
        "FIGHTS": [
            {"fight_id": str(f), "url": str(u)}
            for f, u in zip(batch["fight_id"], batch["url"], strict=True)
        ],
        "CONFIG": cfg,
        "REPO": REPO,
        "COMMIT": pushed_commit(),
        "OVERLAYS": overlays,
    }


@dataclass(frozen=True)
class StageJob:
    """A stage run on Kaggle by itself (kaggle/stage/job.py), on fights processed earlier."""

    stage: str
    kernel_id: str
    """Never the same slug as the dataset: Kaggle refuses a kernel that shares it (409)."""
    gpu: bool
    inputs: tuple[str, ...]
    """Stage folders uploaded with the video; the runner sees them as cached."""
    needs: tuple[str, ...] = ()
    """Stages that must be on the laptop before a fight can go in a batch."""

    @property
    def dataset_slug(self) -> str:
        return f"cornerman-{self.stage}"


STAGE_JOBS = {
    "arena": StageJob("arena", "anshvaishnav29/cornerman-arena-job", False, ("ingest", "shots")),
    "masks": StageJob(
        "masks",
        "anshvaishnav29/cornerman-masks-job",
        True,
        ("ingest", "shots", "tracking", "arena"),
        ("arena",),
    ),
}


def stage_todo(derived: Path, job: StageJob, n: int, hold: Path = HOLD_CSV) -> list[str]:
    """Processed fights (identity done) without `job`'s output, with what it needs, not on
    hold; at most n."""
    held = held_fights(hold)
    todo = sorted(
        f
        for f in done_fights(derived)
        if f not in held
        and not (derived / f / job.stage / "manifest.json").exists()
        and all((derived / f / s / "manifest.json").exists() for s in job.needs)
    )
    return todo[:n] if n else todo


def import_outputs(
    src: Path,
    derived: Path,
    overlays: Path,
    logs: Path,
    crops: Path | None = None,
    final_stage: str = FINAL_STAGE,
    thumbs: Path | None = None,
) -> list[str]:
    """Copy a pulled job output into data/. Returns the fights that finished.

    A fight is finished when `final_stage` has a manifest. Person crops go to
    `crops` and arena thumbnails to `thumbs` (local only, never committed or put
    in DVC). Stage folders the job didn't write are kept.
    """
    finished: list[str] = []
    for fight in sorted((src / "derived").glob("*")):
        if not (fight / final_stage / "manifest.json").exists():
            continue
        # Replace only the stages the job wrote; stages run on the laptop
        # (crop features, identity) stay.
        for stage in (p for p in fight.iterdir() if p.is_dir()):
            dest = derived / fight.name / stage.name
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(stage, dest)
        finished.append(fight.name)
        if crops is not None:
            zipped = src / "crops" / f"{fight.name}.zip"
            if zipped.exists():
                shutil.rmtree(crops / fight.name, ignore_errors=True)
                shutil.unpack_archive(zipped, crops / fight.name)
            elif (src / "crops" / fight.name).exists():
                shutil.rmtree(crops / fight.name, ignore_errors=True)
                shutil.copytree(src / "crops" / fight.name, crops / fight.name)
        zipped_thumbs = src / "thumbs" / f"{fight.name}.zip"
        if thumbs is not None and zipped_thumbs.exists():
            shutil.rmtree(thumbs / fight.name, ignore_errors=True)
            shutil.unpack_archive(zipped_thumbs, thumbs / fight.name)
    if (src / "overlays").exists():
        overlays.mkdir(parents=True, exist_ok=True)
        for clip in (src / "overlays").glob("*.mp4"):
            shutil.copy2(clip, overlays / clip.name)
    if (src / "job_log.json").exists():
        logs.mkdir(parents=True, exist_ok=True)
        # A list of fights (first pilot) or {"gpus", "total_wall_seconds", "fights"}.
        entries: object = json.loads((src / "job_log.json").read_text())
        stamp = len(list(logs.glob("job_log_*.json")))
        (logs / f"job_log_{stamp:03d}.json").write_text(json.dumps(entries, indent=2))
    return finished
