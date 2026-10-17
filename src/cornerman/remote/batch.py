"""Pick a batch of fights for a Kaggle job, and bring its outputs home.

A batch only contains fights whose final stage isn't on the laptop yet, so a
job that died halfway is resumed by simply building the next batch.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = "https://github.com/anshv29/cornerman"
FINAL_STAGE = "pose"


def done_fights(derived: Path) -> set[str]:
    return {p.parent.parent.name for p in derived.glob(f"*/{FINAL_STAGE}/manifest.json")}


def select_fights(fights: pd.DataFrame, derived: Path, n: int) -> pd.DataFrame:
    """The next n unprocessed fights, favouring fighters with the most fights.

    Repeat fighters are the most valuable data (SPEC Layer 0), so the pilot
    starts with them.
    """
    counts = pd.concat([fights["fighter_a_id"], fights["fighter_b_id"]]).value_counts()
    depth = fights["fighter_a_id"].map(counts) + fights["fighter_b_id"].map(counts)
    todo = fights.assign(depth=depth)[~fights["fight_id"].isin(done_fights(derived))]
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


def import_outputs(src: Path, derived: Path, overlays: Path, logs: Path) -> list[str]:
    """Copy a pulled job output into data/. Returns the fights that finished."""
    finished: list[str] = []
    for fight in sorted((src / "derived").glob("*")):
        if not (fight / FINAL_STAGE / "manifest.json").exists():
            continue
        dest = derived / fight.name
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(fight, dest)
        finished.append(fight.name)
    if (src / "overlays").exists():
        overlays.mkdir(parents=True, exist_ok=True)
        for clip in (src / "overlays").glob("*.mp4"):
            shutil.copy2(clip, overlays / clip.name)
    if (src / "job_log.json").exists():
        logs.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, object]] = json.loads((src / "job_log.json").read_text())
        stamp = len(list(logs.glob("job_log_*.json")))
        (logs / f"job_log_{stamp:03d}.json").write_text(json.dumps(entries, indent=2))
    return finished
