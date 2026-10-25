"""Delete every local trace of some fights: per-fight folders and files, and their
rows in shared tables, label runs and job logs.

Used when a fight must leave the dataset entirely (a junior bout, or one whose
age couldn't be confirmed). fights.csv and fighters.csv are rebuilt by
collection, not edited here, and the DVC cache is cleaned afterwards with
`dvc add data/derived` and `dvc gc -w`. Raw video is never kept, so there is
none to delete.
"""

import json
import shutil
from pathlib import Path
from typing import cast

import pandas as pd

PER_FIGHT_DIRS = ("data/derived", "data/derived_fp32", "data/crops")
"""Folders with one sub-folder per fight."""
PER_FIGHT_FILES = ("data/identity_cache/{fid}.npz", "data/overlays/{fid}.mp4")
CLIP_DIR = "data/gold/clips"
"""Clips are named <fight_id>_<frame>.mp4."""
TABLES = (
    "data/gold/clips.parquet",
    "data/gold/rules.parquet",
    "data/gold/silver.parquet",
    "data/derived/identity_quality.parquet",
)
LABEL_RUNS = "data/gold/vlm/labels_*.jsonl"
JOB_LOGS = "data/logs/job_log_*.json"


def _of(fights: set[str], value: object) -> bool:
    """True if a fight id or a clip id (<fight_id>_<frame>) belongs to these fights."""
    s = str(value)
    return s in fights or s.rsplit("_", 1)[0] in fights


def purge(fights: set[str], root: Path = Path(".")) -> dict[str, int]:
    """Remove the fights everywhere under root; returns what was removed, by kind."""
    n: dict[str, int] = {"folders": 0, "files": 0, "clips": 0, "rows": 0}
    for d in PER_FIGHT_DIRS:
        for fid in fights:
            if (root / d / fid).is_dir():
                shutil.rmtree(root / d / fid)
                n["folders"] += 1
    for pattern in PER_FIGHT_FILES:
        for fid in fights:
            f = root / pattern.format(fid=fid)
            if f.exists():
                f.unlink()
                n["files"] += 1
    if (root / CLIP_DIR).is_dir():
        for clip in (root / CLIP_DIR).glob("*.mp4"):
            if _of(fights, clip.stem):
                clip.unlink()
                n["clips"] += 1
    for t in TABLES:
        path = root / t
        if path.exists():
            df = pd.read_parquet(path)
            col = "fight_id" if "fight_id" in df else "clip_id"
            keep = ~df[col].map(lambda v: _of(fights, v))
            n["rows"] += int((~keep).sum())
            df[keep].to_parquet(path, index=False)
    for path in sorted(root.glob(LABEL_RUNS)):
        lines = path.read_text().splitlines()
        kept = [line for line in lines if not _of(fights, json.loads(line)["clip_id"])]
        n["rows"] += len(lines) - len(kept)
        path.write_text("".join(f"{line}\n" for line in kept))
    for path in sorted(root.glob(JOB_LOGS)):
        # Older logs are a list of fights, newer ones a dict with a "fights" list.
        log: dict[str, object] | list[dict[str, object]] = json.loads(path.read_text())
        rows = cast(list[dict[str, object]], log["fights"] if isinstance(log, dict) else log)
        kept_rows = [r for r in rows if not _of(fights, r["fight_id"])]
        n["rows"] += len(rows) - len(kept_rows)
        if isinstance(log, dict):
            log["fights"] = kept_rows
        else:
            log = kept_rows
        path.write_text(json.dumps(log, indent=2))
    return n
