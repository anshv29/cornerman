"""Fallback for when YouTube blocks Kaggle: download on the laptop, upload privately.

Videos are downloaded into data/raw/upload/ (gitignored), uploaded as one
private Kaggle dataset, and deleted from the laptop straight after the upload.
The dataset itself is deleted by `cornerman kaggle import` once the batch's
results are home, so raw video only exists while it is being processed.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from cornerman.remote import kaggle

DATASET_SLUG = "cornerman-videos"


def dataset_ref(user: str) -> str:
    return f"{user}/{DATASET_SLUG}"


def download(url: str, dest: Path, max_height: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", f"bv*[height<={max_height}][ext=mp4]/bv*[height<={max_height}]",
        "--remux-video", "mp4", "-o", str(dest.with_suffix(".%(ext)s")), url,
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        time.sleep(30)  # YouTube sometimes refuses briefly; one retry
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"download failed for {dest.stem}: {result.stderr[-800:]}")
    return dest


def stage_and_upload(batch: pd.DataFrame, user: str, work: Path, max_height: int) -> str:
    """Download the batch's videos, upload them as a private dataset, delete the local copies."""
    ref = dataset_ref(user)
    folder = work / DATASET_SLUG
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    try:
        for fid, url in zip(batch["fight_id"], batch["url"], strict=True):
            download(str(url), folder / f"{fid}.mp4", max_height)
        meta = {"title": DATASET_SLUG, "id": ref, "licenses": [{"name": "unknown"}]}
        (folder / kaggle.DATASET_METADATA).write_text(json.dumps(meta, indent=2))
        kaggle.upload_dataset(folder, ref)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return ref
