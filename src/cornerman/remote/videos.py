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


def matches_processed(video: Path, info: dict[str, object]) -> str | None:
    """How a fresh download matches the video the pipeline processed (ingest/video.json):
    "same file" (sha256), "same frames" (frame count, rate and size, e.g. a remux), or
    None if it is a different video, in which case nothing derived lines up with it."""
    import hashlib

    from cornerman.perception.video import probe

    h = hashlib.sha256()
    with video.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() == info["sha256"]:
        return "same file"
    p = probe(video)
    same = (p.n_frames, p.width, p.height) == (info["n_frames"], info["width"], info["height"])
    return "same frames" if same and abs(p.fps - float(str(info["fps"]))) < 1e-3 else None


ARENA_SLUG = "cornerman-arena"
ARENA_INPUTS = ("ingest", "shots")
"""Stage folders the arena job needs besides the video; with them in place the runner
sees those stages as cached and runs only `arena`."""


def stage_arena(
    batch: pd.DataFrame, derived: Path, user: str, work: Path, max_height: int
) -> tuple[str, dict[str, str]]:
    """Download the batch's videos, keep those that match what was processed, upload them
    with their ingest and shots folders as one private dataset, delete the local copies.

    Returns the dataset ref and a status per fight ("same file", "same frames",
    "different video" or an error); only the first two are uploaded.
    """
    ref = f"{user}/{ARENA_SLUG}"
    folder = work / ARENA_SLUG
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    status: dict[str, str] = {}
    try:
        for fid, url in zip(batch["fight_id"], batch["url"], strict=True):
            fid = str(fid)
            video = folder / "videos" / f"{fid}.mp4"
            try:
                download(str(url), video, max_height)
                info: dict[str, object] = json.loads(
                    (derived / fid / "ingest" / "video.json").read_text()
                )
                match = matches_processed(video, info)
            except Exception as e:  # one bad download shouldn't stop the batch
                video.unlink(missing_ok=True)
                status[fid] = f"error: {e}"[:300]
                print(fid, status[fid], flush=True)
                continue
            if match is None:
                video.unlink(missing_ok=True)
                status[fid] = "different video"
            else:
                for stage in ARENA_INPUTS:
                    shutil.copytree(derived / fid / stage, folder / "derived" / fid / stage)
                status[fid] = match
            print(fid, status[fid], flush=True)
        if any(v in ("same file", "same frames") for v in status.values()):
            meta = {"title": ARENA_SLUG, "id": ref, "licenses": [{"name": "unknown"}]}
            (folder / kaggle.DATASET_METADATA).write_text(json.dumps(meta, indent=2))
            kaggle.upload_dataset(folder, ref)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return ref, status


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


CLIPS_SLUG = "cornerman-clips"


def upload_clips(clips: list[Path], user: str, work: Path) -> str:
    """Upload silver-set clips as one private dataset; the staging copy is removed.

    The local clips stay in data/gold/clips (gitignored). The dataset is deleted
    by `cornerman gold-vlm import` once the labels are home.
    """
    ref = f"{user}/{CLIPS_SLUG}"
    folder = work / CLIPS_SLUG
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    try:
        for c in clips:
            shutil.copy2(c, folder / c.name)
        meta = {"title": CLIPS_SLUG, "id": ref, "licenses": [{"name": "unknown"}]}
        (folder / kaggle.DATASET_METADATA).write_text(json.dumps(meta, indent=2))
        kaggle.upload_dataset(folder, ref)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return ref
