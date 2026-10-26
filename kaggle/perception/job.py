"""Kaggle GPU job: run the real perception pipeline on a batch of fights.

Runs as a private Kaggle script kernel. The batch, the config and the exact
commit to install are filled in at build time by `cornerman kaggle batch`, so
video links never sit in git. For each fight it downloads the video to
/kaggle/tmp (not saved as output), runs the stages, renders a short overlay for
the first few fights, deletes the video, and logs per-stage times.

Kaggle's T4 machine has two GPUs. One worker process runs per GPU, each pinned
to its card with CUDA_VISIBLE_DEVICES before torch is imported, and fights are
dealt out round-robin. Two fights at once is simpler and scales better than
splitting one fight's frames across cards.

Resuming: stage outputs land in /kaggle/working/derived as each stage finishes.
If the session dies, the laptop pulls whatever finished and the next batch
only contains fights that aren't done yet.
"""

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

FIGHTS: list[dict[str, str]] = []  # BUILD:FIGHTS
CONFIG: dict[str, object] = {}  # BUILD:CONFIG
REPO = ""  # BUILD:REPO
COMMIT = ""  # BUILD:COMMIT
OVERLAYS = 0  # BUILD:OVERLAYS

TIME_BUDGET_S = 10 * 3600
"""Stop starting new fights after this, well inside Kaggle's 12 hour limit."""
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
LOG = WORK / "job_log.json"


def install() -> None:
    pkgs = [f"cornerman[gpu,collect] @ git+{REPO}@{COMMIT}", "yt-dlp[default]", "deno"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def gpu_count() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    return max(1, sum(1 for line in out.stdout.splitlines() if line.startswith("GPU ")))


def write_config() -> Path:
    import yaml

    manifest = TMP / "manifest.json"
    TMP.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({f["fight_id"]: f["url"] for f in FIGHTS}))
    config = dict(CONFIG)
    config["paths"] = {
        "derived": str(WORK / "derived"),
        "quarantine": str(WORK / "quarantine"),
        "crops": str(WORK / "crops"),
        "raw": str(TMP / "raw"),
        "manifest": str(manifest),
        # Videos staged in the private dataset, when the batch used one.
        "video_source": "/kaggle/input",
    }
    path = TMP / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def worker(gpu: int, fight_ids: list[str], overlay_ids: list[str], cfg: str, start: float) -> None:
    # Must happen before anything imports torch in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from cornerman.config import load_settings
    from cornerman.perception.stages import REAL_PIPELINE
    from cornerman.perception.video import raw_dir
    from cornerman.pipeline.runner import run
    from cornerman.products.overlay import render

    settings = load_settings(Path(cfg))
    log: list[dict[str, object]] = []
    log_path = WORK / f"job_log_gpu{gpu}.json"
    for fid in fight_ids:
        entry: dict[str, object] = {"fight_id": fid, "gpu": gpu}
        if time.time() - start > TIME_BUDGET_S:
            entry["status"] = "not started (time budget)"
            log.append(entry)
            continue
        t0 = time.time()
        try:
            results = run(REAL_PIPELINE, settings, [fid])
            entry["stages"] = {r.stage: r.status.value for r in results}
            entry["stage_seconds"] = {r.stage: round(r.seconds, 1) for r in results}
            ok = all(r.status.value in ("ran", "cached") for r in results)
            video = next(raw_dir(settings, fid).glob("video.*"), None)
            if ok and video is not None and fid in overlay_ids:
                t1 = time.time()
                render(
                    video,
                    settings.paths.derived / fid,
                    WORK / "overlays" / f"{fid}.mp4",
                    settings.perception.fps,
                    settings.perception.overlay_seconds,
                )
                entry["overlay_seconds"] = round(time.time() - t1, 1)
            info = settings.paths.derived / fid / "ingest" / "video.json"
            if info.exists():
                entry["video_seconds"] = json.loads(info.read_text())["duration_s"]
        except Exception:
            entry["error"] = traceback.format_exc()[-3000:]
        finally:
            shutil.rmtree(raw_dir(settings, fid), ignore_errors=True)
            # One zip per fight: Kaggle's output download fetches files one by one,
            # and thousands of small crops made an import take over ten minutes.
            crops = settings.paths.crops / fid
            if crops.exists():
                shutil.make_archive(str(crops), "zip", crops)
                shutil.rmtree(crops, ignore_errors=True)
        entry["wall_seconds"] = round(time.time() - t0, 1)
        log.append(entry)
        log_path.write_text(json.dumps(log, indent=2))
        print(json.dumps(entry)[:600], flush=True)


def main() -> None:
    start = time.time()
    install()
    cfg = str(write_config())
    n = min(gpu_count(), max(1, len(FIGHTS)))
    ids = [f["fight_id"] for f in FIGHTS]
    overlay_ids = ids[:OVERLAYS]
    shares = [ids[i::n] for i in range(n)]
    print(f"{len(ids)} fights on {n} GPU(s), install took {time.time() - start:.0f}s", flush=True)

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=worker, args=(gpu, share, overlay_ids, cfg, start))
        for gpu, share in enumerate(shares)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    merged: list[dict[str, object]] = []
    for gpu in range(n):
        path = WORK / f"job_log_gpu{gpu}.json"
        if path.exists():
            merged += json.loads(path.read_text())
    summary = {"gpus": n, "total_wall_seconds": round(time.time() - start, 1), "fights": merged}
    LOG.write_text(json.dumps(summary, indent=2))
    shutil.rmtree(TMP / "raw", ignore_errors=True)


if __name__ == "__main__":
    main()
