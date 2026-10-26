"""Kaggle CPU job: run only the `arena` stage (mat outline, round clock, thumbnails).

Runs as a private Kaggle script kernel without a GPU, so it uses no GPU quota.
`cornerman kaggle arena-batch` uploads a private dataset holding each fight's
video ({fight_id}.mp4, checked on the laptop to be the video that was processed)
and its ingest and shots stage files, flat (the CLI skips folders).
With those in place the runner sees ingest and shots as cached and runs only
arena. The fight list, config and commit are filled in at build time.

One worker process per CPU core; fights are dealt out round-robin. Outputs land
in /kaggle/working/derived as each fight finishes, so a session that dies loses
only the fights in progress.
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

FIGHTS: list[str] = []  # BUILD:FIGHTS
CONFIG: dict[str, object] = {}  # BUILD:CONFIG
REPO = ""  # BUILD:REPO
COMMIT = ""  # BUILD:COMMIT

TIME_BUDGET_S = 10 * 3600
"""Stop starting new fights after this, well inside Kaggle's 12 hour limit."""
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
INPUT = Path("/kaggle/input")


def install() -> None:
    pkgs = [f"cornerman[gpu,collect] @ git+{REPO}@{COMMIT}"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def stage_inputs() -> None:
    """Put each fight's ingest and shots files, uploaded flat as
    {fight_id}__{stage}__{file}, back into derived/{fight_id}/{stage}/."""
    for f in INPUT.rglob("*__*__*"):
        fid, stage, name = f.name.split("__", 2)
        if fid in FIGHTS:
            dest = WORK / "derived" / fid / stage
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / name)


def write_config() -> Path:
    import yaml

    config = dict(CONFIG)
    config["paths"] = {
        "derived": str(WORK / "derived"),
        "quarantine": str(WORK / "quarantine"),
        "crops": str(TMP / "crops"),
        "thumbs": str(WORK / "thumbs"),
        "raw": str(TMP / "raw"),
        "manifest": str(TMP / "manifest.json"),
        "video_source": str(INPUT),
    }
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "manifest.json").write_text("{}")
    path = TMP / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def worker(n: int, fight_ids: list[str], cfg: str, start: float) -> None:
    # One OCR and decode process per core; keep each to one thread.
    os.environ["OMP_NUM_THREADS"] = "1"
    from cornerman.config import load_settings
    from cornerman.perception.stages import REAL_PIPELINE
    from cornerman.perception.video import raw_dir
    from cornerman.pipeline.runner import run

    settings = load_settings(Path(cfg))
    log: list[dict[str, object]] = []
    for fid in fight_ids:
        entry: dict[str, object] = {"fight_id": fid, "worker": n}
        if time.time() - start > TIME_BUDGET_S:
            entry["status"] = "not started (time budget)"
            log.append(entry)
            continue
        t0 = time.time()
        try:
            results = run(REAL_PIPELINE, settings, [fid], target="arena")
            entry["stages"] = {r.stage: r.status.value for r in results}
        except Exception:
            entry["error"] = traceback.format_exc()[-3000:]
        finally:
            shutil.rmtree(raw_dir(settings, fid), ignore_errors=True)
            thumbs = settings.paths.thumbs / fid
            if thumbs.exists():
                shutil.make_archive(str(thumbs), "zip", thumbs)
                shutil.rmtree(thumbs, ignore_errors=True)
        entry["wall_seconds"] = round(time.time() - t0, 1)
        log.append(entry)
        (WORK / f"job_log_w{n}.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(entry)[:600], flush=True)


def main() -> None:
    start = time.time()
    install()
    stage_inputs()
    cfg = str(write_config())
    n = max(1, min(os.cpu_count() or 1, len(FIGHTS)))
    shares = [FIGHTS[i::n] for i in range(n)]
    print(f"{len(FIGHTS)} fights on {n} CPU workers, setup took {time.time() - start:.0f}s")

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(i, s, cfg, start)) for i, s in enumerate(shares)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    merged: list[dict[str, object]] = []
    for i in range(n):
        path = WORK / f"job_log_w{i}.json"
        if path.exists():
            merged += json.loads(path.read_text())
    summary = {"workers": n, "total_wall_seconds": round(time.time() - start, 1), "fights": merged}
    (WORK / "job_log.json").write_text(json.dumps(summary, indent=2))
    shutil.rmtree(TMP / "raw", ignore_errors=True)


if __name__ == "__main__":
    main()
