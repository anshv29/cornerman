"""Kaggle GPU job: run the real perception pipeline on a batch of fights.

Runs as a private Kaggle script kernel. The batch, the config and the exact
commit to install are filled in at build time by `cornerman kaggle batch`, so
video links never sit in git. For each fight it downloads the video to
/kaggle/tmp (not saved as output), runs the stages, renders a short overlay for
the first few fights, deletes the video, and appends a line to job_log.json.

Resuming: stage outputs land in /kaggle/working/derived as each stage finishes.
If the session dies, the laptop pulls whatever finished and the next batch
only contains fights that aren't done yet.
"""

import json
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
    pkgs = [f"cornerman[gpu] @ git+{REPO}@{COMMIT}", "yt-dlp[default]", "deno"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def main() -> None:
    start = time.time()
    install()
    import yaml

    from cornerman.config import load_settings
    from cornerman.perception.stages import REAL_PIPELINE
    from cornerman.perception.video import raw_dir
    from cornerman.pipeline.runner import run
    from cornerman.products.overlay import render

    manifest = TMP / "manifest.json"
    TMP.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({f["fight_id"]: f["url"] for f in FIGHTS}))
    config = dict(CONFIG)
    config["paths"] = {
        "derived": str(WORK / "derived"),
        "quarantine": str(WORK / "quarantine"),
        "raw": str(TMP / "raw"),
        "manifest": str(manifest),
    }
    cfg_path = TMP / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    settings = load_settings(cfg_path)

    log: list[dict[str, object]] = []
    overlays = 0
    for f in FIGHTS:
        fid = f["fight_id"]
        if time.time() - start > TIME_BUDGET_S:
            log.append({"fight_id": fid, "status": "not started (time budget)"})
            continue
        t0 = time.time()
        entry: dict[str, object] = {"fight_id": fid}
        try:
            results = run(REAL_PIPELINE, settings, [fid])
            entry["stages"] = {r.stage: r.status.value for r in results}
            ok = all(r.status.value in ("ran", "cached") for r in results)
            video = next(raw_dir(settings, fid).glob("video.*"), None)
            if ok and video is not None and overlays < OVERLAYS:
                render(
                    video,
                    settings.paths.derived / fid,
                    WORK / "overlays" / f"{fid}.mp4",
                    settings.perception.fps,
                    settings.perception.overlay_seconds,
                )
                overlays += 1
            info = settings.paths.derived / fid / "ingest" / "video.json"
            if info.exists():
                entry["video_seconds"] = json.loads(info.read_text())["duration_s"]
        except Exception:
            entry["error"] = traceback.format_exc()[-3000:]
        finally:
            shutil.rmtree(raw_dir(settings, fid), ignore_errors=True)
        entry["wall_seconds"] = round(time.time() - t0, 1)
        log.append(entry)
        LOG.write_text(json.dumps(log, indent=2))
        print(json.dumps(entry)[:500], flush=True)

    LOG.write_text(json.dumps(log, indent=2))
    shutil.rmtree(TMP / "raw", ignore_errors=True)


if __name__ == "__main__":
    main()
