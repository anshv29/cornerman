"""Kaggle GPU job: label silver-set clips with open-weight VLMs (labeller B).

Runs as a private Kaggle script kernel on the T4 x2 machine. The clips come from
a private dataset (deleted by `cornerman gold-vlm import` afterwards). The models,
the clips to label and the exact commit to install are filled in at build time by
`cornerman gold-vlm push`.

Each model runs on its own with device_map="auto", so an 8B model in fp16 is
split over both T4s and a 4B one may be too. Per model the log records load time,
seconds per answer and peak memory per GPU, which is what the model choice needs.
"""

import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = ""  # BUILD:REPO
COMMIT = ""  # BUILD:COMMIT
MODELS: list[dict[str, object]] = []  # BUILD:MODELS
CLIP_IDS: list[str] = []  # BUILD:CLIP_IDS

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def install() -> None:
    pkgs = [f"cornerman @ git+{REPO}@{COMMIT}", "transformers==5.20.1", "accelerate==1.15.0"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def main() -> None:
    start = time.time()
    install()
    import torch

    from cornerman.gold.vlm_run import VlmSpec, label_clips

    found = {p.stem: p for p in INPUT.rglob("*.mp4")}
    clips = [found[c] for c in CLIP_IDS if c in found]
    print(f"{len(clips)} of {len(CLIP_IDS)} clips found, install {time.time() - start:.0f}s")
    log: list[dict[str, object]] = []
    out = WORK / "vlm_labels.jsonl"
    for m in MODELS:
        spec = VlmSpec(**m)  # pyright: ignore[reportArgumentType]
        entry: dict[str, object] = {"model": spec.name, "revision": spec.revision}
        t0 = time.time()
        torch.cuda.init()  # the peak-memory counters need CUDA set up first
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(torch.device("cuda", d))
        try:
            rows = label_clips(spec, clips)
            with out.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            secs = [float(r["seconds"]) for r in rows if not r["error"]]  # pyright: ignore[reportArgumentType]
            entry["answers"] = len(rows)
            entry["errors"] = sum(1 for r in rows if r["error"])
            entry["median_answer_s"] = sorted(secs)[len(secs) // 2] if secs else None
        except Exception:
            entry["error"] = traceback.format_exc()[-3000:]
        entry["peak_gb"] = [
            round(torch.cuda.max_memory_allocated(d) / 1e9, 2)
            for d in range(torch.cuda.device_count())
        ]
        entry["wall_s"] = round(time.time() - t0, 1)
        log.append(entry)
        print(json.dumps(entry)[:1500], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    summary = {"total_wall_s": round(time.time() - start, 1), "models": log}
    (WORK / "vlm_log.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
