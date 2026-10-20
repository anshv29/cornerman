"""Kaggle GPU job: label silver-set clips with open-weight VLMs (labeller B).

Runs as a private Kaggle script kernel on the T4 x2 machine. The clips come from
a private dataset (deleted by `cornerman gold-vlm import` afterwards). The models,
the clips to label and the exact commit to install are filled in at build time by
`cornerman gold-vlm push`.

A model marked "per_gpu" (one that fits a single T4, like the 4B) runs as one
copy per GPU, each in its own process on half the clips. Otherwise the model is
spread over both T4s with device_map="auto" (the 8B in fp16 needs both). Per model
the log records seconds per answer and peak memory per GPU, which is what the
model choice needs.
"""

import gc
import json
import multiprocessing as mp
import os
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


def worker(gpu: int, model: dict[str, object], clips: list[str], out: str) -> None:
    """One copy of the model on one GPU (set before torch is imported)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    from cornerman.gold.vlm_run import VlmSpec, label_clips

    rows = label_clips(VlmSpec(**model), [Path(c) for c in clips], device_map="cuda:0")  # pyright: ignore[reportArgumentType]
    with Path(out).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    Path(out).with_suffix(".peak").write_text(f"{peak:.2f}")


def run_per_gpu(model: dict[str, object], clips: list[Path], n_gpu: int) -> list[dict[str, object]]:
    ctx = mp.get_context("spawn")
    outs = [str(WORK / f"part_{g}.jsonl") for g in range(n_gpu)]
    procs = [
        ctx.Process(target=worker, args=(g, model, [str(c) for c in clips[g::n_gpu]], outs[g]))
        for g in range(n_gpu)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    rows: list[dict[str, object]] = []
    for o in outs:
        if Path(o).exists():
            rows += [json.loads(line) for line in Path(o).read_text().splitlines()]
    return rows


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
    n_gpu = torch.cuda.device_count()
    for m in MODELS:
        m = dict(m)
        per_gpu = bool(m.pop("per_gpu", False))
        spec = VlmSpec(**m)  # pyright: ignore[reportArgumentType]
        entry: dict[str, object] = {
            "model": spec.name,
            "revision": spec.revision,
            "per_gpu": per_gpu,
        }
        t0 = time.time()
        try:
            if per_gpu and n_gpu > 1:
                rows = run_per_gpu(m, clips, n_gpu)
                entry["peak_gb"] = [
                    float(Path(WORK / f"part_{g}.peak").read_text())
                    for g in range(n_gpu)
                    if (WORK / f"part_{g}.peak").exists()
                ]
            else:
                torch.cuda.init()  # the peak-memory counters need CUDA set up first
                for d in range(n_gpu):
                    torch.cuda.reset_peak_memory_stats(torch.device("cuda", d))
                rows = label_clips(spec, clips)
                entry["peak_gb"] = [
                    round(torch.cuda.max_memory_allocated(d) / 1e9, 2) for d in range(n_gpu)
                ]
            with out.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            secs = [float(r["seconds"]) for r in rows if not r["error"]]  # pyright: ignore[reportArgumentType]
            entry["answers"] = len(rows)
            entry["errors"] = sum(1 for r in rows if r["error"])
            entry["median_answer_s"] = sorted(secs)[len(secs) // 2] if secs else None
        except Exception:
            entry["error"] = traceback.format_exc()[-3000:]
        entry["wall_s"] = round(time.time() - t0, 1)
        log.append(entry)
        print(json.dumps(entry)[:1500], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    summary = {"total_wall_s": round(time.time() - start, 1), "models": log}
    (WORK / "vlm_log.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
