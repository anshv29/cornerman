"""Kaggle smoke test: does the GPU show up, and can yt-dlp download from YouTube here?

Runs as a private Kaggle script kernel. Video links are filled in at build time
(see cornerman.remote.kaggle.build_kernel) so they never sit in git. Clips are
written to /kaggle/tmp, which is not saved as kernel output, and deleted right
after each try. Only result.json lands in /kaggle/working.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

VIDEO_URLS: list[str] = []  # BUILD:VIDEO_URLS

TMP = Path("/kaggle/tmp/smoke")
OUT = Path("/kaggle/working/result.json")
FORMAT = "bv*[height<=720]+ba/b[height<=720]"


def sh(cmd: list[str], timeout: int = 900) -> dict[str, object]:
    start = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        code, out, err = -1, "", "timeout"
    return {
        "ok": code == 0,
        "seconds": round(time.time() - start, 1),
        "stdout_tail": out[-1500:],
        "stderr_tail": err[-1500:],
    }


def clip_size() -> int:
    return sum(p.stat().st_size for p in TMP.glob("*") if p.is_file())


def try_url(url: str, full: bool) -> dict[str, object]:
    TMP.mkdir(parents=True, exist_ok=True)
    meta = sh([sys.executable, "-m", "yt_dlp", "--skip-download", "--print", "%(height)s", url])
    out = str(TMP / "clip.%(ext)s")
    section = [] if full else ["--download-sections", "*0-30"]
    dl = sh([sys.executable, "-m", "yt_dlp", "-f", FORMAT, *section, "-o", out, url])
    size = clip_size()
    shutil.rmtree(TMP, ignore_errors=True)
    return {"metadata": meta, "download": dl, "full": full, "bytes": size}


def main() -> None:
    report: dict[str, object] = {"python": sys.version}
    report["nvidia_smi"] = sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = torch.cuda.is_available()
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        report["torch"] = None
    report["ffmpeg"] = shutil.which("ffmpeg")
    report["install"] = sh(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp[default]", "deno"]
    )
    report["yt_dlp"] = sh([sys.executable, "-m", "yt_dlp", "--version"])["stdout_tail"]
    # First link: a full download (gives a speed estimate). The rest: 30 s sections.
    report["tries"] = [try_url(u, full=(i == 0)) for i, u in enumerate(VIDEO_URLS)]
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
