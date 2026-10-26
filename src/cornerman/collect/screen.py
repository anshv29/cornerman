"""Read the on-screen fight graphics, so a junior bout can't pass as adult.

IMMAF runs junior divisions inside its main championships, and a video title
like "A vs. B | FULL FIGHT | 2023 IMMAF World Championships" doesn't say which
division the bout is in. The score bar does ("JUNIORS / MEN / M: BANTAMWEIGHT").
So for every candidate video we grab a few single frames straight from YouTube
(no download), OCR the bottom quarter where the score bar sits, and feed the
text to the same check as titles (adult.minor_reasons).

A frame can miss the bar (a replay, a break, a close-up), so we read several
spread over the fight. The result is cached as <video_id>.screen.json next to
the video's info.json. Frames are not kept.

OCR is RapidOCR (Apache 2.0, PP-OCR models, ONNX on CPU), about a second per
frame. See DECISIONS.md, "Adult bouts only".
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

FRACTIONS = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
"""Where in the video to grab frames, as fractions of its duration."""
BAR_FROM = 0.75
"""The score bar is in the bottom quarter of the frame."""
SCREEN_SUFFIX = ".screen.json"


def stream_url(url: str, max_height: int = 720) -> str:
    cmd = [
        sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-playlist", "-g",
        # H.264 first: the bundled ffmpeg can't decode some of the other codecs.
        "-f", f"bv*[height<={max_height}][vcodec^=avc1]/bv*[height<={max_height}][ext=mp4]",
        url,
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"no stream for {url}: {result.stderr[-400:]}")
    return result.stdout.strip().splitlines()[0]


def grab_frames(stream: str, duration_s: float, out: Path) -> list[Path]:
    """One frame at each of FRACTIONS of the video; frames that fail are left out."""
    from cornerman.perception.decode import ffmpeg_exe

    paths: list[Path] = []
    for i, q in enumerate(FRACTIONS):
        path = out / f"{i}.jpg"
        cmd = [ffmpeg_exe(), "-v", "error", "-y", "-ss", f"{duration_s * q:.1f}", "-i", stream,
               "-frames:v", "1", str(path)]  # fmt: skip
        subprocess.run(cmd, capture_output=True, check=False)
        if path.exists() and path.stat().st_size > 0:
            paths.append(path)
    return paths


class BarReader:
    """OCR over the score-bar strip. The model loads once."""

    def __init__(self) -> None:
        from rapidocr import RapidOCR  # pyright: ignore[reportMissingTypeStubs]

        self._engine: Any = RapidOCR()

    def __call__(self, frame: Path) -> list[str]:
        import cv2

        img = cv2.imread(str(frame))
        if img is None:
            return []
        return self.read_image(img)

    def read_image(self, img: "np.ndarray", upscale: float = 2.0) -> list[str]:
        """OCR lines in the bottom quarter of a BGR frame (upscaled: the division line is small)."""
        import cv2

        strip = img[int(img.shape[0] * BAR_FROM) :]
        big = cv2.resize(strip, None, fx=upscale, fy=upscale) if upscale != 1.0 else strip
        txts: list[object] = list(self._engine(big).txts or [])
        return [str(t) for t in txts]


def read_screen(
    video_id: str, url: str, duration_s: float, cache_dir: Path, reader: BarReader
) -> dict[str, object]:
    """{"frames": frames read, "texts": OCR lines per frame}, cached per video."""
    cache = cache_dir / f"{video_id}{SCREEN_SUFFIX}"
    if cache.exists():
        cached: dict[str, object] = json.loads(cache.read_text())
        return cached
    with tempfile.TemporaryDirectory() as tmp:
        frames = grab_frames(stream_url(url), duration_s, Path(tmp))
        texts = [reader(f) for f in frames]
    if not frames:
        # Never pass a video whose screen couldn't be read, and don't cache the failure.
        raise RuntimeError(f"no frames read for {video_id}")
    result: dict[str, object] = {"frames": len(frames), "texts": texts}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def screen_text(result: dict[str, object]) -> str:
    texts: Any = result.get("texts") or []
    return "\n".join(" ".join(str(t) for t in lines) for lines in texts)
