"""Decode each video once.

H.264 can't skip frames cheaply: to get every 5th frame the decoder still
decodes all of them. Before this module the shots, tracking and pose stages
each decoded the whole video, so every fight was decoded three times.

Now one ffmpeg pass writes two things into the fight's raw folder:
- tiny 48x27 RGB frames for shot detection, at about `shot_fps` (every k-th frame);
- the frames tracking and pose use (every `step`-th frame, about `fps`), as
  high-quality JPEGs named by their frame index.
The cache lives next to the raw video and is deleted with it (hard rule 2), so
it never reaches data/derived or DVC. A meta file records the settings it was
made with; if they change, or the cache is missing, it's rebuilt.
"""

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from cornerman.config import Settings
from cornerman.perception.shots import TRANSNET_SIZE
from cornerman.perception.video import Frame, ensure_video, frame_step, prefetch, raw_dir
from cornerman.pipeline.stage import QuarantineError

JPEG_QUALITY = 2
"""ffmpeg -q:v scale, 2 is near the best (1 to 31). Keeps re-encoding error small."""


def ffmpeg_exe() -> str:
    """System ffmpeg if there is one, else the static binary from imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg  # pyright: ignore[reportMissingTypeStubs]

    exe: str = imageio_ffmpeg.get_ffmpeg_exe()  # pyright: ignore[reportUnknownMemberType]
    return exe


@dataclass(frozen=True)
class FrameCache:
    folder: Path
    step: int
    """Full frames are every step-th frame of the video."""
    shot_step: int
    """Small frames for shot detection are every shot_step-th frame."""

    @property
    def small_path(self) -> Path:
        return self.folder / "small.rgb"

    @property
    def frames_dir(self) -> Path:
        return self.folder / "frames"


def _settings_key(step: int, shot_step: int) -> dict[str, object]:
    return {"step": step, "shot_step": shot_step, "size": TRANSNET_SIZE, "q": JPEG_QUALITY}


def ensure_frames(settings: Settings, fight_id: str, source_fps: float) -> FrameCache:
    p = settings.perception
    step = frame_step(source_fps, p.fps)
    shot_step = frame_step(source_fps, p.shot_fps)
    cache = FrameCache(raw_dir(settings, fight_id) / "cache", step, shot_step)
    meta = cache.folder / "meta.json"
    want = _settings_key(step, shot_step)
    if meta.exists() and json.loads(meta.read_text()) == json.loads(json.dumps(want)):
        return cache

    video = ensure_video(settings, fight_id)
    shutil.rmtree(cache.folder, ignore_errors=True)
    cache.frames_dir.mkdir(parents=True)
    w, h = TRANSNET_SIZE
    graph = (
        f"[0:v]split=2[a][b];"
        f"[a]select='not(mod(n\\,{shot_step}))',scale={w}:{h}[s];"
        f"[b]select='not(mod(n\\,{step}))'[f]"
    )
    cmd = [
        ffmpeg_exe(), "-loglevel", "error", "-i", str(video), "-filter_complex", graph,
        "-map", "[s]", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24",
        str(cache.small_path),
        "-map", "[f]", "-fps_mode", "passthrough", "-q:v", str(JPEG_QUALITY),
        "-start_number", "0", str(cache.frames_dir / "%07d.jpg"),
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise QuarantineError(f"decode failed: {result.stderr[-500:]}")
    meta.write_text(json.dumps(want))
    (cache.folder / "cache.json").write_text(json.dumps({**asdict(cache), "folder": "."}))
    return cache


def small_frames(cache: FrameCache) -> np.ndarray:
    """All small frames as (n, 27, 48, 3) RGB."""
    w, h = TRANSNET_SIZE
    return np.fromfile(cache.small_path, dtype=np.uint8).reshape(-1, h, w, 3)


def cached_indices(cache: FrameCache) -> list[int]:
    """Video frame index of every cached full frame, in order."""
    n = len(list(cache.frames_dir.glob("*.jpg")))
    return [k * cache.step for k in range(n)]


def iter_cached(
    cache: FrameCache, wanted: set[int] | None = None, depth: int = 0
) -> Iterator[tuple[int, Frame]]:
    """(video frame index, BGR frame) for the cached frames, optionally only `wanted` ones."""

    def read() -> Iterator[tuple[int, Frame]]:
        for idx in cached_indices(cache):
            if wanted is not None and idx not in wanted:
                continue
            frame = cv2.imread(str(cache.frames_dir / f"{idx // cache.step:07d}.jpg"))
            if frame is None:
                raise QuarantineError(f"missing cached frame {idx}")
            yield idx, frame.astype(np.uint8)  # pyright: ignore[reportReturnType]

    return prefetch(read(), depth)
