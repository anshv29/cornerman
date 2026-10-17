"""Getting a fight's video onto local disk, probing it, and reading frames.

Raw video lives in `paths.raw/{fight_id}/` and is deleted once the fight is
processed (hard rule 2). Stages call `ensure_video`, which downloads again if a
later rerun needs frames after the file was deleted.
"""

import json
import queue
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cornerman.config import Settings
from cornerman.pipeline.stage import QuarantineError

Frame = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps


def manifest_url(settings: Settings, fight_id: str) -> str:
    urls: dict[str, str] = json.loads(settings.paths.manifest.read_text())
    if fight_id not in urls:
        raise QuarantineError(f"no URL for {fight_id} in {settings.paths.manifest}")
    return urls[fight_id]


def raw_dir(settings: Settings, fight_id: str) -> Path:
    return settings.paths.raw / fight_id


def ensure_video(settings: Settings, fight_id: str) -> Path:
    """Path to the fight's video: already in raw/, staged in video_source, or downloaded."""
    folder = raw_dir(settings, fight_id)
    found = sorted(folder.glob("video.*")) if folder.exists() else []
    if found:
        return found[0]
    folder.mkdir(parents=True, exist_ok=True)
    src = settings.paths.video_source
    staged = sorted(src.rglob(f"{fight_id}.*")) if src is not None and src.exists() else []
    if staged:
        # The source is read-only (a Kaggle dataset); link it rather than copy.
        link = folder / f"video{staged[0].suffix}"
        link.symlink_to(staged[0].resolve())
        return link
    h = settings.perception.max_height
    cmd = [
        sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", f"bv*[height<={h}][ext=mp4]/bv*[height<={h}]",
        "-o", str(folder / "video.%(ext)s"),
        manifest_url(settings, fight_id),
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    found = sorted(folder.glob("video.*"))
    if result.returncode != 0 or not found:
        raise QuarantineError(f"download failed: {result.stderr[-500:]}")
    return found[0]


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise QuarantineError(f"can't open {path.name}")
        info = VideoInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()
    if info.fps <= 0 or info.n_frames <= 0:
        raise QuarantineError(f"bad video properties: {info}")
    return info


def frame_step(source_fps: float, target_fps: float) -> int:
    """Keep every k-th frame so the sampled rate is as close to the target as possible."""
    return max(1, round(source_fps / target_fps))


def iter_frames(
    path: Path,
    step: int = 1,
    size: tuple[int, int] | None = None,
    until_frame: int | None = None,
) -> Iterator[tuple[int, Frame]]:
    """Yield (frame index, BGR frame) for every `step`-th frame, optionally resized to (w, h).

    Frames we skip are only grabbed, not decoded into images, which is much faster.
    """
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while until_frame is None or idx < until_frame:
            if idx % step:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                if size is not None:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                yield idx, frame.astype(np.uint8)  # pyright: ignore[reportReturnType]
            idx += 1
    finally:
        cap.release()


_DONE = object()


def prefetch[T](items: Iterator[T], depth: int) -> Iterator[T]:
    """Run an iterator on a background thread, keeping up to `depth` items ready.

    Decoding video is CPU work and inference is GPU work; this lets them overlap
    instead of taking turns. OpenCV releases the GIL while decoding, so a
    thread is enough.
    """
    if depth <= 0:
        yield from items
        return
    q: queue.Queue[object] = queue.Queue(maxsize=depth)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for item in items:
                q.put(item)
        except BaseException as e:  # re-raised in the consumer
            errors.append(e)
        finally:
            q.put(_DONE)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not _DONE:
        yield item  # pyright: ignore[reportReturnType]
    if errors:
        raise errors[0]
