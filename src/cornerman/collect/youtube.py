"""Fetch video metadata (never video) from YouTube with yt-dlp.

yt-dlp runs as a subprocess with JSON output rather than through its Python
API, which has no type information. Per-video metadata is cached as
`{meta_dir}/{source}/{video_id}.info.json` so reruns don't hit YouTube again.
"""

import json
import subprocess
import sys
from pathlib import Path

from cornerman.collect.registry import VideoMeta

_YT_DLP = (sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings")


def _json(args: list[str]) -> dict[str, object]:
    out = subprocess.run([*_YT_DLP, *args], capture_output=True, text=True, check=True)
    data: dict[str, object] = json.loads(out.stdout)
    return data


def list_channel(channel_url: str) -> list[tuple[str, str]]:
    """(video_id, title) for every upload on a channel's videos tab."""
    data = _json(["--flat-playlist", "-J", channel_url.rstrip("/") + "/videos"])
    entries = data.get("entries")
    if not isinstance(entries, list):
        return []
    found: list[tuple[str, str]] = []
    for e in entries:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(e, dict):
            found.append((str(e.get("id")), str(e.get("title"))))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return found


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_info(video_id: str, cache_dir: Path) -> dict[str, object]:
    path = cache_dir / f"{video_id}.info.json"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        info = _json(["--skip-download", "-J", "--no-write-comments", video_url(video_id)])
        path.write_text(json.dumps(info))
    cached: dict[str, object] = json.loads(path.read_text())
    return cached


def to_meta(info: dict[str, object], source: str) -> VideoMeta:
    video_id = str(info["id"])
    return VideoMeta(
        source=source,
        video_id=video_id,
        url=video_url(video_id),
        title=str(info.get("title") or ""),
        description=str(info.get("description") or ""),
        upload_date=str(info.get("upload_date") or ""),
        height=int(str(info.get("height") or 0)),
        duration_s=float(str(info.get("duration") or 0)),
    )
