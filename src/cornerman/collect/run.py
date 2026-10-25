"""Collect metadata from every configured source and write the local CSVs."""

import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from cornerman.collect import youtube
from cornerman.collect.adult import minor_reasons
from cornerman.collect.registry import Tables, VideoMeta, build_tables
from cornerman.collect.screen import BarReader, read_screen, screen_text
from cornerman.config import CollectConfig
from cornerman.contracts.fights import FIGHTERS_SCHEMA, FIGHTS_SCHEMA

log = logging.getLogger(__name__)


def gather(cfg: CollectConfig) -> list[VideoMeta]:
    reader = BarReader()
    videos: list[VideoMeta] = []
    for src in cfg.sources:
        uploads = youtube.list_channel(src.channel_url)
        wanted = [vid for vid, title in uploads if src.title_contains.lower() in title.lower()]
        log.info(
            "%s: %d uploads, %d match %r", src.name, len(uploads), len(wanted), src.title_contains
        )
        cache = cfg.meta_dir / src.name
        metas = [youtube.to_meta(youtube.fetch_info(v, cache), src.name) for v in wanted]
        videos.extend(with_screen(m, cache, reader) for m in metas)
    return videos


def with_screen(m: VideoMeta, cache: Path, reader: BarReader) -> VideoMeta:
    """Add the score-bar text. Videos the metadata already rules out skip the fetch."""
    if minor_reasons(f"{m.title}\n{m.description}\n{m.tags}"):
        return m
    return replace(
        m, screen=screen_text(read_screen(m.video_id, m.url, m.duration_s, cache, reader))
    )


def load_verdicts(path: Path) -> dict[str, str]:
    """video_id -> results-check verdict (adult, junior, unclear); empty if no file."""
    if not path.exists():
        return {}
    v = pd.read_csv(path)
    return dict(zip(v["video_id"].astype(str), v["verdict"].astype(str), strict=True))


def age_hold(
    fights: pd.DataFrame, videos: list[VideoMeta], verdicts: dict[str, str]
) -> pd.DataFrame:
    """Kept fights whose score bar never named the division and that no results check
    has cleared: they go on hold (contracts.hold) until one does."""
    screens = {v.video_id: v.screen for v in videos}
    held = [
        fid
        for fid, vid in zip(fights["fight_id"], fights["video_id"], strict=True)
        if "SENIOR" not in screens.get(str(vid), "").upper() and verdicts.get(str(vid)) != "adult"
    ]
    reason = "division not shown on screen; not yet checked against the event's results"
    return pd.DataFrame({"fight_id": held, "reason": reason})


def collect(cfg: CollectConfig) -> Tables:
    existing = pd.read_csv(cfg.fighters_csv) if cfg.fighters_csv.exists() else None
    verdicts = load_verdicts(cfg.age_verdicts_csv)
    videos = gather(cfg)
    tables = build_tables(videos, cfg.min_height, existing, verdicts)
    fights = FIGHTS_SCHEMA.validate(tables.fights)
    fighters = FIGHTERS_SCHEMA.validate(tables.fighters)
    cfg.fights_csv.parent.mkdir(parents=True, exist_ok=True)
    fights.to_csv(cfg.fights_csv, index=False)
    fighters.to_csv(cfg.fighters_csv, index=False)
    tables.skipped.to_csv(cfg.skipped_csv, index=False)
    age_hold(fights, videos, verdicts).to_csv(cfg.hold_csv, index=False)
    return Tables(fights=fights, fighters=fighters, skipped=tables.skipped)
