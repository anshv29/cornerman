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


def collect(cfg: CollectConfig) -> Tables:
    existing = pd.read_csv(cfg.fighters_csv) if cfg.fighters_csv.exists() else None
    tables = build_tables(gather(cfg), cfg.min_height, existing)
    fights = FIGHTS_SCHEMA.validate(tables.fights)
    fighters = FIGHTERS_SCHEMA.validate(tables.fighters)
    cfg.fights_csv.parent.mkdir(parents=True, exist_ok=True)
    fights.to_csv(cfg.fights_csv, index=False)
    fighters.to_csv(cfg.fighters_csv, index=False)
    tables.skipped.to_csv(cfg.skipped_csv, index=False)
    return Tables(fights=fights, fighters=fighters, skipped=tables.skipped)
