"""Turn video metadata into the fights table and the fighter registry.

Order of filters, each logged with a reason in the skipped table:
1. title doesn't parse as a single fight;
2. might involve a minor (see adult.py), from the title, description, tags
   and playlist, and the on-screen graphics (see screen.py). Every bout of any
   fighter seen in such a video is dropped too. This runs before the resolution check so a
   low-resolution youth video still marks its fighters;
3. below the minimum resolution;
4. duplicates: same video, or the same pairing at the same event and meeting.

Fighter IDs (F0001, ...) are anonymous and stable: fighters already in the
registry keep their ID, new ones get the next number. Fight IDs are a hash of
the source and video ID, so they don't reveal the video either. Real names
only appear in the local CSVs, never in anything that leaves the laptop.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from cornerman.collect.adult import minor_reasons
from cornerman.collect.titles import name_key, parse_title


@dataclass(frozen=True)
class VideoMeta:
    source: str
    video_id: str
    url: str
    title: str
    description: str
    upload_date: str
    height: int
    duration_s: float
    tags: str = ""
    """YouTube tags and playlist title, which can name the event ("Youth World Championships")."""
    screen: str = ""
    """OCR text of the score bar from a few frames (screen.py); empty if not read."""


@dataclass(frozen=True)
class Tables:
    fights: pd.DataFrame
    fighters: pd.DataFrame
    skipped: pd.DataFrame


FIGHT_COLUMNS = [
    "fight_id",
    "source",
    "video_id",
    "url",
    "event",
    "year",
    "meeting",
    "fighter_a_id",
    "fighter_b_id",
    "fighter_a_name",
    "fighter_b_name",
    "weight_class",
    "result",
    "method",
    "round",
    "time",
    "height",
    "duration_s",
    "upload_date",
    "title",
]
FIGHTER_COLUMNS = ["fighter_id", "name", "name_key", "variants", "n_fights"]
SKIPPED_COLUMNS = ["source", "video_id", "title", "reason"]


def fight_id(source: str, video_id: str) -> str:
    return "ft_" + hashlib.sha256(f"{source}:{video_id}".encode()).hexdigest()[:10]


def build_tables(
    videos: Iterable[VideoMeta],
    min_height: int,
    existing_fighters: pd.DataFrame | None = None,
) -> Tables:
    skipped: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    minor_keys: set[str] = set()

    def skip(v: VideoMeta, reason: str) -> None:
        skipped.append(
            {"source": v.source, "video_id": v.video_id, "title": v.title, "reason": reason}
        )

    for v in videos:
        parsed = parse_title(v.title)
        if parsed is None:
            skip(v, "title is not a single fight")
            continue
        reasons = [
            *minor_reasons(f"{v.title}\n{v.description}"),
            *(f"tags: {r}" for r in minor_reasons(v.tags)),
            *(f"on screen: {r}" for r in minor_reasons(v.screen)),
        ]
        if reasons:
            minor_keys.update({name_key(parsed.fighter_a), name_key(parsed.fighter_b)})
            skip(v, "might involve a minor: " + ", ".join(reasons))
            continue
        if v.height < min_height:
            skip(v, f"resolution {v.height}p is below {min_height}p")
            continue
        rows.append(
            {
                "fight_id": fight_id(v.source, v.video_id),
                "source": v.source,
                "video_id": v.video_id,
                "url": v.url,
                "event": parsed.event,
                "year": parsed.year,
                "meeting": parsed.meeting,
                "fighter_a_name": parsed.fighter_a,
                "fighter_b_name": parsed.fighter_b,
                "key_a": name_key(parsed.fighter_a),
                "key_b": name_key(parsed.fighter_b),
                "height": v.height,
                "duration_s": v.duration_s,
                "upload_date": v.upload_date,
                "title": v.title,
            }
        )

    kept: list[dict[str, object]] = []
    seen_videos: set[tuple[object, object]] = set()
    seen_bouts: set[tuple[object, ...]] = set()
    # Highest resolution first, so the copy we keep from a duplicate is the best one.
    for r in sorted(rows, key=lambda r: (-int(str(r["height"])), str(r["video_id"]))):
        v = _row_video(r)
        if r["key_a"] in minor_keys or r["key_b"] in minor_keys:
            skip(v, "a fighter also appears in a bout that might involve a minor")
            continue
        bout = (r["event"], r["meeting"], *sorted([str(r["key_a"]), str(r["key_b"])]))
        if (r["source"], r["video_id"]) in seen_videos or bout in seen_bouts:
            skip(v, "duplicate")
            continue
        seen_videos.add((r["source"], r["video_id"]))
        seen_bouts.add(bout)
        kept.append(r)

    fighters = _assign_ids(kept, existing_fighters)
    ids = dict(zip(fighters["name_key"], fighters["fighter_id"], strict=True))
    fights = pd.DataFrame(kept, columns=[*FIGHT_COLUMNS, "key_a", "key_b"])
    fights["fighter_a_id"] = fights["key_a"].map(ids)
    fights["fighter_b_id"] = fights["key_b"].map(ids)
    for col in ("weight_class", "result", "method", "round", "time"):
        fights[col] = None
    fights = fights[FIGHT_COLUMNS].sort_values(["year", "event", "fight_id"]).reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped, columns=SKIPPED_COLUMNS).sort_values(
        ["source", "video_id"], ignore_index=True
    )
    return Tables(fights=fights, fighters=fighters, skipped=skipped_df)


def _row_video(r: dict[str, object]) -> VideoMeta:
    return VideoMeta(
        source=str(r["source"]),
        video_id=str(r["video_id"]),
        url=str(r["url"]),
        title=str(r["title"]),
        description="",
        upload_date=str(r["upload_date"]),
        height=int(str(r["height"])),
        duration_s=float(str(r["duration_s"])),
    )


def _assign_ids(kept: list[dict[str, object]], existing: pd.DataFrame | None) -> pd.DataFrame:
    """Keep existing IDs; number new fighters in order of first appearance."""
    known: dict[str, str] = {}
    if existing is not None and not existing.empty:
        known = dict(zip(existing["name_key"], existing["fighter_id"], strict=True))
    next_n = 1 + max((int(i[1:]) for i in known.values()), default=0)

    names: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    order: list[tuple[int, str, str]] = []
    for r in kept:
        for side in ("a", "b"):
            key, name = str(r[f"key_{side}"]), str(r[f"fighter_{side}_name"])
            names.setdefault(key, set()).add(name)
            counts[key] = counts.get(key, 0) + 1
            year = r["year"]
            order.append((int(str(year)) if year is not None else 9999, str(r["fight_id"]), key))

    ids = dict(known)
    for _, _, key in sorted(order):
        if key not in ids:
            ids[key] = f"F{next_n:04d}"
            next_n += 1

    rows = [
        {
            "fighter_id": ids[key],
            "name": sorted(variants)[0],
            "name_key": key,
            "variants": "|".join(sorted(variants)),
            "n_fights": counts[key],
        }
        for key, variants in names.items()
    ]
    return (
        pd.DataFrame(rows, columns=FIGHTER_COLUMNS).sort_values("fighter_id").reset_index(drop=True)
    )
