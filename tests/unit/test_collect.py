"""Footage collection: adult-only filter, title parsing, and the fighter registry.

All names here are made up.
"""

import pandas as pd
import pytest

from cornerman.collect.adult import minor_reasons
from cornerman.collect.registry import VideoMeta, build_tables, fight_id
from cornerman.collect.titles import name_key, parse_title
from cornerman.contracts.fights import FIGHTERS_SCHEMA, FIGHTS_SCHEMA


@pytest.mark.parametrize(
    "text",
    [
        "2022 Youth World Championships",
        "won the Junior World Champion title",
        "Cadet division final",
        "U18 final",
        "U-16 bout",
        "U21 European Open",
        "under 18 category",
        "Under-17s",
        "the 16 year old prospect",
        "aged 17, she won",
        "kids MMA showcase",
        "schoolboys tournament",
    ],
)
def test_minor_markers_are_caught(text: str) -> None:
    assert minor_reasons(text)


@pytest.mark.parametrize(
    "text",
    [
        "2023 World Championships | FULL FIGHT",
        "the 24 year old prospect",
        "aged 19",
        "Bantamweight final, 3 rounds",
        "Senior European Championships",
        "Round 2 stoppage at 1:40",
        "Up 2 rounds to none",
    ],
)
def test_adult_text_passes(text: str) -> None:
    assert minor_reasons(text) == []


def test_parse_title() -> None:
    p = parse_title("Alex Doe vs. Sam Roe | FULL FIGHT | 2023 Test Championships")
    assert p is not None
    assert (p.fighter_a, p.fighter_b, p.meeting, p.year) == ("Alex Doe", "Sam Roe", 1, 2023)
    assert p.event == "2023 Test Championships"


def test_parse_rematch_and_no_dot() -> None:
    p = parse_title("Alex Doe vs Sam Roe 2 | Full Fight | Open Cup")
    assert p is not None
    assert (p.fighter_b, p.meeting, p.year) == ("Sam Roe", 2, None)


@pytest.mark.parametrize(
    "title", ["Best Knockouts of 2023", "Day 3 Highlights | 2024 Championships"]
)
def test_non_fight_titles_are_rejected(title: str) -> None:
    assert parse_title(title) is None


def test_name_key_folds_accents_and_case() -> None:
    assert name_key("Paweł Ąbćdę") == name_key("PAWEL abcde")
    assert name_key("Jöhn  O'Neil-Smith") == "john o neil smith"


def video(vid: str, a: str, b: str, event: str = "2023 Test Cup", **kw: object) -> VideoMeta:
    base: dict[str, object] = {
        "source": "test",
        "video_id": vid,
        "url": f"https://example.org/{vid}",
        "title": f"{a} vs. {b} | FULL FIGHT | {event}",
        "description": "",
        "upload_date": "20240101",
        "height": 1080,
        "duration_s": 600.0,
    }
    base.update(kw)
    return VideoMeta(**base)  # pyright: ignore[reportArgumentType]


def test_registry_filters_and_ids() -> None:
    videos = [
        video("v1", "Alex Doe", "Sam Roe"),
        video("v2", "Alex Doe", "Kim Poe", event="2024 Test Cup"),
        video("v3", "Lee Moe", "Ash Noe", description="Junior World Champion"),
        video("v4", "Lee Moe", "Jo Zoe", event="2025 Test Cup"),  # dropped: Lee in a junior bout
        video("v5", "Low Res", "Other One", height=360),
        video("v6", "ALEX DOE", "Sam Roe", height=720),  # duplicate of v1 at lower resolution
        video("v7", "", "", title="Highlights reel"),
    ]
    t = build_tables(videos, min_height=480)
    FIGHTS_SCHEMA.validate(t.fights)
    FIGHTERS_SCHEMA.validate(t.fighters)

    assert sorted(t.fights["video_id"]) == ["v1", "v2"]
    reasons = dict(zip(t.skipped["video_id"], t.skipped["reason"], strict=True))
    assert reasons["v3"].startswith("might involve a minor")
    assert reasons["v4"].startswith("a fighter also appears")
    assert reasons["v5"].startswith("resolution")
    assert reasons["v6"] == "duplicate"
    assert reasons["v7"] == "title is not a single fight"

    by_name = dict(zip(t.fighters["name_key"], t.fighters["fighter_id"], strict=True))
    assert by_name["alex doe"] == "F0001"
    assert set(by_name) == {"alex doe", "sam roe", "kim poe"}
    assert t.fighters.set_index("name_key").loc["alex doe", "n_fights"] == 2


def test_existing_ids_are_kept_and_new_ones_continue() -> None:
    existing = pd.DataFrame(
        {
            "fighter_id": ["F0007"],
            "name": ["Sam Roe"],
            "name_key": ["sam roe"],
            "variants": ["Sam Roe"],
            "n_fights": [1],
        }
    )
    t = build_tables([video("v1", "Alex Doe", "Sam Roe")], 480, existing)
    ids = dict(zip(t.fighters["name_key"], t.fighters["fighter_id"], strict=True))
    assert ids == {"sam roe": "F0007", "alex doe": "F0008"}


def test_fight_id_is_stable_and_hides_the_video() -> None:
    assert fight_id("test", "abc") == fight_id("test", "abc")
    assert "abc" not in fight_id("test", "abc")
    assert fight_id("test", "abc") != fight_id("other", "abc")


def test_junior_bout_under_an_adult_title_is_caught_by_tags_or_screen() -> None:
    """IMMAF files junior bouts under adult event names; only tags or the score bar say so."""
    videos = [
        video("v1", "Alex Doe", "Sam Roe", screen="JUNIORS / MEN / M: BANTAMWEIGHT 61.2 KG"),
        video("v2", "Kim Poe", "Lee Moe", tags="MMA\n2026 Youth World Championships"),
        video("v3", "Alex Doe", "Ray Coe", event="2024 Test Cup"),  # same fighter as v1
        video("v4", "Ann Hoe", "Bo Loe", screen="SENIORS / WOMEN / F: FLYWEIGHT"),
    ]
    t = build_tables(videos, min_height=480)
    assert t.fights["video_id"].tolist() == ["v4"]
    reasons = dict(zip(t.skipped["video_id"], t.skipped["reason"], strict=True))
    assert "on screen: word: juniors" in reasons["v1"]
    assert "tags: word: youth" in reasons["v2"]
    assert reasons["v3"] == "a fighter also appears in a bout that might involve a minor"


def test_screen_text_joins_frames() -> None:
    from cornerman.collect.screen import screen_text

    res: dict[str, object] = {"frames": 2, "texts": [["A", "B"], ["JUNIORS / MEN"]]}
    assert screen_text(res) == "A B\nJUNIORS / MEN"
    assert screen_text({"frames": 0, "texts": []}) == ""
