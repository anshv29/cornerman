"""Reading the VLM's answers (no model needed)."""

import pytest

from cornerman.gold.vlm import PROMPTS, extract_json, parse_direct, parse_questions, prompt


def _q(**kw: str) -> str:
    base = {"live": "yes", "level": "standing", "top": "neither", "clinch": "no",
            "clinch_control": "even", "fence": "neither", "hurt": "neither"}  # fmt: skip
    base.update(kw)
    return "Sure.\n```json\n" + str(base).replace("'", '"') + "\n```"


def test_extract_json_takes_the_last_object() -> None:
    assert extract_json('think {"a": 1} then {"live": "Yes"}') == {"live": "yes"}
    assert extract_json("no json here") is None


def test_parse_direct() -> None:
    ok = parse_direct('{"live": "yes", "red": "ground-top", "blue": "ground-bottom"}')
    assert ok == {"live": "yes", "red": "ground-top", "blue": "ground-bottom"}
    assert parse_direct('{"live": "yes", "red": "ground-top", "blue": "ground-top"}') is None
    assert parse_direct('{"live": "yes", "red": "flying", "blue": "ground-bottom"}') is None
    assert parse_direct('{"live": "no", "red": "x", "blue": "y"}') == {
        "live": "no", "red": None, "blue": None}  # fmt: skip


@pytest.mark.parametrize(
    ("answers", "red", "blue"),
    [
        ({}, "standing-neutral", "standing-neutral"),
        ({"level": "ground", "top": "blue"}, "ground-bottom", "ground-top"),
        ({"level": "ground", "top": "neither"}, "ground-scrambling", "ground-scrambling"),
        ({"clinch": "yes", "clinch_control": "red"}, "clinch-controlling", "clinch-controlled"),
        ({"fence": "blue"}, "standing-cage-pressured", "standing-cage-pressuring"),
        ({"fence": "blue", "hurt": "red"}, "standing-cage-pressured", "standing-opponent-hurt"),
        ({"hurt": "blue"}, "standing-opponent-hurt", "standing-neutral"),
        (
            {"clinch": "yes", "clinch_control": "blue", "fence": "blue"},
            "clinch-controlled",
            "clinch-controlling",
        ),
    ],
)
def test_parse_questions(answers: dict[str, str], red: str, blue: str) -> None:
    assert parse_questions(_q(**answers)) == {"live": "yes", "red": red, "blue": blue}


def test_parse_questions_refuses_to_guess() -> None:
    assert parse_questions(_q(clinch="yes", clinch_control="even")) is None
    assert parse_questions(_q(level="sitting")) is None
    assert parse_questions(_q(live="no"))["live"] == "no"  # pyright: ignore[reportOptionalSubscript]


def test_prompts_mention_both_boxes() -> None:
    for name in PROMPTS:
        p = prompt(name)
        assert "RED box" in p and "BLUE box" in p and "JSON" in p


def test_swapped_prompt_only_changes_the_order_of_the_corners() -> None:
    from cornerman.gold.vlm import PROMPTS, swap_colours

    q, sw = PROMPTS["questions"], PROMPTS["questions_swapped"]
    assert sw != q and swap_colours(sw) == q
    assert sorted(q.lower().split()) == sorted(sw.lower().split())  # same words
    assert "BLUE box and the other with a RED box" in sw
    assert "blue / red / even" in sw
    # The answer keys don't change, so the same parser reads both.
    assert sw.split("Answer with JSON only")[1] == q.split("Answer with JSON only")[1]
