"""Silver consensus: both prompts and the rules must agree."""

import pandas as pd

from cornerman.gold.silver import by_state, report, silver

Answer = tuple[str, str | None, str | None]


def _rows(clip: str, direct: Answer, questions: Answer) -> list[dict[str, object]]:
    return [
        {"model": "m", "clip_id": clip, "prompt": p, "live": a[0], "red": a[1], "blue": a[2]}
        for p, a in (("direct", direct), ("questions", questions))
    ]


def test_silver_needs_all_three() -> None:
    top = ("yes", "ground-top", "ground-bottom")
    labels = pd.DataFrame(
        _rows("a", top, top)  # all agree
        + _rows("b", top, ("yes", "ground-bottom", "ground-top"))  # prompts disagree
        + _rows("c", top, top)  # rules disagree
        + _rows("d", ("no", None, None), ("no", None, None))  # not live
    )
    rules = pd.DataFrame(
        {"clip_id": list("abcd"), "rules": ["ground-red-top", "ground-red-top", "standing", None]}
    )
    clips = pd.DataFrame({"clip_id": list("abcd"), "fight_id": "f", "stratum": "ground"})
    w = silver(labels, rules, clips).set_index("clip_id")
    assert w["silver"].to_dict() == {"a": True, "b": False, "c": False, "d": False}
    assert w.loc["b", "live"] and not w.loc["b", "prompts_agree"]
    assert not w.loc["d", "live"]
    r = report(w.reset_index())
    assert r.loc[("m", "ground"), "silver"] == 0.25
    s = by_state(w.reset_index()).loc["m"]
    # a, b and c are live; a and c agree on both fighters, b names the states the other way.
    want = {"direct": 3, "questions": 3, "both": 2, "pair": 2, "silver": 1}
    assert {k: s.at["ground-top", k] for k in want} == want
    assert s.loc["ground-top", "agree"] == 0.5  # 2 shared out of 4 distinct fighter-clips
    assert s.loc["ground-top", "rules_pass"] == 0.5  # c fails the rules
    assert s.loc["standing-neutral", "direct"] == 0
