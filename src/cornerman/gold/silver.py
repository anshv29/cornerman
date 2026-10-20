"""The silver consensus set: clips where both VLM prompts and the rules agree.

A clip is silver for a model when:
- both prompts gave a readable answer saying the clip is live fighting;
- both gave the same (red, blue) state pair;
- the rules labeller's coarse label fits that pair (gold.rules.agrees).

It is called silver, not gold, everywhere: three automatic labellers agreeing
is evidence, not ground truth, and the set leans towards the easy cases where
they can agree (rules need both fighters boxed separately, for example).
"""

import pandas as pd

from cornerman.gold.rules import agrees
from cornerman.taxonomy import State


def by_clip(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, clip) with each prompt's live, red and blue side by side."""
    wide = labels.pivot_table(
        index=["model", "clip_id"],
        columns="prompt",
        values=["live", "red", "blue"],
        aggfunc="first",
        dropna=False,
    )
    wide.columns = [f"{v}_{p}" for v, p in wide.columns]
    for col in ("live", "red", "blue"):
        for p in ("direct", "questions"):
            if f"{col}_{p}" not in wide:
                wide[f"{col}_{p}"] = None
    return wide.reset_index()


def silver(labels: pd.DataFrame, rules: pd.DataFrame, clips: pd.DataFrame) -> pd.DataFrame:
    """Every (model, clip) with stratum, both prompts' labels, the rules label and flags:
    readable (both prompts parsed), live (both said live), prompts_agree, rules_agree,
    silver (all of it)."""
    w = by_clip(labels).merge(rules[["clip_id", "rules"]], on="clip_id", how="left")
    w = w.merge(clips[["clip_id", "fight_id", "stratum"]], on="clip_id", how="left")
    w["readable"] = w["live_direct"].notna() & w["live_questions"].notna()
    w["live"] = (w["live_direct"] == "yes") & (w["live_questions"] == "yes")
    w["prompts_agree"] = (
        w["live"]
        & (w["red_direct"] == w["red_questions"])
        & (w["blue_direct"] == w["blue_questions"])
    )
    w["rules_agree"] = [
        bool(ok) and agrees(State(r), State(b), rl if isinstance(rl, str) else None)
        for ok, r, b, rl in zip(w["prompts_agree"], w["red_direct"], w["blue_direct"], w["rules"],
                                strict=True)
    ]  # fmt: skip
    w["silver"] = w["prompts_agree"] & w["rules_agree"]
    return w


def report(w: pd.DataFrame) -> pd.DataFrame:
    """Per model and stratum: clips, then the share readable, live, with both prompts
    agreeing, and silver."""
    g = w.groupby(["model", "stratum"])
    out = pd.DataFrame(
        {
            "clips": g.size(),
            "readable": g["readable"].mean(),
            "live": g["live"].mean(),
            "prompts_agree": g["prompts_agree"].mean(),
            "silver": g["silver"].mean(),
        }
    )
    return out.round(2)
