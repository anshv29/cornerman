"""The silver consensus set: clips where both VLM prompts and the rules agree.

A clip is silver for a model when:
- both prompts gave a readable answer saying the clip is live fighting;
- both gave the same (red, blue) state pair;
- the rules labeller's coarse label fits that pair (gold.rules.agrees).

It is called silver, not gold, everywhere: three automatic labellers agreeing
is evidence, not ground truth, and the set leans towards the easy cases where
they can agree (rules need both fighters boxed separately, for example).

A lower tier, prompts_only, keeps clips where both prompts agree but the rules
abstain (no coarse label, mostly tangled or merged boxes). It is weaker: the
two prompts come from the same model, so their errors are correlated. It is
kept apart from silver and never counted as silver.
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
    silver (all of it), and prompts_only (prompts agree, rules abstain)."""
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
    w["prompts_only"] = w["prompts_agree"] & w["rules"].isna()
    return w


def by_state(w: pd.DataFrame) -> pd.DataFrame:
    """Per model and state, counted in fighter-clips (each clip gives a red and a blue one),
    live clips only:
    - direct, questions: how often each prompt names the state;
    - both: how often both prompts name it for the same fighter;
    - agree: both / (direct + questions - both), so 1 means the prompts never split on it;
    - pair: of `both`, the clips where the prompts also agree on the other fighter;
    - silver: of `pair`, the clips the rules accept too; rules_pass = silver / pair;
    - prompts_only: of `pair`, the clips where the rules abstain (the lower tier).

    These are agreement numbers between labellers, not accuracy against ground truth.
    """
    live = w[w["live"]]
    sides = pd.concat(
        [
            pd.DataFrame(
                {
                    "model": live["model"],
                    "direct": live[f"{side}_direct"],
                    "questions": live[f"{side}_questions"],
                    "prompts_agree": live["prompts_agree"],
                    "silver": live["silver"],
                    "prompts_only": live["prompts_only"],
                }
            )
            for side in ("red", "blue")
        ],
        ignore_index=True,
    )
    rows: list[dict[str, object]] = []
    for model, g in sides.groupby("model"):
        for state in State:
            d = g["direct"] == state.value
            q = g["questions"] == state.value
            both = d & q
            n_both = int(both.sum())
            n_any = int(d.sum() + q.sum()) - n_both
            pair = int((both & g["prompts_agree"]).sum())
            kept = int((both & g["silver"]).sum())
            lower = int((both & g["prompts_only"]).sum())
            rows.append(
                {
                    "model": model,
                    "state": state.value,
                    "direct": int(d.sum()),
                    "questions": int(q.sum()),
                    "both": n_both,
                    "agree": n_both / n_any if n_any else float("nan"),
                    "pair": pair,
                    "silver": kept,
                    "rules_pass": kept / pair if pair else float("nan"),
                    "prompts_only": lower,
                }
            )
    return pd.DataFrame(rows).set_index(["model", "state"]).round(2)


def report(w: pd.DataFrame) -> pd.DataFrame:
    """Per model and stratum: clips, then the share readable, live, with both prompts
    agreeing, silver, and the lower prompts_only tier."""
    g = w.groupby(["model", "stratum"])
    out = pd.DataFrame(
        {
            "clips": g.size(),
            "readable": g["readable"].mean(),
            "live": g["live"].mean(),
            "prompts_agree": g["prompts_agree"].mean(),
            "silver": g["silver"].mean(),
            "prompts_only": g["prompts_only"].mean(),
        }
    )
    return out.round(2)
