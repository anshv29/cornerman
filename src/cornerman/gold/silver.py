"""The silver consensus set: clips where both VLM prompts and the rules agree.

A clip is silver for a model when:
- both prompts gave a readable answer saying the clip is live fighting;
- both gave the same (red, blue) state pair;
- the rules labeller's coarse label fits that pair (gold.rules.agrees).

Clinches are the exception. The direct prompt called red the controller in all
114 clinches it saw, which is position bias, so its choice of controller is not
used. A clinch needs all three prompts (direct, questions, questions_swapped)
to say clinch, and questions and questions_swapped (blue named first) to name
the same controller; the pair then comes from questions. Without a swapped run
no clinch can be silver. The final pair is in red_final and blue_final.

It is called silver, not gold, everywhere: three automatic labellers agreeing
is evidence, not ground truth, and the set leans towards the easy cases where
they can agree (rules need both fighters boxed separately, for example).

A lower tier, prompts_only, keeps clips where both prompts agree but the rules
abstain (no coarse label, mostly tangled or merged boxes). It is weaker: the
two prompts come from the same model, so their errors are correlated. It is
kept apart from silver and never counted as silver.
"""

import numpy as np
import pandas as pd

from cornerman.gold.rules import agrees
from cornerman.taxonomy import State

CLINCH = {State.CLINCH_CONTROLLING.value, State.CLINCH_CONTROLLED.value}
FAMILIES = {
    "clinch control": State.CLINCH_CONTROLLING,
    "ground top": State.GROUND_TOP,
    "cage pressure": State.STANDING_CAGE_PRESSURING,
}
"""Directional states: who is the controller, the one on top, the one backing up."""


def _is(col: pd.Series, values: set[str]) -> pd.Series:
    return col.isin(list(values))


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
        for p in ("direct", "questions", "questions_swapped"):
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
    clinch = _is(w["red_direct"], CLINCH) | _is(w["red_questions"], CLINCH)
    same = (w["red_direct"] == w["red_questions"]) & (w["blue_direct"] == w["blue_questions"])
    swap_same = (w["red_questions"] == w["red_questions_swapped"]) & (
        w["blue_questions"] == w["blue_questions_swapped"]
    )
    clinch_ok = (
        _is(w["red_direct"], CLINCH)
        & _is(w["red_questions"], CLINCH)
        & (w["live_questions_swapped"] == "yes")
        & swap_same
    )
    w["prompts_agree"] = w["live"] & ((~clinch & same) | (clinch & clinch_ok))
    w["red_final"] = w["red_questions"].where(w["prompts_agree"])
    w["blue_final"] = w["blue_questions"].where(w["prompts_agree"])
    w["rules_agree"] = [
        bool(ok) and agrees(State(r), State(b), rl if isinstance(rl, str) else None)
        for ok, r, b, rl in zip(w["prompts_agree"], w["red_final"], w["blue_final"], w["rules"],
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
    - pair: fighter-clips whose final state (after the clinch swap rule) is this one;
    - silver: of `pair`, the clips the rules accept too; rules_pass = silver / pair;
    - prompts_only: of `pair`, the clips where the rules abstain (the lower tier).
    For clinch states, direct and both don't count toward pair: the direct prompt's
    controller isn't used.

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
                    "final": live[f"{side}_final"],
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
            final = (g["final"] == state.value) & g["prompts_agree"]
            pair = int(final.sum())
            kept = int((final & g["silver"]).sum())
            lower = int((final & g["prompts_only"]).sum())
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


def _dominant(w: pd.DataFrame, prompt: str, dominant: State) -> pd.Series:
    """'red' or 'blue' for the fighter in the dominant state, else None."""
    red, blue = w[f"red_{prompt}"] == dominant.value, w[f"blue_{prompt}"] == dominant.value
    return pd.Series(np.where(red, "red", "blue"), index=w.index).where(red | blue)


def swap_report(w: pd.DataFrame) -> pd.DataFrame:
    """Position bias, per model and directional family (FAMILIES), over clips where
    both the questions and the swapped prompt name a dominant fighter in that family:
    clips; same, the share naming the same fighter both times; red_first and
    red_swapped, the share naming red in each run. Without position bias red_first
    and red_swapped are about equal and same is high."""
    rows: list[dict[str, object]] = []
    for model, g in w.groupby("model"):
        for family, dominant in FAMILIES.items():
            a, b = _dominant(g, "questions", dominant), _dominant(g, "questions_swapped", dominant)
            both = a.notna() & b.notna()
            n = int(both.sum())
            nan = float("nan")
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "clips": n,
                    "same": float((a[both] == b[both]).mean()) if n else nan,
                    "red_first": float((a[both] == "red").mean()) if n else nan,
                    "red_swapped": float((b[both] == "red").mean()) if n else nan,
                }
            )
    return pd.DataFrame(rows).set_index(["model", "family"]).round(2)
