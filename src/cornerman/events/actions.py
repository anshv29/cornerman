"""Actions from changes of state (Step 6, rules first).

Most of the taxonomy's actions are changes of position, which the state
classifier (events/states.py) already tracks: a clinch entry is standing turning
into clinch, a takedown is standing or clinch turning into ground, a stand-up is
ground turning into standing or clinch, a sweep is the top fighter changing, a
reset is a pressured fighter getting back to neutral. So this labeller reads
actions off the smoothed states instead of the video.

A takedown needs more than the change of state (Ansh, 2026-11-14, after the first
blind check found 4 of 17 rule takedowns real; most were the classifier starting a
ground run late or calling clinch at the fence ground):
- standing held for LOOK_S right before, ground held for LOOK_S right after, with
  nothing in between (at most MAX_GAP_S);
- stable: the classifier's family probability before smoothing (the corrected `q_*`,
  which match the real family rates) averages at least STABLE over each of the two
  windows. First written with the balanced `p_*`, which put standing at 0.38 on
  average before a change, since balanced training spreads every moment toward a
  third per family: the wrong scale for "how sure", so changed before any check;
- a real drop in height: the fighters' height in the image (`pair_h`, top of the
  highest box to the bottom of the lowest) after is at most DROP times what it was
  before, both measured inside the same shot (a cut or a zoom across shots would fake
  one). Hip keypoints would be the direct measure, but skeletons aren't trusted while
  the fighters touch (Ansh, 2026-10-27), and a takedown is all touching.
So a clinch going to the ground isn't a takedown here for now, and takedowns in shots
where the boxes aren't both seen are missed. Precision first: these numbers end up
in per-fighter counts once backed.

Two guards on every change, both set by reasoning before looking at any result:
- A position counts only if held for HOLD_S (2 s). Clinch and ground flip back
  and forth between the classifier's runs (489 changes between them over 406
  minutes, against 123 between standing and clinch); a 1 s blip is more likely a
  wrong call than a takedown and a stand-up.
- The two held positions must be at most BRIDGE_S apart (3 s): a longer gap is a
  cut, a break or missing frames, and what happened in between is unknown.

The actor is named only when the states say who it was:
- takedown: the fighter on top in the first LOOK_S of the ground position;
- stand-up: the fighter on the bottom in the last LOOK_S before it;
- sweep: the fighter on top after the change;
- reset: the fighter who was pressured;
- clinch entry or exit: the fighter whose box moved faster over LOOK_S before
  (entry) or after (exit) the change, if at least FASTER times faster.
Only directions the classifier counts as known are used (`direction_known`), and
only in live shots (no red or blue outside them). Otherwise the action is kept
with no actor: it still counts for the fight, not for a fighter.

Not produced here: strikes (perception/strikes.py, skeleton speed), pass guard and
submission attempts (nothing in our features shows guard or a submission; they
come from the VLM).
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cornerman.events.states import row_sum
from cornerman.taxonomy import Action
from cornerman.weak_labels.labellers import FAMILY, PAIR_STATES

HOLD_S = 2.0
BRIDGE_S = 3.0
LOOK_S = 2.0
FASTER = 1.5
MAX_GAP_S = 1.0
"""Moments further apart than this (0.5 s grid) break a run, as in events/states.py."""
STABLE = 0.5
"""A takedown's standing and ground windows need this mean family probability: most of
the classifier's mass, not just the most likely family."""
DROP = 0.7
"""A takedown's height after over before. Set by reasoning: standing fighters' boxes span
a body height; two fighters on the ground make a pile about half that, a top fighter
posturing up perhaps 0.7 of it. A clinch on the feet stays near 1."""
MIN_HEIGHTS = 2
"""Height readings needed on each side of a takedown (about 1 s at 0.5 s)."""

COLUMNS = ["fight_id", "t", "actor", "action", "rule", "confidence"]
Add = Callable[[int, str | None, Action, str, float], None]
"""Records one action: row, actor, action, rule name, confidence."""


@dataclass(frozen=True)
class Held:
    """One held position: a run of one family lasting at least HOLD_S."""

    family: str
    start: int
    end: int
    """Row indices into the fight's moments, end exclusive."""


def held_runs(fam: np.ndarray, t: np.ndarray, hold_s: float = HOLD_S) -> list[Held]:
    """Runs of one family (a gap over MAX_GAP_S starts a new run) lasting `hold_s`."""
    out: list[Held] = []
    start = 0
    for i in range(1, len(fam) + 1):
        if i == len(fam) or fam[i] != fam[i - 1] or t[i] - t[i - 1] > MAX_GAP_S:
            if t[i - 1] - t[start] + 0.5 >= hold_s:
                out.append(Held(str(fam[start]), start, i))
            start = i
    return out


def _known(g: pd.DataFrame, rows: range, states: tuple[str, ...]) -> pd.Series:
    """The known directional calls among `rows`, limited to `states`."""
    sub = g.iloc[list(rows)]
    sub = sub[sub["direction_known"].astype(bool) & sub["pred"].isin(states)]
    return sub["pred"]


def _majority(calls: pd.Series) -> tuple[str | None, float]:
    """The colour named by most calls ("red-top" -> "red"), and its share. A tie names
    nobody (value_counts would break it by order, which would favour a colour)."""
    if calls.empty:
        return None, float("nan")
    colour = calls.str.split("-").str[0]
    r, b = int((colour == "red").sum()), int((colour == "blue").sum())
    if r == b:
        return None, 0.5
    return ("red", r / (r + b)) if r > b else ("blue", b / (r + b))


def _faster(g: pd.DataFrame, rows: range) -> str | None:
    sub = g.iloc[list(rows)]
    r, b = float(sub["red_speed"].mean()), float(sub["blue_speed"].mean())
    if np.isnan(r) or np.isnan(b):
        return None
    if r >= FASTER * b and r > b:  # r > b: two still fighters (0 and 0) name nobody
        return "red"
    if b >= FASTER * r and b > r:
        return "blue"
    return None


def _family_p(g: pd.DataFrame, rows: range, family: str) -> float:
    """Mean corrected probability of `family` over `rows`, before smoothing (`q_*`)."""
    cols = [f"q_{p}" for p in PAIR_STATES if FAMILY[p] == family]
    sub = g.iloc[list(rows)]
    return float(row_sum(sub[cols].to_numpy(float)).mean()) if len(sub) else 0.0


def _height_drop(g: pd.DataFrame, before: range, after: range, at: int) -> float:
    """The fighters' height after over before, inside the shot of row `at` (nan if either
    side has under MIN_HEIGHTS readings)."""
    shot = g["shot_id"].iloc[at]

    def heights(rows: range) -> pd.Series:
        sub = g.iloc[list(rows)]
        return sub.loc[sub["shot_id"] == shot, "pair_h"].dropna()

    h0, h1 = heights(before), heights(after)
    if len(h0) < MIN_HEIGHTS or len(h1) < MIN_HEIGHTS or h0.median() <= 0:
        return float("nan")
    return float(h1.median() / h0.median())


def _takedown(g: pd.DataFrame, prev: Held, nxt: Held, before: range, after: range) -> bool:
    """Standing held, then ground held, stable on both sides, with a real height drop."""
    t = g["t"].to_numpy(float)
    if prev.family != "standing" or t[nxt.start] - t[prev.end - 1] > MAX_GAP_S:
        return False
    if _family_p(g, before, "standing") < STABLE or _family_p(g, after, "ground") < STABLE:
        return False
    return _height_drop(g, before, after, nxt.start) <= DROP  # nan fails


def _window(g: pd.DataFrame, at: int, seconds: float, forward: bool) -> range:
    """Rows within `seconds` after (forward) or before row `at`, inside the fight."""
    t = g["t"].to_numpy(float)
    if forward:
        j = at
        while j < len(g) and t[j] - t[at] < seconds:
            j += 1
        return range(at, j)
    i = at
    while i > 0 and t[at] - t[i - 1] < seconds:
        i -= 1
    return range(i, at)


def fight_actions(g: pd.DataFrame) -> pd.DataFrame:
    """Actions of one fight from its predicted states (sorted by t)."""
    g = g.reset_index(drop=True)
    fam = g["pred_family"].to_numpy()
    t = g["t"].to_numpy(float)
    live = g["live"].astype(bool).to_numpy()
    rows: list[dict[str, object]] = []

    def add(i: int, actor: str | None, action: Action, rule: str, conf: float) -> None:
        rows.append({"fight_id": g["fight_id"].iloc[0], "t": float(t[i]),
                     "actor": actor if live[i] else None, "action": action.value,
                     "rule": rule, "confidence": conf})  # fmt: skip

    held = held_runs(fam, t)
    for prev, nxt in itertools.pairwise(held):
        if prev.family == nxt.family or t[nxt.start] - t[prev.end - 1] > BRIDGE_S:
            continue
        i = nxt.start
        before = _window(g, prev.end, LOOK_S, forward=False)
        after = _window(g, nxt.start, LOOK_S, forward=True)
        if nxt.family == "ground":
            if _takedown(g, prev, nxt, before, after):
                actor, share = _majority(_known(g, after, ("red-top", "blue-top")))
                add(i, actor, Action.SHOOT_TAKEDOWN, "standing-to-ground", share)
        elif prev.family == "ground":
            top, share = _majority(_known(g, before, ("red-top", "blue-top")))
            actor = {"red": "blue", "blue": "red"}.get(top or "")
            add(i, actor, Action.STAND_UP, f"ground-to-{nxt.family}", share)
        elif nxt.family == "clinch":
            add(i, _faster(g, before), Action.CLINCH_ENTRY_EXIT, "clinch-entry", float("nan"))
        else:
            add(i, _faster(g, after), Action.CLINCH_ENTRY_EXIT, "clinch-exit", float("nan"))

    # Inside a position: the top fighter changes (sweep), a pressured fighter gets out.
    known = g["direction_known"].astype(bool).to_numpy() & live
    pred = g["pred"].to_numpy()
    for h in held:
        if h.family == "ground":
            _changes(g, h, known, pred, ("red-top", "blue-top"), add)
        elif h.family == "standing":
            _resets(g, h, known, pred, add)
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values("t", kind="stable").reset_index(drop=True)


def _segments(
    idx: range, known: np.ndarray, pred: np.ndarray, t: np.ndarray, states: tuple[str, ...]
) -> list[tuple[str, int, int]]:
    """Known runs of one of `states` inside `idx`, held HOLD_S: (state, start, end)."""
    keep = [i for i in idx if known[i] and pred[i] in states]
    out: list[tuple[str, int, int]] = []
    start = 0
    for k in range(1, len(keep) + 1):
        if k == len(keep) or pred[keep[k]] != pred[keep[k - 1]]:
            a, b = keep[start], keep[k - 1]
            if t[b] - t[a] + 0.5 >= HOLD_S:
                out.append((str(pred[a]), a, b + 1))
            start = k
    return out


def _changes(g: pd.DataFrame, h: Held, known: np.ndarray, pred: np.ndarray,
             states: tuple[str, ...], add: Add) -> None:  # fmt: skip
    t = g["t"].to_numpy(float)
    segs = _segments(range(h.start, h.end), known, pred, t, states)
    for (s0, _, e0), (s1, b1, _) in itertools.pairwise(segs):
        if s0 != s1 and t[b1] - t[e0 - 1] <= BRIDGE_S:
            sure = float(g["direction_sure"].iloc[b1])
            add(b1, s1.split("-")[0], Action.SWEEP_REVERSAL, "top-changes", sure)


def _resets(g: pd.DataFrame, h: Held, known: np.ndarray, pred: np.ndarray, add: Add) -> None:
    t = g["t"].to_numpy(float)
    segs = _segments(range(h.start, h.end), known, pred, t,
                     ("neutral", "red-pressuring", "blue-pressuring"))  # fmt: skip
    for (s0, _, e0), (s1, b1, _) in itertools.pairwise(segs):
        if s0 != "neutral" and s1 == "neutral" and t[b1] - t[e0 - 1] <= BRIDGE_S:
            pressured = "blue" if s0 == "red-pressuring" else "red"
            add(b1, pressured, Action.RESET_CIRCLE_OUT, "pressure-to-neutral",
                float(g["direction_sure"].iloc[b1]))  # fmt: skip


def all_actions(pred: pd.DataFrame) -> pd.DataFrame:
    """Actions of every fight in a predictions table (events/states.py output)."""
    parts = [fight_actions(g.sort_values("t")) for _, g in pred.groupby("fight_id")]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
