"""Label-free checks on event tables (Phase 2): numbers that need no ground truth.

A state labeller can be wrong in ways that show up without any labels at all:
- the two fighters' states contradict each other (both on top);
- a fighter's state changes in a way that can't happen: top and bottom swap with
  no sweep and no scramble in between, or the state flickers to something else
  and straight back faster than anyone can move.
These rates go next to every accuracy number in Phase 3, and they can be
computed on all fights, not only on the gold set.

Replay-pair consistency (a replay and its live footage should get the same
labels) needs replay detection first and comes later.
"""

import pandas as pd

from cornerman.contracts.events import compatibility_violations
from cornerman.taxonomy import Action, State

FLICKER_S = 0.3
"""A state that lasts less than this between two runs of the same other state is flicker."""


def compatibility_violation_rate(events: pd.DataFrame) -> float:
    """Share of moments with both fighters labelled where the pair of states is impossible."""
    fighters = sorted(events["fighter"].unique())
    if len(fighters) < 2:
        return float("nan")
    a, b = fighters[:2]
    both = events.loc[events["fighter"] == a, ["t"]].merge(
        events.loc[events["fighter"] == b, ["t"]], on="t"
    )
    if both.empty:
        return float("nan")
    return len(compatibility_violations(events)) / len(both)


def _runs(states: pd.DataFrame) -> pd.DataFrame:
    """Runs of one state for one fighter: state, start, end (time of the last sample)."""
    s = states.sort_values("t")
    new = s["state"].ne(s["state"].shift())
    run_id = new.cumsum()
    return s.groupby(run_id).agg(state=("state", "first"), start=("t", "min"), end=("t", "max"))


def impossible_transitions(events: pd.DataFrame, window_s: float = 1.0) -> pd.DataFrame:
    """State changes that can't happen, one row each: fighter, t, before, after, kind.

    - flip: ground-top straight to ground-bottom or back, with no sweep or reversal
      action by either fighter within `window_s` of the change;
    - flicker: A, then B for less than FLICKER_S, then A again.
    """
    rows: list[dict[str, object]] = []
    sweeps = events.loc[events["action"] == Action.SWEEP_REVERSAL.value, "t"].to_numpy()
    flip = {State.GROUND_TOP.value, State.GROUND_BOTTOM.value}
    for fighter, g in events.groupby("fighter"):
        runs = _runs(g[["t", "state"]]).reset_index(drop=True)
        for i in range(1, len(runs)):
            before, after = str(runs.at[i - 1, "state"]), str(runs.at[i, "state"])
            t = float(runs.at[i, "start"])  # pyright: ignore[reportArgumentType]
            if {before, after} == flip and not (abs(sweeps - t) <= window_s).any():
                rows.append({"fighter": fighter, "t": t, "before": before, "after": after,
                             "kind": "flip"})  # fmt: skip
            if i + 1 < len(runs) and str(runs.at[i + 1, "state"]) == before:
                length = float(runs.at[i + 1, "start"]) - t  # pyright: ignore[reportArgumentType]
                if length < FLICKER_S:
                    rows.append({"fighter": fighter, "t": t, "before": before, "after": after,
                                 "kind": "flicker"})  # fmt: skip
    return pd.DataFrame(rows, columns=["fighter", "t", "before", "after", "kind"])


def impossible_transition_rate(events: pd.DataFrame) -> float:
    """Impossible state changes as a share of all state changes."""
    changes = sum(len(_runs(g[["t", "state"]])) - 1 for _, g in events.groupby("fighter"))
    if changes == 0:
        return 0.0
    return len(impossible_transitions(events)) / changes
