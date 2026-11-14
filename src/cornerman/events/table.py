"""The per-fight event tables (Layer 2 output): `data/events/{fight_id}.parquet`.

One row per fighter per moment (0.5 s of fighting time), checked by the event
contract (contracts/events.py). Built from the state classifier's out-of-fold
predictions (events/states.py), the actions read off state changes
(events/actions.py) and the strike rule (perception/strikes.py).

What goes in, and what is left out:
- Only moments whose identity can be used per fighter: live shots of unflagged
  fights and the usable shots of flagged ones (contracts.perception.usable_frames).
  Elsewhere red and blue aren't known, so there is nothing to say per fighter.
- state: each fighter's state from the predicted pair state. Where the direction
  isn't known (`direction_known` false: always in a clinch, and wherever the
  forward-backward marginal is under 0.8) the state is the best guess and must be
  used only at family level (standing, clinch, ground).
- confidence: the corrected probability of the predicted family (q_*, which matches
  the label rates) times how sure the direction is.
- action: a strike thrown in the moment's 0.5 s, else nothing. Positional actions
  (takedown, stand-up, sweep, reset, clinch entry or exit) stay in the actions table
  and out of anything per fighter until the VLM action labels back them up (Ansh,
  2026-11-14: the first takedown check found 1 in 4 right). PER_FIGHTER_ACTIONS lists
  the ones let through; a positional action with an actor overrides a strike.
- landed: empty; landed or missed is Tier 2. momentum: false; not modelled yet.
- fighter: registry IDs (F0001 ...), never names. Red is the fighter listed first in
  the title (identity.corner_fighters).
"""

import numpy as np
import pandas as pd

from cornerman.contracts.events import validate_events
from cornerman.events.states import row_sum
from cornerman.weak_labels.labellers import FAMILY, PAIR_STATES, fighter_states

PER_FIGHTER_ACTIONS: frozenset[str] = frozenset()
"""Positional actions backed well enough to count per fighter. None yet."""
MOMENT_S = 0.5
"""The moments' grid: an action belongs to the moment at or just before it."""


def _family_q(pred: pd.DataFrame) -> np.ndarray:
    """Corrected probability of each row's predicted family."""
    out = np.zeros(len(pred))
    for f in ("standing", "clinch", "ground"):
        cols = [f"q_{p}" for p in PAIR_STATES if FAMILY[p] == f]
        rows = (pred["pred_family"] == f).to_numpy()
        out[rows] = row_sum(pred.loc[rows, cols].to_numpy(float))
    return out


def _moment_of(times: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Index of the moment each time in `at` falls in (the last moment at or before it,
    if within MOMENT_S), or -1."""
    i = np.searchsorted(times, at, side="right") - 1
    ok = (i >= 0) & (at - times[np.clip(i, 0, None)] < MOMENT_S)
    return np.where(ok, i, -1)


def fight_events(
    pred: pd.DataFrame,
    actions: pd.DataFrame,
    strikes: pd.DataFrame,
    corners: dict[str, str],
    usable: set[int],
) -> pd.DataFrame:
    """The event table of one fight. `pred` are its predicted moments, `actions` and
    `strikes` its rows of those tables, `corners` maps red and blue to fighter IDs,
    `usable` the frames whose identity can be used per fighter."""
    m = pred[pred["live"].astype(bool) & pred["frame"].isin(usable)]
    m = m.sort_values("t").reset_index(drop=True)
    if m.empty:
        return pd.DataFrame()
    times = m["t"].to_numpy(float)
    act: dict[str, list[str | None]] = {c: [None] * len(m) for c in ("red", "blue")}
    s = strikes[strikes["actor"].isin(["red", "blue"])]
    for i, c in zip(_moment_of(times, s["t"].to_numpy(float)), s["actor"], strict=True):
        if i >= 0:
            act[str(c)][int(i)] = "strike"
    a = actions[
        actions["actor"].isin(["red", "blue"]) & actions["action"].isin(PER_FIGHTER_ACTIONS)
    ]
    for i, c, name in zip(_moment_of(times, a["t"].to_numpy(float)), a["actor"], a["action"],
                          strict=True):  # fmt: skip
        if i >= 0:
            act[str(c)][int(i)] = str(name)
    conf = np.clip(_family_q(m) * m["direction_sure"].to_numpy(float), 0.0, 1.0)
    states = [fighter_states(str(p)) for p in m["pred"]]
    rows: list[dict[str, object]] = []
    for k in range(len(m)):
        for j, c in enumerate(("red", "blue")):
            rows.append(
                {
                    "fight_id": str(m["fight_id"].iloc[0]),
                    "t": float(times[k]),
                    "fighter": corners[c],
                    "state": states[k][j].value,
                    "momentum": False,
                    "action": act[c][k],
                    "landed": pd.NA,
                    "confidence": float(conf[k]),
                    "source": "aggregate",
                    "direction_known": bool(m["direction_known"].iloc[k]),
                }
            )
    df = pd.DataFrame(rows)
    df["landed"] = df["landed"].astype("boolean")
    return validate_events(df)
