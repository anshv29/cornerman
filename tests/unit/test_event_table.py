import numpy as np
import pandas as pd

from cornerman.events.table import fight_events
from cornerman.weak_labels.labellers import FAMILY, PAIR_STATES


def _pred(states: list[str]) -> pd.DataFrame:
    n = len(states)
    df = pd.DataFrame(
        {
            "fight_id": ["f"] * n,
            "frame": np.arange(n) * 5,
            "t": np.arange(n) * 0.5,
            "live": [True] * n,
            "pred": states,
            "pred_family": [FAMILY[s] for s in states],
            "direction_sure": [0.9] * n,
            "direction_known": [FAMILY[s] != "clinch" for s in states],
        }
    )
    for p in PAIR_STATES:
        df[f"q_{p}"] = [0.8 if p == s else 0.2 / 7 for s in states]
    return df


def test_events_from_states_actions_and_strikes() -> None:
    pred = _pred(["neutral", "neutral", "blue-top", "blue-top"])
    actions = pd.DataFrame({"t": [1.0, 1.0], "actor": ["blue", None],
                            "action": ["shoot-takedown", "stand-up"]})  # fmt: skip
    strikes = pd.DataFrame({"t": [0.1, 1.2], "actor": ["red", "blue"]})
    ev = fight_events(pred, actions, strikes, {"red": "F0001", "blue": "F0002"},
                      usable=set(pred["frame"]))  # fmt: skip
    assert len(ev) == 8 and set(ev["fighter"]) == {"F0001", "F0002"}
    blue = ev[ev["fighter"] == "F0002"]
    assert blue["state"].tolist()[2:] == ["ground-top", "ground-top"]
    assert blue["action"].fillna("-").tolist() == ["-", "-", "shoot-takedown", "-"]
    assert ev[ev["fighter"] == "F0001"]["action"].fillna("-").tolist() == ["strike", "-", "-", "-"]
    assert np.allclose(ev["confidence"], (0.8 + 2 * 0.2 / 7) * 0.9)  # family q x direction


def test_only_usable_live_moments() -> None:
    pred = _pred(["neutral"] * 4)
    pred.loc[0, "live"] = False
    ev = fight_events(pred, pd.DataFrame(columns=["t", "actor", "action"]),
                      pd.DataFrame(columns=["t", "actor"]), {"red": "F0001", "blue": "F0002"},
                      usable={5, 10})  # fmt: skip
    assert sorted(set(ev["t"])) == [0.5, 1.0]
