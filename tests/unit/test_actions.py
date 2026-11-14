import numpy as np
import pandas as pd

from cornerman.events.actions import fight_actions, held_runs
from cornerman.weak_labels.labellers import FAMILY, PAIR_STATES


def _fight(states: list[str], known: bool = True, **cols: list[float] | list[int]) -> pd.DataFrame:
    """Moments every 0.5 s with these predicted pair states, live, directions known."""
    n = len(states)
    df = pd.DataFrame(
        {
            "fight_id": ["f"] * n,
            "t": np.arange(n) * 0.5,
            "pred": states,
            "pred_family": [FAMILY[s] for s in states],
            "live": [True] * n,
            "direction_known": [known and FAMILY[s] != "clinch" for s in states],
            "direction_sure": [0.9] * n,
            "red_speed": [0.1] * n,
            "blue_speed": [0.1] * n,
            "shot_id": [0] * n,
            "pair_h": [0.25 if FAMILY[s] == "ground" else 0.5 for s in states],
        }
    )
    for p in PAIR_STATES:
        df[f"q_{p}"] = [0.9 if p == s else 0.1 / 7 for s in states]
    for k, v in cols.items():
        df[k] = v
    return df


def test_held_runs_drop_a_blip() -> None:
    fam = np.array(["standing"] * 6 + ["ground"] * 2 + ["standing"] * 6)
    runs = held_runs(fam, np.arange(len(fam)) * 0.5)
    assert [r.family for r in runs] == ["standing", "standing"]


def test_takedown_names_the_fighter_on_top() -> None:
    a = fight_actions(_fight(["neutral"] * 6 + ["blue-top"] * 6))
    assert a[["action", "actor", "rule"]].values.tolist() == [
        ["shoot-takedown", "blue", "standing-to-ground"]
    ]
    a = fight_actions(_fight(["neutral"] * 6 + ["blue-top"] * 6, known=False))
    assert a["actor"].tolist() == [None]


def test_takedown_needs_standing_stable_states_and_a_height_drop() -> None:
    states = ["neutral"] * 6 + ["blue-top"] * 6
    assert len(fight_actions(_fight(states))) == 1
    # No drop in height (a clinch at the fence called ground, say).
    assert fight_actions(_fight(states, pair_h=[0.5] * 12)).empty
    # A cut between the two sides: heights from different shots don't count.
    assert fight_actions(_fight(states, shot_id=[0] * 6 + [1] * 6)).empty
    # The classifier wasn't sure it was standing before.
    df = _fight(states)
    df.loc[:5, "q_neutral"] = 0.3
    df.loc[:5, "q_red-top"] = 0.6
    assert fight_actions(df).empty
    # From the clinch: not a takedown for now.
    a = fight_actions(_fight(["red-clinch"] * 6 + ["blue-top"] * 6))
    assert "shoot-takedown" not in a["action"].tolist()


def test_stand_up_names_the_fighter_who_was_underneath() -> None:
    a = fight_actions(_fight(["red-top"] * 6 + ["neutral"] * 6))
    assert a[["action", "actor"]].values.tolist() == [["stand-up", "blue"]]


def test_sweep_and_reset_inside_a_position() -> None:
    a = fight_actions(_fight(["red-top"] * 5 + ["scramble"] * 2 + ["blue-top"] * 5))
    assert a[["action", "actor"]].values.tolist() == [["sweep-reversal", "blue"]]
    a = fight_actions(_fight(["red-pressuring"] * 5 + ["neutral"] * 5))
    assert a[["action", "actor"]].values.tolist() == [["reset-circle-out", "blue"]]


def test_clinch_entry_goes_to_the_faster_fighter() -> None:
    states = ["neutral"] * 6 + ["red-clinch"] * 6
    a = fight_actions(_fight(states, red_speed=[0.6] * 12))
    assert a[["action", "actor", "rule"]].values.tolist() == [
        ["clinch-entry-exit", "red", "clinch-entry"]
    ]
    assert fight_actions(_fight(states))["actor"].tolist() == [None]


def test_no_action_across_a_long_gap_or_outside_live_shots() -> None:
    df = _fight(["neutral"] * 6 + ["blue-top"] * 6)
    df.loc[6:, "t"] += 5.0
    assert fight_actions(df).empty
    df = _fight(["neutral"] * 6 + ["blue-top"] * 6)
    df["live"] = False
    assert fight_actions(df)["actor"].tolist() == [None]
