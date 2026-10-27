import numpy as np
import pandas as pd

from cornerman.taxonomy import State, compatible
from cornerman.weak_labels.labellers import (
    FAMILIES,
    PAIR_STATES,
    direction,
    family_votes,
    fighter_states,
    pair_of,
    soft_labels,
    vlm_label,
)


def test_every_pair_state_gives_compatible_fighter_states() -> None:
    for pair in PAIR_STATES:
        red, blue = fighter_states(pair)
        assert compatible(red, blue)
        assert pair_of(red.value, blue.value) == pair
    assert pair_of(State.STANDING_OPPONENT_HURT.value, State.STANDING_NEUTRAL.value) == "neutral"
    assert pair_of(State.GROUND_TOP.value, State.GROUND_TOP.value) is None


def _row(**kw: str) -> dict[str, object]:
    base = {"live_direct": "yes", "live_questions": "yes", "live_questions_swapped": "yes"}
    return {**base, **kw}


def test_vlm_label_needs_prompts_to_agree() -> None:
    n, c, cd = "standing-neutral", "clinch-controlling", "clinch-controlled"
    assert vlm_label(_row(red_direct=n, blue_direct=n, red_questions="standing-cage-pressuring",
                          blue_questions="standing-cage-pressured")) == "standing"  # fmt: skip
    # Clinch: the controller only if the swapped run names the same one.
    clinch = {"red_direct": c, "blue_direct": cd, "red_questions": cd, "blue_questions": c}
    assert vlm_label(_row(**clinch, red_questions_swapped=cd, blue_questions_swapped=c)) == (
        "clinch-blue"
    )
    assert vlm_label(_row(**clinch, red_questions_swapped=c, blue_questions_swapped=cd)) == "clinch"
    # Families differ: no label.
    assert vlm_label(_row(red_direct=n, blue_direct=n, red_questions=c, blue_questions=cd)) is None
    assert vlm_label({"live_direct": "no"}) is None


def test_direction_votes() -> None:
    d = direction({"G": "red-pressuring"})["standing"]
    assert d["red-pressuring"] > 0.45 and d["blue-pressuring"] < 0.05
    d = direction({"A": "ground-red-top", "B": "ground-red-top"})["ground"]
    assert d["red-top"] > 0.95
    d = direction({})["clinch"]
    assert d["red-clinch"] == d["blue-clinch"] == 0.5


def test_soft_labels_multiply_family_and_direction() -> None:
    m = pd.DataFrame({"A": [None, "standing"], "M": ["standing-contact", None],
                      "G": [None, "neutral"], "B": ["clinch-red", None]})  # fmt: skip
    votes = family_votes(m)
    assert votes.tolist() == [[1, 1], [0, -1]]
    fam = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]])
    s = soft_labels(m, fam)
    assert np.allclose(s.sum(1), 1)
    assert s[0, PAIR_STATES.index("red-clinch")] == np.float64(0.8 * 0.75)
    assert s[1, PAIR_STATES.index("neutral")] > 0.75
    assert len(FAMILIES) == 3
