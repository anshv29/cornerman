import numpy as np
import pandas as pd

from cornerman.events.states import folds, smooth, transitions, viterbi
from cornerman.weak_labels.labellers import PAIR_STATES


def test_folds_never_share_a_fight() -> None:
    fights = [f"ft_{i:02d}" for i in range(23)]
    parts = folds(fights, 5, seed=0)
    assert sorted(f for p in parts for f in p) == sorted(fights)
    assert len({f for p in parts for f in p}) == len(fights)
    assert folds(fights, 5, seed=0) == parts


def test_transitions_are_sticky_and_forbid_top_swaps() -> None:
    t = transitions(3, 0.9, [(0, 1)])
    assert np.allclose(t.sum(1), 1) and np.allclose(np.diag(t), 0.9)
    assert t[0, 1] == t[1, 0] == 0.0
    assert transitions(1, 0.9).tolist() == [[1.0]]


def test_viterbi_removes_a_one_step_flicker() -> None:
    p = np.array([[0.9, 0.1]] * 3 + [[0.4, 0.6]] + [[0.9, 0.1]] * 3)
    assert viterbi(p, transitions(2, 0.9)).tolist() == [0] * 7
    assert viterbi(p, transitions(2, 0.5)).tolist() == [0, 0, 0, 1, 0, 0, 0]


def test_smooth_picks_the_family_first() -> None:
    """A clinch split evenly between controllers (0.3 + 0.3) beats neutral at 0.4."""
    k = len(PAIR_STATES)
    p = np.zeros((4, k))
    p[:, PAIR_STATES.index("neutral")] = 0.4
    p[:, PAIR_STATES.index("red-clinch")] = 0.3
    p[:, PAIR_STATES.index("blue-clinch")] = 0.3
    m = pd.DataFrame({"fight_id": ["a"] * 4, "t": [0.0, 0.5, 1.0, 1.5]})
    path: list[str] = [PAIR_STATES[int(i)] for i in smooth(m, p, 0.9, 1.0)]
    assert set(path) <= {"red-clinch", "blue-clinch"}


def test_family_balance_equalises_family_mass() -> None:
    from cornerman.events.states import family_balance
    from cornerman.weak_labels.labellers import FAMILIES, FAMILY

    soft = np.zeros((10, len(PAIR_STATES)))
    soft[:8, PAIR_STATES.index("neutral")] = 1
    soft[8, PAIR_STATES.index("red-clinch")] = 1
    soft[9, PAIR_STATES.index("red-top")] = 1
    w = family_balance(soft)
    mass = soft.sum(0) * w
    per = [sum(mass[i] for i, p in enumerate(PAIR_STATES) if FAMILY[p] == f) for f in FAMILIES]
    assert np.allclose(per, per[0])


def test_balanced_accuracy_of_always_majority() -> None:
    from cornerman.eval.states import balanced

    truth = pd.Series(["a"] * 8 + ["b", "c"])
    bal, f1 = balanced(truth, pd.Series(["a"] * 10))
    assert np.isclose(bal, 1 / 3) and f1 < 0.33
