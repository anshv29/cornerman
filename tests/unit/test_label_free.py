"""Label-free checks on event tables."""

import pandas as pd
import pytest

from cornerman.eval.label_free import (
    compatibility_violation_rate,
    impossible_transition_rate,
    impossible_transitions,
)


def _events(a: list[str], b: list[str], actions: dict[int, str] | None = None) -> pd.DataFrame:
    """Two fighters sampled every 0.1 s."""
    rows = [
        {"t": round(i * 0.1, 1), "fighter": f, "state": s,
         "action": (actions or {}).get(i) if f == "A" else None}
        for f, states in (("A", a), ("B", b))
        for i, s in enumerate(states)
    ]  # fmt: skip
    return pd.DataFrame(rows)


def test_compatibility_violation_rate() -> None:
    a = ["ground-top"] * 4
    b = ["ground-bottom", "ground-bottom", "ground-top", "ground-top"]  # last two: both on top
    assert compatibility_violation_rate(_events(a, b)) == pytest.approx(0.5)
    assert compatibility_violation_rate(_events(a, ["ground-bottom"] * 4)) == 0.0


def test_top_bottom_flip_needs_a_sweep() -> None:
    a = ["ground-top"] * 10 + ["ground-bottom"] * 10
    b = ["ground-bottom"] * 10 + ["ground-top"] * 10
    bad = impossible_transitions(_events(a, b))
    assert set(bad["kind"]) == {"flip"} and len(bad) == 2
    swept = _events(a, b, actions={9: "sweep-reversal"})
    assert impossible_transitions(swept).empty


def test_flicker_is_caught_but_a_real_change_is_not() -> None:
    a = ["standing-neutral"] * 10 + ["clinch-controlling"] * 2 + ["standing-neutral"] * 10
    b = ["standing-neutral"] * 10 + ["clinch-controlling"] * 10 + ["clinch-controlled"] * 2
    bad = impossible_transitions(_events(a, b))
    assert bad[["fighter", "kind"]].values.tolist() == [["A", "flicker"]]
    assert impossible_transition_rate(_events(a, b)) == pytest.approx(1 / 4)  # 4 changes, 1 flicker
