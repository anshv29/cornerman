"""ViCoS position mapping (no data needed)."""

import pytest

from cornerman.gold.vicos import states
from cornerman.taxonomy import State, compatible

POSITIONS = [
    "standing", "5050_guard", "takedown1", "takedown2",
    *(f"{p}{i}" for p in ("mount", "side_control", "back", "turtle", "open_guard",
                          "closed_guard", "half_guard") for i in (1, 2)),
]  # fmt: skip


def test_named_athlete_is_on_top_or_in_guard() -> None:
    assert states("mount1") == (State.GROUND_TOP, State.GROUND_BOTTOM)
    assert states("mount2") == (State.GROUND_BOTTOM, State.GROUND_TOP)
    assert states("closed_guard1") == (State.GROUND_BOTTOM, State.GROUND_TOP)
    assert states("takedown1") is None


@pytest.mark.parametrize("position", POSITIONS)
def test_every_mapped_pair_is_compatible(position: str) -> None:
    s = states(position)
    assert s is None or compatible(*s)


def test_unknown_position_is_an_error() -> None:
    with pytest.raises(ValueError):
        states("kneebar1")
