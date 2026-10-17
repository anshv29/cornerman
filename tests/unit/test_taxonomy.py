import itertools

from cornerman.taxonomy import Action, State, compatible, mirror_states


def test_counts_match_spec() -> None:
    assert len(State) == 9
    assert len(Action) == 8


def test_every_state_has_a_partner() -> None:
    for s in State:
        assert mirror_states(s), s


def test_compatibility_is_symmetric() -> None:
    for a, b in itertools.product(State, State):
        assert compatible(a, b) == compatible(b, a), (a, b)


def test_examples_from_spec() -> None:
    assert compatible(State.GROUND_TOP, State.GROUND_BOTTOM)
    assert not compatible(State.GROUND_TOP, State.GROUND_TOP)
    assert not compatible(State.CLINCH_CONTROLLING, State.STANDING_NEUTRAL)
