"""The locked v1 taxonomy from SPEC.md: 9 states, a momentum flag, and 8 actions.

States are from each fighter's own perspective. At any moment the two fighters'
states must be compatible (ground-top for one means ground-bottom for the
other). Do not change this file without asking Ansh first.
"""

from enum import StrEnum


class State(StrEnum):
    STANDING_NEUTRAL = "standing-neutral"
    STANDING_CAGE_PRESSURING = "standing-cage-pressuring"
    STANDING_CAGE_PRESSURED = "standing-cage-pressured"
    STANDING_OPPONENT_HURT = "standing-opponent-hurt"
    CLINCH_CONTROLLING = "clinch-controlling"
    CLINCH_CONTROLLED = "clinch-controlled"
    GROUND_TOP = "ground-top"
    GROUND_BOTTOM = "ground-bottom"
    GROUND_SCRAMBLING = "ground-scrambling"


class Action(StrEnum):
    STRIKE = "strike"
    SHOOT_TAKEDOWN = "shoot-takedown"
    CLINCH_ENTRY_EXIT = "clinch-entry-exit"
    PASS_GUARD = "pass-guard"
    SUBMISSION_ATTEMPT = "submission-attempt"
    SWEEP_REVERSAL = "sweep-reversal"
    STAND_UP = "stand-up"
    RESET_CIRCLE_OUT = "reset-circle-out"


# Momentum is a boolean modifier carried alongside the state, not a state.
MOMENTUM_FLAG = "momentum"

# For each state, the set of states the opponent may be in at the same moment.
# The hurt fighter has no "hurt" state of their own, so standing-opponent-hurt
# pairs with whatever standing state the hurt fighter is in (neutral, or pinned
# on the fence).
_COMPATIBLE: dict[State, frozenset[State]] = {
    State.STANDING_NEUTRAL: frozenset({State.STANDING_NEUTRAL, State.STANDING_OPPONENT_HURT}),
    State.STANDING_CAGE_PRESSURING: frozenset({State.STANDING_CAGE_PRESSURED}),
    State.STANDING_CAGE_PRESSURED: frozenset(
        {State.STANDING_CAGE_PRESSURING, State.STANDING_OPPONENT_HURT}
    ),
    State.STANDING_OPPONENT_HURT: frozenset(
        {State.STANDING_NEUTRAL, State.STANDING_CAGE_PRESSURED}
    ),
    State.CLINCH_CONTROLLING: frozenset({State.CLINCH_CONTROLLED}),
    State.CLINCH_CONTROLLED: frozenset({State.CLINCH_CONTROLLING}),
    State.GROUND_TOP: frozenset({State.GROUND_BOTTOM}),
    State.GROUND_BOTTOM: frozenset({State.GROUND_TOP}),
    State.GROUND_SCRAMBLING: frozenset({State.GROUND_SCRAMBLING}),
}


def compatible(a: State, b: State) -> bool:
    """True if fighter A in state `a` and fighter B in state `b` can happen together."""
    return b in _COMPATIBLE[a]


def mirror_states(s: State) -> frozenset[State]:
    """All states the opponent may be in when this fighter is in `s`."""
    return _COMPATIBLE[s]
