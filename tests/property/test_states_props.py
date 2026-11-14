"""Properties of the state decision step that must hold for any probabilities."""

import itertools

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from cornerman.events.states import DIRECTION_MIN, marginals, smooth, transitions
from cornerman.weak_labels.labellers import PAIR_STATES

SWAP = [
    PAIR_STATES.index(s.replace("red", "X").replace("blue", "red").replace("X", "blue"))
    for s in PAIR_STATES
]
"""Column order that swaps every red pair state with its blue mirror."""


def _probs(n: int, k: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(np.float64, (n, k), elements=st.floats(0.01, 1.0))


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 5).flatmap(lambda n: _probs(n, 3)), st.floats(0.5, 0.95))
def test_marginals_match_brute_force(p: np.ndarray, stay: float) -> None:
    """Forward-backward equals summing over every path, with forbidden moves."""
    trans = transitions(3, stay, [(0, 1)])
    n = len(p)
    total = np.zeros((n, 3))
    for path in itertools.product(range(3), repeat=n):
        w = np.prod([p[i, s] for i, s in enumerate(path)]) / 3
        w *= np.prod([trans[a, b] for a, b in itertools.pairwise(path)])
        for i, s in enumerate(path):
            total[i, s] += w
    assert np.allclose(marginals(p, trans), total / total.sum(1, keepdims=True), atol=1e-9)


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 12).flatmap(lambda n: _probs(n, len(PAIR_STATES))))
def test_smooth_has_no_colour_preference(p: np.ndarray) -> None:
    """Swapping red and blue in the input swaps them in the output wherever the
    direction counts as known. A lopsided transition would show up here as a colour
    lean. On an exact tie Viterbi's argmax takes the first state, red, so a tie may
    break red, but a tie is never sure enough to count."""
    m = pd.DataFrame({"fight_id": ["a"] * len(p), "t": np.arange(len(p)) * 0.5})
    path, sure = smooth(m, p, 0.9, 1.0)
    path_sw, _ = smooth(m, p[:, SWAP], 0.9, 1.0)
    for i, j, s in zip(path, path_sw, sure, strict=True):
        assert SWAP[int(i)] == int(j) or s < DIRECTION_MIN
