"""Property tests for the Dirichlet-multinomial math."""

import itertools

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from cornerman.models.dirichlet_multinomial import (
    FloatArray,
    log_pmf,
    posterior,
    posterior_mean,
)

K = 4
counts_st = arrays(np.float64, K, elements=st.integers(0, 30).map(float))
pi_st = arrays(np.float64, K, elements=st.floats(0.05, 1.0)).map(lambda p: p / p.sum())
alpha_st = st.floats(0.1, 100.0)


@given(counts=counts_st, alpha=alpha_st, pi=pi_st, data=st.data())
def test_update_does_not_depend_on_order(
    counts: FloatArray, alpha: float, pi: FloatArray, data: st.DataObject
) -> None:
    """Adding observations one at a time, in any order, equals one batch update."""
    obs = [k for k in range(K) for _ in range(int(counts[k]))]
    order = data.draw(st.permutations(obs))
    params = alpha * pi
    for k in order:
        params = posterior(np.eye(K)[k], 1.0, params)
    np.testing.assert_allclose(params, posterior(counts, alpha, pi))


@given(counts=counts_st, alpha=alpha_st, pi=pi_st)
def test_posterior_mean_is_shrunk_frequency(
    counts: FloatArray, alpha: float, pi: FloatArray
) -> None:
    """The posterior mean is w * pi + (1 - w) * raw frequency with w = alpha / (N + alpha)."""
    mean = posterior_mean(counts, alpha, pi)
    assert np.isclose(mean.sum(), 1.0)
    assert (mean > 0).all()
    total = counts.sum()
    if total > 0:
        w = alpha / (total + alpha)
        np.testing.assert_allclose(mean, w * pi + (1 - w) * counts / total)


@settings(max_examples=30)
@given(
    total=st.integers(0, 6), alpha=alpha_st, pi=arrays(np.float64, 3, elements=st.floats(0.05, 1.0))
)
def test_pmf_sums_to_one(total: int, alpha: float, pi: FloatArray) -> None:
    """Summing the pmf over every count vector with a given total gives 1."""
    conc = alpha * pi / pi.sum()
    vectors = [v for v in itertools.product(range(total + 1), repeat=3) if sum(v) == total]
    probs = np.exp(log_pmf(np.array(vectors, dtype=np.float64), conc))
    assert np.isclose(probs.sum(), 1.0)
