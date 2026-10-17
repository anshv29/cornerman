"""Parameter recovery tests for the toy Dirichlet-multinomial model.

These prove the recovery harness works: the correct posterior passes, and a
deliberately overconfident one fails. Every seed is fixed, so a failure here
is a real change, not bad luck.
"""

import numpy as np
import pytest

from cornerman.eval.recovery import FloatArray, SbcDraw, check_coverage, run_sbc
from cornerman.models.dirichlet_multinomial import (
    credible_intervals,
    fit_empirical_bayes,
    posterior,
)

ALPHA = 6.0
PI = np.array([0.5, 0.25, 0.15, 0.1])
K = len(PI)
N_PER_FIGHTER = 30
LEVEL = 0.9


def simulate(
    rng: np.random.Generator, n_fighters: int, alpha: float = ALPHA, n: int = N_PER_FIGHTER
) -> tuple[FloatArray, FloatArray]:
    theta = rng.dirichlet(alpha * PI, size=n_fighters)
    counts = np.stack([rng.multinomial(n, t) for t in theta]).astype(np.float64)
    return theta, counts


def one_component_per_fighter(arr: FloatArray) -> FloatArray:
    """Fighter f contributes component f mod K, so checks stay independent."""
    idx = np.arange(len(arr))
    return arr[idx, idx % K]


def test_coverage_with_known_hyperparameters() -> None:
    def replicate(rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
        theta, counts = simulate(rng, n_fighters=20)
        lo, hi = credible_intervals(posterior(counts, ALPHA, PI), LEVEL)
        pick = one_component_per_fighter
        return pick(theta), pick(lo), pick(hi)

    result = check_coverage(replicate, LEVEL, n_reps=100, rng=np.random.default_rng(1))
    assert result.ok(), result
    assert abs(result.rate - LEVEL) < 0.02


def test_coverage_with_empirical_bayes_hyperparameters() -> None:
    """Plugging in estimated (alpha, pi) should still give close to nominal coverage."""

    def replicate(rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
        theta, counts = simulate(rng, n_fighters=200)
        fit = fit_empirical_bayes(counts)
        lo, hi = credible_intervals(posterior(counts, fit.alpha, fit.pi), LEVEL)
        pick = one_component_per_fighter
        return pick(theta), pick(lo), pick(hi)

    result = check_coverage(replicate, LEVEL, n_reps=10, rng=np.random.default_rng(2))
    # Plug-in EB ignores the uncertainty in (alpha, pi), so allow a small gap.
    assert abs(result.rate - LEVEL) < 0.03, result


@pytest.mark.parametrize("true_alpha", [2.0, 6.0, 25.0])
def test_empirical_bayes_recovers_hyperparameters(true_alpha: float) -> None:
    rng = np.random.default_rng(3)
    _, counts = simulate(rng, n_fighters=2000, alpha=true_alpha)
    fit = fit_empirical_bayes(counts)
    assert fit.converged
    assert abs(np.log(fit.alpha / true_alpha)) < 0.2, fit.alpha
    np.testing.assert_allclose(fit.pi, PI, atol=0.015)


def test_empirical_bayes_finds_no_spread_when_there_is_none() -> None:
    """If every fighter has the same theta, alpha should run to its upper bound."""
    rng = np.random.default_rng(4)
    counts = rng.multinomial(N_PER_FIGHTER, PI, size=500).astype(np.float64)
    assert fit_empirical_bayes(counts).alpha > 500


def sbc_draw(scale: float) -> SbcDraw:
    """SBC run for one fighter. `scale` > 1 makes the posterior overconfident."""

    def draw(rng: np.random.Generator, n_draws: int) -> tuple[FloatArray, FloatArray]:
        theta, counts = simulate(rng, n_fighters=1)
        params = posterior(counts, ALPHA, PI)[0] * scale
        return theta[0], rng.dirichlet(params, size=n_draws)

    return draw


def test_sbc_passes_for_correct_posterior() -> None:
    result = run_sbc(sbc_draw(1.0), n_sims=1000, rng=np.random.default_rng(5))
    assert result.ok(), result.p_values


def test_sbc_catches_overconfident_posterior() -> None:
    """Doubling the posterior concentration keeps the mean but halves the variance."""
    result = run_sbc(sbc_draw(2.0), n_sims=1000, rng=np.random.default_rng(5))
    assert not result.ok(), result.p_values


def test_coverage_catches_overconfident_posterior() -> None:
    def replicate(rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
        theta, counts = simulate(rng, n_fighters=20)
        lo, hi = credible_intervals(posterior(counts, ALPHA, PI) * 2.0, LEVEL)
        pick = one_component_per_fighter
        return pick(theta), pick(lo), pick(hi)

    result = check_coverage(replicate, LEVEL, n_reps=100, rng=np.random.default_rng(1))
    assert not result.ok()
    assert result.rate < LEVEL
