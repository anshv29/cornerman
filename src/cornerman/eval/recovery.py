"""Parameter recovery and simulation-based calibration (SBC).

Both answer the same question: if we simulate data from parameters we know,
does the fitting code give those parameters back with honest uncertainty?

Coverage: run many simulate-then-fit replicates and count how often the true
value lands inside the level-L interval. If the intervals are honest, the hit
count is Binomial(n, L), so a two-sided binomial test tells us whether the
observed rate is off by more than chance.

SBC: draw the truth from the prior, simulate data, then rank the truth among
draws from the fitted posterior. If the posterior is correct, the rank is
uniform. Overconfident posteriors pile ranks at the ends (U shape),
underconfident ones in the middle, and biased ones on one side. A chi-square
test on binned ranks checks uniformity.

The model-specific parts are passed in as functions, so the same harness works
for every model in the repo.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

Replicate = Callable[[np.random.Generator], tuple[FloatArray, FloatArray, FloatArray]]
"""One simulate-then-fit run: returns (truth, lower, upper) for a set of
quantities. Quantities inside one replicate must be independent of each other,
or the binomial test will overstate its confidence."""

SbcDraw = Callable[[np.random.Generator, int], tuple[FloatArray, FloatArray]]
"""One SBC run: draw truth from the prior, simulate, fit, and return
(truth of shape (Q,), posterior draws of shape (L, Q))."""


@dataclass(frozen=True)
class CoverageResult:
    level: float
    n: int
    hits: int
    p_value: float

    @property
    def rate(self) -> float:
        return self.hits / self.n

    def ok(self, threshold: float = 0.01) -> bool:
        """False if the rate is off the nominal level by more than chance allows."""
        return self.p_value >= threshold


def check_coverage(
    replicate: Replicate, level: float, n_reps: int, rng: np.random.Generator
) -> CoverageResult:
    hits = 0
    n = 0
    for _ in range(n_reps):
        truth, lo, hi = replicate(rng)
        inside = (lo <= truth) & (truth <= hi)
        hits += int(inside.sum())
        n += inside.size
    p = float(stats.binomtest(hits, n, level).pvalue)
    return CoverageResult(level=level, n=n, hits=hits, p_value=p)


@dataclass(frozen=True)
class SbcResult:
    ranks: IntArray
    """Shape (n_sims, Q). Each rank is in 0..n_draws."""
    n_draws: int
    p_values: FloatArray
    """One chi-square uniformity p-value per quantity."""

    def ok(self, threshold: float = 0.01) -> bool:
        """False if any quantity's ranks fail the test, with a Bonferroni correction."""
        return bool((self.p_values >= threshold / len(self.p_values)).all())


def rank_uniformity_pvalues(ranks: IntArray, n_draws: int, n_bins: int) -> FloatArray:
    """Chi-square test that ranks in 0..n_draws are uniform, per column."""
    if (n_draws + 1) % n_bins != 0:
        raise ValueError("n_draws + 1 must be divisible by n_bins so every bin is equally likely")
    width = (n_draws + 1) // n_bins
    binned = ranks // width
    p: list[float] = []
    for q in range(ranks.shape[1]):
        observed = np.bincount(binned[:, q], minlength=n_bins)
        p.append(float(stats.chisquare(observed).pvalue))
    return np.asarray(p, dtype=np.float64)


def run_sbc(
    draw: SbcDraw,
    n_sims: int,
    rng: np.random.Generator,
    n_draws: int = 99,
    n_bins: int = 20,
) -> SbcResult:
    ranks: list[IntArray] = []
    for _ in range(n_sims):
        truth, post = draw(rng, n_draws)
        ranks.append((post < truth[None, :]).sum(axis=0).astype(np.int64))
    r = np.stack(ranks)
    return SbcResult(ranks=r, n_draws=n_draws, p_values=rank_uniformity_pvalues(r, n_draws, n_bins))
