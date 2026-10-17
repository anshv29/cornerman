"""Hierarchical Dirichlet-multinomial model for one state (the toy version of 4a).

For each fighter f in a given state, the action counts n_f (a vector over K
actions) are Multinomial(N_f, theta_f), and theta_f ~ Dirichlet(alpha * pi).
pi is the population policy (a point on the simplex) and alpha is how strongly
fighters are pulled toward it.

Because the Dirichlet is conjugate to the multinomial, each fighter's posterior
is Dirichlet(alpha * pi + n_f) in closed form. Empirical Bayes picks alpha and
pi by maximizing the marginal likelihood of all fighters' counts, with theta
integrated out.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special, stats

FloatArray = NDArray[np.float64]


def posterior(counts: FloatArray, alpha: float, pi: FloatArray) -> FloatArray:
    """Dirichlet posterior parameters, one row per fighter."""
    return np.asarray(counts, dtype=np.float64) + alpha * np.asarray(pi, dtype=np.float64)


def posterior_mean(counts: FloatArray, alpha: float, pi: FloatArray) -> FloatArray:
    """(n + alpha * pi) / (N + alpha): a weighted average of pi and the raw frequencies."""
    a = posterior(counts, alpha, pi)
    return a / a.sum(axis=-1, keepdims=True)


def credible_intervals(params: FloatArray, level: float = 0.9) -> tuple[FloatArray, FloatArray]:
    """Equal-tailed intervals for each component of Dirichlet(params).

    Each component of a Dirichlet is Beta(a_k, a_0 - a_k), so the intervals
    come straight from Beta quantiles.
    """
    a = np.asarray(params, dtype=np.float64)
    rest = a.sum(axis=-1, keepdims=True) - a
    tail = (1.0 - level) / 2.0
    lo = np.asarray(stats.beta.ppf(tail, a, rest), dtype=np.float64)
    hi = np.asarray(stats.beta.ppf(1.0 - tail, a, rest), dtype=np.float64)
    return lo, hi


def log_pmf(counts: FloatArray, conc: FloatArray) -> FloatArray:
    """Dirichlet-multinomial log probability of each row of `counts`.

    conc is the Dirichlet parameter vector (alpha * pi). Includes the
    multinomial coefficient, so it is a proper distribution over count vectors.
    """
    n = np.asarray(counts, dtype=np.float64)
    c = np.asarray(conc, dtype=np.float64)
    total = n.sum(axis=-1)
    c0 = c.sum(axis=-1)
    coef = special.gammaln(total + 1) - special.gammaln(n + 1).sum(axis=-1)
    ratio = special.gammaln(c0) - special.gammaln(total + c0)
    per_action = (special.gammaln(n + c) - special.gammaln(c)).sum(axis=-1)
    return np.asarray(coef + ratio + per_action, dtype=np.float64)


def log_marginal_likelihood(counts: FloatArray, alpha: float, pi: FloatArray) -> float:
    return float(log_pmf(counts, alpha * np.asarray(pi, dtype=np.float64)).sum())


@dataclass(frozen=True)
class EBFit:
    alpha: float
    pi: FloatArray
    log_lik: float
    converged: bool


# log(alpha) is searched in this range. A very large alpha means "no
# fighter-to-fighter variation", and the likelihood goes flat there.
LOG_ALPHA_BOUNDS = (-5.0, 12.0)


def _unpack(x: FloatArray) -> tuple[float, FloatArray]:
    logits = np.concatenate([x[1:], [0.0]])
    pi = np.exp(logits - logits.max())
    return float(np.exp(x[0])), pi / pi.sum()


def fit_empirical_bayes(counts: FloatArray) -> EBFit:
    """Maximize the marginal likelihood over (alpha, pi).

    alpha is optimized on the log scale and pi through a softmax with the last
    logit fixed at 0, so the search is unconstrained apart from the alpha bounds.
    """
    n = np.asarray(counts, dtype=np.float64)
    k = n.shape[-1]
    pooled = (n.sum(axis=0) + 0.5) / (n.sum() + 0.5 * k)
    logits = np.log(pooled[:-1]) - np.log(pooled[-1])
    x0 = np.concatenate([[np.log(10.0)], logits])

    def neg_ll(x: FloatArray) -> float:
        alpha, pi = _unpack(x)
        return -log_marginal_likelihood(n, alpha, pi)

    bounds = [LOG_ALPHA_BOUNDS] + [(None, None)] * (k - 1)
    res = optimize.minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    alpha, pi = _unpack(np.asarray(res.x, dtype=np.float64))
    return EBFit(alpha=alpha, pi=pi, log_lik=-float(res.fun), converged=bool(res.success))
