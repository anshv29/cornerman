"""Dawid-Skene: combine noisy labellers without ground truth (SPEC Layer 3).

Each item i (here a moment of a fight) has a true class z_i in 1..K that nobody
sees. Each labeller j either abstains or gives a label from its own alphabet of
L_j labels; its behaviour is a confusion matrix C_j (K x L_j), where C_j[k, l] is
the probability it says l when the truth is k. Given the truth, labellers err
independently. The prior pi[k] is how common each class is.

The likelihood of one item's votes, summed over the unknown truth, is

    p(votes_i) = sum_k pi[k] * prod_{j voting on i} C_j[k, v_ij]

and EM maximises it over pi and every C_j:
- E step: the posterior over the truth of each item,
  q_i(k) proportional to pi[k] * prod_j C_j[k, v_ij];
- M step: pi[k] = mean over items of q_i(k); C_j[k, l] = share of the posterior
  mass of class k, among items labeller j voted on, that it labelled l.
A small pseudo-count (`smoothing`) is added in the M step, a Dirichlet prior
that keeps a confusion row away from exact zeros when a class is rare.

Labellers keep their own alphabets (the rules say "clinch" without a controller,
the fence rule speaks only about cage pressure), which plain majority voting
cannot handle. The labels mean nothing to EM by themselves: classes can swap
names and fit the data just as well. The starting confusion matrices (`init`)
tie each class to the labels that are compatible with it, which fixes the names.

Independence matters. Two labellers that make the same mistakes (two prompts of
one VLM) would be counted as two independent witnesses and swamp the others, so
they enter as one labeller (weak_labels/labellers.py).
"""

from dataclasses import dataclass

import numpy as np

ABSTAIN = -1


@dataclass(frozen=True)
class DSResult:
    posterior: np.ndarray
    """(n_items, K): the probability of each true class for each item."""
    prior: np.ndarray
    """(K,)"""
    confusion: list[np.ndarray]
    """Per labeller, (K, L_j): rows sum to 1."""
    loglik: list[float]
    """Marginal log-likelihood after each iteration (never decreases, up to smoothing)."""
    voted: np.ndarray
    """(n_items,) True where at least one labeller voted."""


def _log_lik_terms(
    votes: np.ndarray, log_conf: list[np.ndarray], log_prior: np.ndarray
) -> np.ndarray:
    """(n, K) log pi[k] + sum_j log C_j[k, v_ij] over the labellers that voted."""
    out = np.tile(log_prior, (len(votes), 1))
    for j, lc in enumerate(log_conf):
        v = votes[:, j]
        voted = v != ABSTAIN
        out[voted] += lc[:, v[voted]].T
    return out


def _normalise(log_terms: np.ndarray) -> tuple[np.ndarray, float]:
    m = log_terms.max(axis=1, keepdims=True)
    p = np.exp(log_terms - m)
    s = p.sum(axis=1, keepdims=True)
    return p / s, float((np.log(s) + m).sum())


def compatible_init(compat: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """A starting confusion matrix from a K x L 0/1 table of which labels fit which
    class: `strength` of each row spread over its compatible labels, the rest over all."""
    compat = np.asarray(compat, float)
    k, n_labels = compat.shape
    fit = np.where(compat.sum(1, keepdims=True) > 0, compat, 1.0)
    fit = fit / fit.sum(1, keepdims=True)
    return strength * fit + (1 - strength) / n_labels * np.ones((k, n_labels))


def fit(
    votes: np.ndarray,
    init: list[np.ndarray],
    prior: np.ndarray | None = None,
    smoothing: float = 0.01,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> DSResult:
    """Fit Dawid-Skene by EM.

    `votes` is (n_items, n_labellers) of label indices, ABSTAIN where a labeller
    says nothing. `init` holds each labeller's starting K x L_j confusion matrix
    (which also sets K and each alphabet's size). Items nobody voted on get the
    prior as their posterior and play no part in the fit.
    """
    votes = np.asarray(votes, int)
    k = init[0].shape[0]
    if votes.shape[1] != len(init):
        raise ValueError("one init matrix per labeller")
    for j, c in enumerate(init):
        if c.shape[0] != k:
            raise ValueError("every init matrix needs K rows")
        if votes[:, j].max(initial=ABSTAIN) >= c.shape[1]:
            raise ValueError(f"labeller {j} has a label outside its alphabet")
    seen = (votes != ABSTAIN).any(axis=1)
    v = votes[seen]
    conf = [np.asarray(c, float) / np.asarray(c, float).sum(1, keepdims=True) for c in init]
    pi = np.full(k, 1.0 / k) if prior is None else np.asarray(prior, float) / np.sum(prior)
    history: list[float] = []
    q = np.zeros((0, k))
    for _ in range(max_iter):
        q, ll = _normalise(_log_lik_terms(v, [np.log(c) for c in conf], np.log(pi)))
        history.append(ll)
        pi = (q.sum(0) + smoothing) / (len(q) + k * smoothing)
        new: list[np.ndarray] = []
        for j, c in enumerate(conf):
            counts = np.full(c.shape, smoothing)
            voted = v[:, j] != ABSTAIN
            np.add.at(counts.T, v[voted, j], q[voted])
            new.append(counts / counts.sum(1, keepdims=True))
        conf = new
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol * max(1.0, abs(ll)):
            break
    post = np.tile(pi, (len(votes), 1))
    post[seen] = posterior(v, conf, pi)
    return DSResult(posterior=post, prior=pi, confusion=conf, loglik=history, voted=seen)


def posterior(votes: np.ndarray, confusion: list[np.ndarray], prior: np.ndarray) -> np.ndarray:
    """The posterior over the true class for items given fitted confusions and prior."""
    p, _ = _normalise(
        _log_lik_terms(np.asarray(votes, int), [np.log(c) for c in confusion], np.log(prior))
    )
    return p


def accuracy_estimate(result: DSResult) -> float:
    """Expected share of voted items whose most likely class is the true one, from the
    model alone (no labels): the mean of each item's highest posterior probability.
    Only as good as the model's independence assumption."""
    return float(result.posterior[result.voted].max(axis=1).mean())
