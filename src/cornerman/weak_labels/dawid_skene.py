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

`compat` switches to a constrained model, needed when labellers rarely vote on the
same item. Confusions are learned from agreement between labellers; where they
seldom overlap, the free K x L_j matrices have too many parameters and EM drifts to
answers where the classes lose their meaning (seen on the real moments: the rules
and the mask labeller never vote on the same moment, by design, and "neutral"
ended up explaining the masks' clinch votes). The constrained row for class k is

    C_j[k, l] = a_jk * s_jk[l]               for labels l compatible with k,
    C_j[k, l] = (1 - a_jk) / (#incompatible) for the others,

with a_jk the labeller's accuracy on class k (the chance its label fits) and s_jk
a split over the compatible labels, both learned. Errors are spread evenly, so a
class can't quietly take over another class's labels. A class with no compatible
label in an alphabet gets a uniform row: that labeller says nothing about it.
The M step is closed form: a_jk = (mass on compatible labels + 1) / (all mass + 2),
a Beta(1, 1) prior, and s_jk is proportional to the compatible counts.

`tied` pairs of classes share one prior probability. Here they are mirror images
(red on top, blue on top): the corner colour is arbitrary, and a vote that names no
fighter ("they are clinched") leaves the likelihood flat in how the prior splits
the pair, so untied EM drifts to giving it all to one side.

`fixed` labellers keep their starting confusion matrix: an anchor whose accuracy
was measured elsewhere. With two labellers, free confusions are not identified;
fixing one of them makes the other's estimable from their agreement.

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
    compat: list[np.ndarray] | None = None,
    tied: list[tuple[int, int]] | None = None,
    fixed: list[int] | None = None,
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
        for a, b in tied or []:
            pi[a] = pi[b] = (pi[a] + pi[b]) / 2
        new: list[np.ndarray] = []
        for j, c in enumerate(conf):
            if j in (fixed or []):
                new.append(c)
                continue
            counts = np.zeros(c.shape)
            voted = v[:, j] != ABSTAIN
            np.add.at(counts.T, v[voted, j], q[voted])
            if compat is None:
                counts += smoothing
                new.append(counts / counts.sum(1, keepdims=True))
            else:
                new.append(_constrained(counts, np.asarray(compat[j]) > 0, smoothing))
        conf = new
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol * max(1.0, abs(ll)):
            break
    post = np.tile(pi, (len(votes), 1))
    post[seen] = posterior(v, conf, pi)
    return DSResult(posterior=post, prior=pi, confusion=conf, loglik=history, voted=seen)


def _constrained(counts: np.ndarray, fits: np.ndarray, smoothing: float) -> np.ndarray:
    """The constrained confusion matrix (see the module docstring) from expected counts."""
    out = np.empty(counts.shape)
    n_labels = counts.shape[1]
    for k in range(len(counts)):
        ok = fits[k]
        n_ok = int(ok.sum())
        if n_ok == 0:
            out[k] = 1.0 / n_labels
            continue
        split = counts[k, ok] + smoothing
        split = split / split.sum()
        if n_ok == n_labels:
            out[k] = split
            continue
        acc = (counts[k, ok].sum() + 1) / (counts[k].sum() + 2)
        out[k, ok] = acc * split
        out[k, ~ok] = (1 - acc) / (n_labels - n_ok)
    return out


def posterior(votes: np.ndarray, confusion: list[np.ndarray], prior: np.ndarray) -> np.ndarray:
    """The posterior over the true class for items given fitted confusions and prior."""
    p, _ = _normalise(
        _log_lik_terms(np.asarray(votes, int), [np.log(c) for c in confusion], np.log(prior))
    )
    return p


def fit_prior(
    votes: np.ndarray, confusion: list[np.ndarray], start: np.ndarray, n_iter: int = 200
) -> np.ndarray:
    """EM for the class prior alone, confusions held fixed, over the items with a vote.
    Used when confusions are fitted where labellers overlap (which identifies them) but
    the overlap isn't a representative sample of items."""
    v = np.asarray(votes, int)
    v = v[(v != ABSTAIN).any(axis=1)]
    pi = np.asarray(start, float) / np.sum(start)
    for _ in range(n_iter):
        pi = posterior(v, confusion, pi).mean(axis=0)
    return pi


def accuracy_estimate(result: DSResult) -> float:
    """Expected share of voted items whose most likely class is the true one, from the
    model alone (no labels): the mean of each item's highest posterior probability.
    Only as good as the model's independence assumption."""
    return float(result.posterior[result.voted].max(axis=1).mean())
