"""Parameter recovery for Dawid-Skene (weak_labels/dawid_skene.py).

Simulate items with a known true class, labellers with known confusion matrices,
their own alphabets and abstentions; fit by EM; check that the confusions and the
class prior come back, and that the posteriors are calibrated: when the model says
an item is class k with probability 0.8, it is right about 80% of the time. Seeds
are fixed, so a failure is a real change, not bad luck.

The design has to be identifiable. With only one full-alphabet labeller and two
that each merge two classes, EM converges to a different answer that fits the votes
just as well (confusion errors of 0.1 to 0.15 on every seed tried): Kruskal's
condition for a unique decomposition fails. So, like the VLM in the real data, a
second labeller here sees the full alphabet (test_coarse_labellers_alone_...
shows the failure).
"""

import numpy as np
from scipy import stats

from cornerman.weak_labels.dawid_skene import ABSTAIN, compatible_init, fit

K = 4
PRIOR = np.array([0.5, 0.2, 0.2, 0.1])
# Labeller 0: the full alphabet, fairly good.
C0 = np.array(
    [
        [0.85, 0.05, 0.05, 0.05],
        [0.10, 0.75, 0.10, 0.05],
        [0.10, 0.10, 0.70, 0.10],
        [0.05, 0.05, 0.10, 0.80],
    ]
)
# Labeller 1: three labels; can't tell classes 1 and 2 apart (like "clinch" without
# a controller).
C1 = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
# Labeller 2: speaks about classes 1 and 2 only (like the fence rule); "other" else.
C2 = np.array([[0.9, 0.05, 0.05], [0.2, 0.75, 0.05], [0.2, 0.05, 0.75], [0.9, 0.05, 0.05]])
# Labeller 3: the full alphabet, noisier, often silent.
C3 = np.array(
    [
        [0.75, 0.10, 0.10, 0.05],
        [0.10, 0.70, 0.15, 0.05],
        [0.10, 0.15, 0.70, 0.05],
        [0.10, 0.05, 0.05, 0.80],
    ]
)
TRUE = [C0, C1, C2, C3]
ABSTAIN_RATE = [0.4, 0.3, 0.5, 0.5]
COMPAT = [
    np.eye(4),
    np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    np.eye(4),
]


def simulate(
    rng: np.random.Generator, n: int, labellers: list[np.ndarray] = TRUE
) -> tuple[np.ndarray, np.ndarray]:
    z = rng.choice(K, size=n, p=PRIOR)
    votes = np.full((n, len(labellers)), ABSTAIN)
    for j, c in enumerate(labellers):
        cum = np.cumsum(c[z], axis=1)
        says = (rng.random((n, 1)) > cum).sum(axis=1)
        speaks = rng.random(n) >= ABSTAIN_RATE[j]
        votes[speaks, j] = says[speaks]
    return z, votes


def test_recovers_confusions_and_prior() -> None:
    """Errors shrink with n (about 0.06, 0.02, 0.01 at 8k, 30k, 100k items): EM is
    consistent here. At 30k the worst confusion entry is within 0.04."""
    rng = np.random.default_rng(0)
    _, votes = simulate(rng, 30000)
    res = fit(votes, [compatible_init(c) for c in COMPAT])
    assert np.abs(res.prior - PRIOR).max() < 0.01
    for est, true in zip(res.confusion, TRUE, strict=True):
        assert np.abs(est - true).max() < 0.04
    # EM never lowers the likelihood (up to the tiny smoothing prior).
    assert np.all(np.diff(res.loglik) > -1e-6 * abs(res.loglik[-1]))


def test_posteriors_are_calibrated() -> None:
    """The 90% credible set (most likely classes until 0.9 of the mass) holds the truth
    at the rate its mass promises, and accuracy matches stated confidence per bin."""
    rng = np.random.default_rng(1)
    z, votes = simulate(rng, 30000)
    res = fit(votes, [compatible_init(c) for c in COMPAT])
    voted = res.voted
    post, truth = res.posterior[voted], z[voted]
    order = np.argsort(-post, axis=1)
    cum = np.cumsum(np.take_along_axis(post, order, axis=1), axis=1)
    size = (cum < 0.9).sum(axis=1) + 1
    in_set = np.array([t in o[:s] for t, o, s in zip(truth, order, size, strict=True)])
    mass = cum[np.arange(len(cum)), size - 1]
    p = stats.binomtest(int(in_set.sum()), len(in_set), float(mass.mean())).pvalue
    assert p > 0.01, (in_set.mean(), mass.mean())
    conf = post.max(axis=1)
    right = post.argmax(axis=1) == truth
    for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        sel = (conf >= lo) & (conf < hi)
        assert sel.sum() > 200
        assert abs(right[sel].mean() - conf[sel].mean()) < 0.04


def test_beats_majority_of_the_full_alphabet_labeller() -> None:
    """Combining helps: the posterior's best guess beats labeller 0 alone where it voted."""
    rng = np.random.default_rng(2)
    z, votes = simulate(rng, 5000)
    res = fit(votes, [compatible_init(c) for c in COMPAT])
    has0 = votes[:, 0] != ABSTAIN
    alone = (votes[has0, 0] == z[has0]).mean()
    combined = (res.posterior[has0].argmax(axis=1) == z[has0]).mean()
    assert combined > alone + 0.03


def test_coarse_labellers_alone_are_not_identifiable() -> None:
    """Without the second full-alphabet labeller the fit lands elsewhere: a documented
    limit (Kruskal), not a bug. If this starts passing, the comment above is wrong."""
    rng = np.random.default_rng(0)
    _, votes = simulate(rng, 30000, TRUE[:3])
    res = fit(votes, [compatible_init(c) for c in COMPAT[:3]])
    worst = max(float(np.abs(e - t).max()) for e, t in zip(res.confusion, TRUE[:3], strict=True))
    assert worst > 0.08


def test_constrained_model_recovers_accuracies_from_two_labellers() -> None:
    """The constrained model needs less: two labellers that each cover only part of the
    classes (like the rules and the mask labeller) plus one full-alphabet labeller that
    is often silent. Truth drawn from the constrained family; each accuracy comes back."""
    rng = np.random.default_rng(3)
    compat = [
        np.array([[1, 0], [1, 0], [0, 1], [0, 1]]),  # says "first half" / "second half"
        np.eye(4),
        np.eye(4),
    ]
    acc = [np.array([0.9, 0.8, 0.85, 0.7]), np.array([0.8, 0.7, 0.75, 0.9]),
           np.array([0.7, 0.75, 0.8, 0.85])]  # fmt: skip
    true: list[np.ndarray] = []
    for c, a in zip(compat, acc, strict=True):
        n_ok = c.sum(1, keepdims=True)
        n_bad = c.shape[1] - n_ok
        true.append(np.where(c > 0, a[:, None] / n_ok, (1 - a[:, None]) / n_bad))
    z = rng.choice(K, size=100000, p=PRIOR)  # errors shrink slowly: ~0.05 at 30k
    votes = np.full((len(z), 3), ABSTAIN)
    for j, (c, rate) in enumerate(zip(true, (0.3, 0.5, 0.8), strict=True)):
        cum = np.cumsum(c[z], axis=1)
        says = (rng.random((len(z), 1)) > cum).sum(axis=1)
        speaks = rng.random(len(z)) >= rate
        votes[speaks, j] = says[speaks]
    res = fit(votes, [compatible_init(c) for c in compat], compat=compat)
    assert np.abs(res.prior - PRIOR).max() < 0.02
    for est, t in zip(res.confusion, true, strict=True):
        assert np.abs(est - t).max() < 0.04
