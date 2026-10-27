"""Properties of Dawid-Skene that must hold for any votes."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from cornerman.weak_labels.dawid_skene import ABSTAIN, compatible_init, fit

K = 3
INIT = [compatible_init(np.eye(3)), compatible_init(np.array([[1, 0], [0, 1], [0, 1]]))]


@st.composite
def vote_tables(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(5, 40))
    a = draw(st.lists(st.integers(ABSTAIN, 2), min_size=n, max_size=n))
    b = draw(st.lists(st.integers(ABSTAIN, 1), min_size=n, max_size=n))
    return np.column_stack([a, b])


@settings(max_examples=40, deadline=None)
@given(vote_tables(), st.randoms(use_true_random=False))
def test_item_order_does_not_matter(votes: np.ndarray, rnd: object) -> None:
    perm = np.random.default_rng(0).permutation(len(votes))
    a = fit(votes, INIT)
    b = fit(votes[perm], INIT)
    assert np.allclose(a.posterior[perm], b.posterior, atol=1e-8)
    for x, y in zip(a.confusion, b.confusion, strict=True):
        assert np.allclose(x, y, atol=1e-8)


@settings(max_examples=40, deadline=None)
@given(vote_tables())
def test_a_labeller_that_always_abstains_changes_nothing(votes: np.ndarray) -> None:
    silent = np.column_stack([votes, np.full(len(votes), ABSTAIN)])
    a = fit(votes, INIT)
    b = fit(silent, [*INIT, compatible_init(np.eye(3))])
    assert np.allclose(a.posterior, b.posterior, atol=1e-8)


@settings(max_examples=40, deadline=None)
@given(vote_tables())
def test_renaming_a_labellers_labels_only_permutes_its_confusion(votes: np.ndarray) -> None:
    """Swap the two labels of labeller 1 (in the votes and its init): same posterior, and
    its confusion matrix comes back with the columns swapped."""
    swapped = votes.copy()
    v = swapped[:, 1]
    swapped[:, 1] = np.where(v == ABSTAIN, ABSTAIN, 1 - v)
    a = fit(votes, INIT)
    b = fit(swapped, [INIT[0], INIT[1][:, ::-1]])
    assert np.allclose(a.posterior, b.posterior, atol=1e-8)
    assert np.allclose(a.confusion[1], b.confusion[1][:, ::-1], atol=1e-8)


@settings(max_examples=40, deadline=None)
@given(vote_tables())
def test_outputs_are_distributions(votes: np.ndarray) -> None:
    r = fit(votes, INIT)
    assert np.allclose(r.posterior.sum(axis=1), 1)
    assert np.allclose(r.prior.sum(), 1)
    for c in r.confusion:
        assert np.allclose(c.sum(axis=1), 1) and (c > 0).all()


@settings(max_examples=30, deadline=None)
@given(vote_tables())
def test_constrained_rows_spread_errors_evenly(votes: np.ndarray) -> None:
    compat = [np.eye(3), np.array([[1, 0], [0, 1], [0, 1]])]
    r = fit(votes, INIT, compat=compat)
    c = r.confusion[0]
    for k in range(3):
        wrong = np.delete(c[k], k)
        assert np.allclose(wrong, wrong[0]) and np.isclose(c[k].sum(), 1)


@settings(max_examples=30, deadline=None)
@given(vote_tables())
def test_tied_classes_share_a_prior(votes: np.ndarray) -> None:
    r = fit(votes, INIT, tied=[(1, 2)])
    assert np.isclose(r.prior[1], r.prior[2])
