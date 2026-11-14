"""State classifier (Layer 2): features per moment to the pair state, smoothed in time.

Input: the moments table (events/moments.py) with the VLM attached. Training
labels: the weak labels of weak_labels/labellers.py, P(family) from Dawid-Skene
times P(direction | family). Model: scikit-learn's HistGradientBoostingClassifier,
which is fast on CPU and takes missing values as they are (skeleton features are
missing in every touching moment, red and blue ones outside live shots). Soft
labels are used as sample weights: each moment appears once per pair state it may
be in, weighted by that probability. By default the weights are also class-balanced
at the family level (`family_balance`): clinch is about 8% of the label mass and
standing about 68%, and unbalanced the model never predicted clinch.

No leakage: grouped 5-fold cross-validation by fight. For each fold, Dawid-Skene is
refit on the training fights only, the classifier learns from them, and the held-out
fights are predicted. Every number in eval/states.py is out of fold, so a fight's
own labels (including its VLM answers, which silver is built from) never reach the
model that predicts it.

Decisions are made in two steps, family first. A plain argmax over the 8 pair
states is biased against clinch and ground: when nothing says who controls a
clinch, its probability is split evenly between red-clinch and blue-clinch, and
neutral wins with less than half the mass (the first run predicted standing 98% of
the time while the labels put clinch near 15%). So:
1. family: Viterbi over standing, clinch and ground, on the summed probabilities;
2. direction: within each run of one family, Viterbi over that family's states on
   the probabilities renormalised inside the family.
Both use sticky transitions (stay with probability `stay`, else move evenly), over
each run of moments (a gap of more than `max_gap_s` starts a new run). Red-top and
blue-top can't swap directly: a reversal passes through a scramble.

Outside live shots red and blue are unknown, so there the direction of a prediction
means nothing and only its family is used (column `family_only`).

A direction is only reported where the model is sure of it (column
`direction_known`): the forward-backward marginal of the Viterbi state, within its
family run, must reach DIRECTION_MIN. Without this, Viterbi turned near-ties into
long runs of one colour: 67% of ground top calls sat on moments where neither
fighter reached 0.7, and those leaned red 2 to 1 while the confident ones were even
(DECISIONS.md, 2026-11-14). Clinch direction is never reported for now (Ansh,
2026-11-14): the controller rests on one unconfirmed VLM answer.

The classifier can't prefer a colour (Ansh, 2026-11-14: swapping red and blue in the
inputs must flip every output exactly). A tree model isn't symmetric on its own: it
learns from whichever colour the labels lean to, and splits break ties by column
order. So it is trained on every moment twice, as it is and with the colours swapped
(`swap_colours`), in one fixed row order, and its answer is the average of the answer
for the moment and the mirrored answer for the swapped moment. Sums over a family's
states are taken in an order that doesn't depend on the colours, so even rounding is
the same both ways. tests/property/test_colour_swap.py checks the whole chain.

Balanced training makes the probabilities say clinch and ground too often. Anything
per fighter uses `q_*` instead of `p_*`: the family probabilities moved from the
training mix back to the family rates over all fighting time of the training fights
(`prior_correct`), with directions keeping their relative mass. The decision `pred`
stays on the balanced probabilities, which find clinch.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (  # pyright: ignore[reportMissingTypeStubs]
    HistGradientBoostingClassifier,  # pyright: ignore[reportUnknownVariableType]
)

from cornerman.events.moments import FEATURES, swap_colours
from cornerman.weak_labels.labellers import (
    FAMILIES,
    FAMILY,
    PAIR_STATES,
    SWAP,
    family_votes,
    fit_family,
    soft_labels,
)

WINDOW = [
    "gap",
    "iou",
    "contact",
    "mask_pair",
    "mask_ratio",
    "mask_top",
    "mask_dy",
    "red_speed",
    "blue_speed",
    "hip_diff",
    "red_fence",
    "blue_fence",
    "n_people",
]
"""Features also averaged over +-1 s (the moment and two either side, at 0.5 s)."""
MIN_WEIGHT = 0.02
"""Soft-label entries below this are dropped from training (they only add rows)."""
DIRECTION_MIN = 0.8
"""A direction is reported only where its forward-backward marginal reaches this. Set by
reasoning before any check: the measured trust in the directional labellers runs from
0.7 to 0.95, so a call below 0.8 is no better than one weak vote."""
NO_DIRECTION = ("clinch",)
"""Families whose direction is never reported (Ansh, 2026-11-14: clinch without the
controller for now)."""


def with_window(moments: pd.DataFrame, half: int = 2) -> pd.DataFrame:
    """Moments plus each WINDOW feature's mean over the moment and `half` neighbours on
    each side within the same fight (missing values skipped)."""
    out = moments.sort_values(["fight_id", "t"]).reset_index(drop=True)
    roll = (
        out.groupby("fight_id")[WINDOW]
        .rolling(2 * half + 1, center=True, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    for col in WINDOW:
        out[f"{col}_w"] = roll[col].astype(float)
    return out


def feature_columns() -> list[str]:
    return [*FEATURES, *(f"{c}_w" for c in WINDOW)]


def folds(fights: list[str], k: int, seed: int) -> list[list[str]]:
    """Fights split into k groups at random (seeded)."""
    order = np.random.default_rng(seed).permutation(sorted(fights)).tolist()
    return [order[i::k] for i in range(k)]


def family_balance(soft: np.ndarray) -> np.ndarray:
    """Per pair state, a weight that gives each family the same total training mass:
    (1 / number of families) / that family's share of the soft-label mass. Directions
    inside a family keep their relative mass."""
    mass = soft.sum(0)
    mass = mass + mass[SWAP]  # the same whichever colour the labels lean to
    total = math.fsum(mass)
    out = np.ones(len(PAIR_STATES))
    for f in FAMILIES:
        idx = [PAIR_STATES.index(p) for p in PAIR_STATES if FAMILY[p] == f]
        share = math.fsum(mass[idx]) / total if total > 0 else 0.0
        out[idx] = (1 / len(FAMILIES)) / share if share > 0 else 1.0
    return out


def row_sum(a: np.ndarray) -> np.ndarray:
    """Sum of each row, added up in sorted order so that it doesn't depend on the column
    order (red before blue or after): the same bits either way."""
    return np.sort(a, axis=1).sum(1)


def family_probs(probs: np.ndarray) -> np.ndarray:
    """(n, 3) probability of each family: its states' probabilities summed."""
    return np.stack([row_sum(probs[:, idx]) for idx in _members()], axis=1)


def _members() -> list[list[int]]:
    return [[PAIR_STATES.index(p) for p in PAIR_STATES if FAMILY[p] == f] for f in FAMILIES]


def family_mass(weights: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Share of the total weight in each family, for training rows of pair-state `classes`."""
    fam = np.array([FAMILIES.index(FAMILY[PAIR_STATES[int(c)]]) for c in classes], int)
    mass = np.bincount(fam, weights=weights, minlength=len(FAMILIES))
    return mass / mass.sum()


def prior_correct(probs: np.ndarray, trained: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """(n, 8) probabilities moved from the family mix the model was trained on (`trained`)
    to the family rates of the data (`rates`): each family's states are scaled by
    rates / trained, then each row is renormalised. Directions inside a family keep
    their relative mass. This is the usual prior-shift correction: the classifier
    estimates P(family | x) under the training prior, and Bayes' rule swaps the prior."""
    scale = np.ones(len(PAIR_STATES))
    for k, p in enumerate(PAIR_STATES):
        f = FAMILIES.index(FAMILY[p])
        scale[k] = rates[f] / trained[f] if trained[f] > 0 else 0.0
    out = probs * scale
    return out / np.clip(row_sum(out)[:, None], 1e-12, None)


def train(moments: pd.DataFrame, seed: int, balanced: bool = True) -> tuple[Any, dict[str, object]]:
    """Fit family Dawid-Skene and the classifier on these moments. Returns the model and
    a summary: the Dawid-Skene fit, the family mix the model was trained on ("trained")
    and the family rates over all of these moments ("rates", the mean family
    posterior, which is the Dawid-Skene prior where nobody voted)."""
    votes = family_votes(moments)
    fam = fit_family(votes)
    soft = soft_labels(moments, fam.posterior)
    keep = fam.voted
    x = moments.loc[keep, feature_columns()].to_numpy(float)
    s = soft[keep]
    # Every moment twice, as it is and with the colours swapped, in a fixed row order:
    # the training set is then the same rows whichever way round the colours came in.
    x2 = np.vstack([x, swap_colours(moments.loc[keep]).loc[:, feature_columns()].to_numpy(float)])
    s2 = np.vstack([s, s[:, SWAP]])
    rows, cls = np.nonzero(s2 >= MIN_WEIGHT)
    w = s2[rows, cls] * (family_balance(s)[cls] if balanced else 1.0)
    xr = x2[rows]
    order = np.lexsort([w, cls, *xr.T[::-1]])
    xr, cls, w = xr[order], cls[order], w[order]
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    model.fit(xr, cls, sample_weight=w)  # pyright: ignore[reportUnknownMemberType]
    summary: dict[str, object] = {
        "prior": dict(zip(FAMILIES, fam.prior.round(3).tolist(), strict=True)),
        "rules_confusion": fam.confusion[0].round(3).tolist(),
        "vlm_confusion": fam.confusion[1].round(3).tolist(),
        "n_voted": int(keep.sum()),
        "trained": family_mass(w, cls).tolist(),
        "rates": fam.posterior.mean(0).tolist(),
    }
    return model, summary


def _raw_proba(model: Any, moments: pd.DataFrame) -> np.ndarray:
    p = np.asarray(model.predict_proba(moments[feature_columns()].to_numpy(float)), float)
    out = np.zeros((len(moments), len(PAIR_STATES)))
    out[:, np.asarray(model.classes_, int)] = p
    return out


def predict_proba(model: Any, moments: pd.DataFrame) -> np.ndarray:
    """(n, 8) probabilities over PAIR_STATES (classes never seen in training get 0): the
    average of the model's answer and its mirrored answer for the colour-swapped moments."""
    return (_raw_proba(model, moments) + _raw_proba(model, swap_colours(moments))[:, SWAP]) / 2


def transitions(k: int, stay: float, forbidden: list[tuple[int, int]] | None = None) -> np.ndarray:
    """k x k sticky transition matrix; `forbidden` pairs (both ways) get probability 0."""
    t = np.ones((k, k))
    for a, b in forbidden or []:
        t[a, b] = t[b, a] = 0.0
    np.fill_diagonal(t, 0.0)
    if k > 1:
        t = t / t.sum(1, keepdims=True) * (1 - stay)
    np.fill_diagonal(t, stay if k > 1 else 1.0)
    return t


def viterbi(probs: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Most likely state path given per-step probabilities and a transition matrix."""
    n, k = probs.shape
    if n == 0:
        return np.zeros(0, int)
    with np.errstate(divide="ignore"):
        le, lt = np.log(np.clip(probs, 1e-12, None)), np.log(trans)
    score = le[0] - np.log(k)
    back = np.zeros((n, k), int)
    for i in range(1, n):
        cand = score[:, None] + lt
        back[i] = cand.argmax(axis=0)
        score = cand.max(axis=0) + le[i]
    path = np.zeros(n, int)
    path[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def marginals(probs: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(n, k) posterior P(state at step i | all steps) by forward-backward, with the
    same uniform start as `viterbi`. Each step is rescaled to sum to 1 (only ratios
    matter), so long runs don't underflow."""
    n, k = probs.shape
    if n == 0:
        return np.zeros((0, k))
    e = np.clip(probs, 1e-12, None)
    fwd = np.zeros((n, k))
    bwd = np.ones((n, k))
    fwd[0] = e[0] / e[0].sum()
    for i in range(1, n):
        f = (fwd[i - 1] @ trans) * e[i]
        fwd[i] = f / f.sum()
    for i in range(n - 2, -1, -1):
        b = trans @ (e[i + 1] * bwd[i + 1])
        bwd[i] = b / b.sum()
    post = fwd * bwd
    return post / post.sum(1, keepdims=True)


def _runs(moments: pd.DataFrame, max_gap_s: float) -> list[tuple[int, int]]:
    """(start, end) of each run of moments: same fight, at most `max_gap_s` apart."""
    fight = moments["fight_id"].to_numpy()
    t = moments["t"].to_numpy(float)
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(moments) + 1):
        if i == len(moments) or fight[i] != fight[i - 1] or t[i] - t[i - 1] > max_gap_s:
            out.append((start, i))
            start = i
    return out


def smooth(
    moments: pd.DataFrame, probs: np.ndarray, stay: float, max_gap_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pair state index per moment: family by Viterbi, then direction by Viterbi within
    each run of one family. Also how sure the direction is: the forward-backward
    marginal of the chosen state within its family run (1 for a family with one state).
    `moments` must be sorted by fight and time."""
    members = _members()
    fam_probs = family_probs(probs)
    fam_trans = transitions(len(FAMILIES), stay)
    out = np.zeros(len(moments), int)
    sure = np.ones(len(moments))
    for a, b in _runs(moments, max_gap_s):
        fam = viterbi(fam_probs[a:b], fam_trans)
        i = a
        while i < b:
            j = i
            while j < b and fam[j - a] == fam[i - a]:
                j += 1
            idx = members[int(fam[i - a])]
            names = [PAIR_STATES[k] for k in idx]
            forbid = (
                [(names.index("red-top"), names.index("blue-top"))] if "red-top" in names else []
            )
            sub = probs[i:j][:, idx]
            sub = sub / np.clip(row_sum(sub)[:, None], 1e-12, None)
            trans = transitions(len(idx), stay, forbid)
            path = viterbi(sub, trans)
            out[i:j] = np.asarray(idx)[path]
            sure[i:j] = marginals(sub, trans)[np.arange(j - i), path]
            i = j
    return out, sure


def cross_validate(
    moments: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 0,
    stay: float = 0.9,
    max_gap_s: float = 1.0,
    balanced: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Out-of-fold predictions for every moment: probabilities per pair state (`p_*` as
    trained, `q_*` corrected to the family rates), the raw and smoothed pair state, its
    family, how sure the direction is and whether it counts as known, and the fold.
    Also each fold's training summary."""
    m = with_window(moments)
    pieces: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for i, test in enumerate(folds(m["fight_id"].unique().tolist(), n_folds, seed)):
        is_test = m["fight_id"].isin(test)
        model, summary = train(m[~is_test], seed, balanced)
        summaries.append({"fold": i, **summary})
        part = m[is_test].copy()
        probs = predict_proba(model, part)
        corrected = prior_correct(probs, np.asarray(summary["trained"]),
                                  np.asarray(summary["rates"]))  # fmt: skip
        for k, pair in enumerate(PAIR_STATES):
            part[f"p_{pair}"] = probs[:, k]
            part[f"q_{pair}"] = corrected[:, k]
        part["raw_family"] = np.array(FAMILIES)[family_probs(probs).argmax(1)]
        path, sure = smooth(part, probs, stay, max_gap_s)
        part["pred"] = np.array(PAIR_STATES)[path]
        part["direction_sure"] = sure
        part["fold"] = i
        pieces.append(part)
    out = pd.concat(pieces).sort_values(["fight_id", "t"]).reset_index(drop=True)
    out["pred_family"] = out["pred"].map(FAMILY)
    out["family_only"] = ~out["live"].astype(bool)
    out["direction_known"] = (
        ~out["family_only"]
        & ~out["pred_family"].isin(NO_DIRECTION)
        & (out["direction_sure"] >= DIRECTION_MIN)
    )
    return out, summaries
