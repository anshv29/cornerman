"""State classifier (Layer 2): features per moment to the pair state, smoothed in time.

Input: the moments table (events/moments.py) with the VLM attached. Training
labels: the weak labels of weak_labels/labellers.py, P(family) from Dawid-Skene
times P(direction | family). Model: scikit-learn's HistGradientBoostingClassifier,
which is fast on CPU and takes missing values as they are (skeleton features are
missing in every touching moment, red and blue ones outside live shots). Soft
labels are used as sample weights: each moment appears once per pair state it may
be in, weighted by that probability.

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
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (  # pyright: ignore[reportMissingTypeStubs]
    HistGradientBoostingClassifier,  # pyright: ignore[reportUnknownVariableType]
)

from cornerman.events.moments import FEATURES
from cornerman.weak_labels.labellers import (
    FAMILIES,
    FAMILY,
    PAIR_STATES,
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


def train(moments: pd.DataFrame, seed: int) -> tuple[Any, dict[str, object]]:
    """Fit family Dawid-Skene and the classifier on these moments. Returns the model and
    a summary of the Dawid-Skene fit."""
    votes = family_votes(moments)
    fam = fit_family(votes)
    soft = soft_labels(moments, fam.posterior)
    keep = fam.voted
    x = moments.loc[keep, feature_columns()].to_numpy(float)
    s = soft[keep]
    rows, cls = np.nonzero(s >= MIN_WEIGHT)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    model.fit(x[rows], cls, sample_weight=s[rows, cls])  # pyright: ignore[reportUnknownMemberType]
    summary: dict[str, object] = {
        "prior": dict(zip(FAMILIES, fam.prior.round(3).tolist(), strict=True)),
        "rules_confusion": fam.confusion[0].round(3).tolist(),
        "vlm_confusion": fam.confusion[1].round(3).tolist(),
        "n_voted": int(keep.sum()),
    }
    return model, summary


def predict_proba(model: Any, moments: pd.DataFrame) -> np.ndarray:
    """(n, 8) probabilities over PAIR_STATES (classes never seen in training get 0)."""
    p = np.asarray(model.predict_proba(moments[feature_columns()].to_numpy(float)), float)
    out = np.zeros((len(moments), len(PAIR_STATES)))
    classes = np.asarray(model.classes_, int)
    out[:, classes] = p
    return out


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


def smooth(moments: pd.DataFrame, probs: np.ndarray, stay: float, max_gap_s: float) -> np.ndarray:
    """Pair state index per moment: family by Viterbi, then direction by Viterbi within
    each run of one family. `moments` must be sorted by fight and time."""
    members = [[PAIR_STATES.index(p) for p in PAIR_STATES if FAMILY[p] == f] for f in FAMILIES]
    fam_probs = np.stack([probs[:, idx].sum(1) for idx in members], axis=1)
    fam_trans = transitions(len(FAMILIES), stay)
    out = np.zeros(len(moments), int)
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
            sub = sub / np.clip(sub.sum(1, keepdims=True), 1e-12, None)
            out[i:j] = np.asarray(idx)[viterbi(sub, transitions(len(idx), stay, forbid))]
            i = j
    return out


def cross_validate(
    moments: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 0,
    stay: float = 0.9,
    max_gap_s: float = 1.0,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Out-of-fold predictions for every moment: probabilities per pair state, the raw
    and smoothed pair state, its family, and the fold. Also each fold's Dawid-Skene
    summary."""
    m = with_window(moments)
    pieces: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for i, test in enumerate(folds(m["fight_id"].unique().tolist(), n_folds, seed)):
        is_test = m["fight_id"].isin(test)
        model, summary = train(m[~is_test], seed)
        summaries.append({"fold": i, **summary})
        part = m[is_test].copy()
        probs = predict_proba(model, part)
        for k, pair in enumerate(PAIR_STATES):
            part[f"p_{pair}"] = probs[:, k]
        members = [[PAIR_STATES.index(p) for p in PAIR_STATES if FAMILY[p] == f] for f in FAMILIES]
        fam_p = np.stack([probs[:, idx].sum(1) for idx in members], axis=1)
        part["raw_family"] = np.array(FAMILIES)[fam_p.argmax(1)]
        part["pred"] = np.array(PAIR_STATES)[smooth(part, probs, stay, max_gap_s)]
        part["fold"] = i
        pieces.append(part)
    out = pd.concat(pieces).sort_values(["fight_id", "t"]).reset_index(drop=True)
    out["pred_family"] = out["pred"].map(FAMILY)
    out["family_only"] = ~out["live"].astype(bool)
    return out, summaries
