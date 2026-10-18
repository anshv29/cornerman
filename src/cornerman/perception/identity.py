"""Who is who: label each tracklet as the red corner, the blue corner, the referee, or other.

A tracker gives short tracklets that break at every camera cut, occlusion and
clinch. Identity glues them back together across the whole fight. It runs
after pose, on colour sampled at gloves, shins and shorts (kitcolour.py), and
needs no GPU, so it can be rerun on the laptop as often as needed.

1. Summarise each tracklet by the median colour of its gloves, shins and
   shorts, as Lab chroma (a and b, centred on zero). Red kit has a strongly
   positive, blue kit b strongly negative; black kit and the referee's
   uniform sit near zero; skin is mildly positive in both.
2. Cluster tracklets into up to four groups (red, blue, referee, everyone
   else on the canvas such as cornermen) with a cannot-link rule: two
   tracklets seen in the same frame are two different people, so they may not
   share a group. When fewer than four people are really there, k-means splits
   one person in two, so groups whose colours are close (within MERGE_DIST)
   and which are never on screen together are merged back.
3. Name the groups by colour: the group with the reddest gloves and shins is
   the red corner, the bluest is the blue corner. Of the rest, the group with
   the most screen time is the referee (the camera follows the action, and
   the referee stays near it); any other group is "other".

Why not colour over the whole box: on the pilot, IMMAF kits were mostly black
with red or blue trim, the referee wore black, and the light-blue canvas filled
much of every box. Box-level colour couldn't separate the fighters
(DECISIONS.md, "Identity is the blocker").

Corners link to names through `corner_fighters`: IMMAF titles list the red
corner first (checked against the broadcast score bar, DECISIONS.md).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROLES = ("red", "blue", "referee")
FIGHTERS = ROLES[:2]
OTHER = "other"
N_GROUPS = 4
"""Red, blue, referee, and one group for everyone else on the canvas."""
PARTS = ("glove", "shin", "shorts")
FEATURES = [f"{p}_{c}" for p in PARTS for c in ("a", "b")]
CHROMA_SCALE = 40.0
"""Chroma units per feature unit: a fully red glove is about one unit away from black."""
MERGE_DIST = 0.5
"""Groups this close in colour (20 chroma units) that never co-occur are one person."""


def corner_features(samples: pd.DataFrame, min_pixels: int = 5) -> pd.DataFrame:
    """Per tracklet: median a and b (centred on 0) of each part, scaled. Missing parts are 0."""
    s = samples[samples["n"] >= min_pixels]
    med = s.groupby(["track_id", "part"])[["a", "b"]].median() - 128.0
    wide = med.unstack("part")
    out = pd.DataFrame(index=wide.index)
    for p in PARTS:
        for c in ("a", "b"):
            col = (c, p)
            out[f"{p}_{c}"] = wide[col] if col in wide.columns else np.nan
    return (out[FEATURES] / CHROMA_SCALE).fillna(0.0)


@dataclass(frozen=True)
class Tracklets:
    ids: np.ndarray
    features: np.ndarray
    weights: np.ndarray
    """Number of frames each tracklet covers."""
    conflicts: list[set[int]]
    """conflicts[i] = indices of tracklets that share a frame with tracklet i."""


def summarise(tracks: pd.DataFrame, features: pd.DataFrame, min_frames: int) -> Tracklets:
    """Tracklets with at least `min_frames` frames and a feature row."""
    counts = tracks.groupby("track_id").size()
    keep = [int(t) for t in counts[counts >= min_frames].index if t in features.index]
    ids = np.array(sorted(keep), dtype=np.int64)
    pos = {int(tid): i for i, tid in enumerate(ids)}
    conflicts: list[set[int]] = [set() for _ in ids]
    t = tracks[tracks["track_id"].isin(ids)]
    for _, group in t.groupby("frame")["track_id"]:
        members = [pos[int(x)] for x in group.unique()]
        for i in members:
            conflicts[i].update(m for m in members if m != i)
    return Tracklets(
        ids=ids,
        features=np.asarray(features.reindex(ids.tolist())[FEATURES], dtype=float),
        weights=np.asarray(counts.reindex(ids.tolist()), dtype=float),
        conflicts=conflicts,
    )


def _seeds(tl: Tracklets, k: int) -> list[int]:
    """Start from heavy tracklets that were on screen together, so surely different people."""
    order = [int(i) for i in np.argsort(-tl.weights, kind="stable")]
    chosen = [order[0]]
    for i in order[1:]:
        if len(chosen) == k:
            break
        if all(i in tl.conflicts[c] for c in chosen):
            chosen.append(i)
    while len(chosen) < min(k, len(order)):
        # Not enough co-occurring tracklets: take the one furthest from the chosen ones.
        d = np.min(
            np.linalg.norm(tl.features[:, None, :] - tl.features[chosen][None], axis=2), axis=1
        )
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return chosen


def cluster(tl: Tracklets, k: int, max_dist: float, n_iter: int = 20) -> np.ndarray:
    """Cannot-link k-means. Label per tracklet in 0..k-1, or -1 if no allowed group is near."""
    n = len(tl.ids)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    centroids = tl.features[_seeds(tl, k)].copy()
    labels = np.full(n, -1, dtype=np.int64)
    order = [int(x) for x in np.argsort(-tl.weights, kind="stable")]
    for _ in range(n_iter):
        new = np.full(n, -1, dtype=np.int64)
        dist = np.linalg.norm(tl.features[:, None, :] - centroids[None], axis=2)
        for i in order:
            taken = {int(new[j]) for j in tl.conflicts[i]}
            for c in np.argsort(dist[i], kind="stable"):
                if dist[i, c] > max_dist:
                    break
                if int(c) not in taken:
                    new[i] = c
                    break
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(len(centroids)):
            members = labels == c
            if members.any():
                centroids[c] = np.average(tl.features[members], axis=0, weights=tl.weights[members])
    return labels


def merge_groups(tl: Tracklets, labels: np.ndarray, max_dist: float = MERGE_DIST) -> np.ndarray:
    """Merge pairs of groups that look alike and are never on screen at the same time."""
    labels = labels.copy()
    while True:
        groups = sorted({int(g) for g in labels if g >= 0})
        centre = {
            g: np.average(tl.features[labels == g], axis=0, weights=tl.weights[labels == g])
            for g in groups
        }
        best: tuple[float, int, int] | None = None
        for i, a in enumerate(groups):
            members_a = np.flatnonzero(labels == a)
            for b in groups[i + 1 :]:
                d = float(np.linalg.norm(centre[a] - centre[b]))
                if d >= max_dist or (best is not None and d >= best[0]):
                    continue
                if any(labels[j] == b for m in members_a for j in tl.conflicts[int(m)]):
                    continue  # on screen together: two different people
                best = (d, a, b)
        if best is None:
            return labels
        labels[labels == best[2]] = best[1]


def name_groups(tl: Tracklets, labels: np.ndarray) -> dict[int, str]:
    """Red = reddest gloves and shins, blue = bluest, referee = most screen time of the rest."""
    groups = sorted({int(g) for g in labels if g >= 0})
    if not groups:
        return {}
    cols = {f: i for i, f in enumerate(FEATURES)}
    centre: dict[int, np.ndarray] = {
        g: np.average(tl.features[labels == g], axis=0, weights=tl.weights[labels == g])
        for g in groups
    }
    time = {g: float(tl.weights[labels == g].sum()) for g in groups}

    def redness(g: int) -> float:
        return float(centre[g][cols["glove_a"]] + centre[g][cols["shin_a"]])

    def blueness(g: int) -> float:
        return float(-(centre[g][cols["glove_b"]] + centre[g][cols["shin_b"]]))

    names: dict[int, str] = {}
    red = max(groups, key=lambda g: (redness(g), -g))
    names[red] = "red"
    rest = [g for g in groups if g != red]
    if rest:
        blue = max(rest, key=lambda g: (blueness(g), -g))
        names[blue] = "blue"
        rest = [g for g in rest if g != blue]
    if rest:
        names[max(rest, key=lambda g: (time[g], -g))] = "referee"
    return names


def assign_roles(
    tracks: pd.DataFrame, samples: pd.DataFrame, min_frames: int, max_dist: float
) -> pd.DataFrame:
    """One row per tracklet: track_id, role, n_frames, distance to its group's centre."""
    tl = summarise(tracks, corner_features(samples), min_frames)
    labels = merge_groups(tl, cluster(tl, k=N_GROUPS, max_dist=max_dist))
    names = name_groups(tl, labels)
    rows: list[dict[str, object]] = []
    for i, tid in enumerate(tl.ids):
        g = int(labels[i])
        own = np.asarray(tl.features[i], dtype=float)
        members = labels == g
        centre = np.asarray(tl.features[members].mean(axis=0), dtype=float) if g >= 0 else own
        rows.append(
            {
                "track_id": int(tid),
                "role": names.get(g, OTHER),
                "n_frames": int(tl.weights[i]),
                "distance": float(np.linalg.norm(own - centre)),
            }
        )
    all_ids: list[int] = tracks["track_id"].to_numpy(dtype=np.int64).tolist()
    missing = sorted(set(all_ids) - {int(x) for x in tl.ids})
    extra: list[dict[str, object]] = [
        {"track_id": t_, "role": OTHER, "n_frames": 0, "distance": np.nan} for t_ in missing
    ]
    rows.extend(extra)
    out = pd.DataFrame(rows, columns=["track_id", "role", "n_frames", "distance"])
    return out.sort_values("track_id").reset_index(drop=True)


def corner_fighters(fighter_a_id: str, fighter_b_id: str) -> Mapping[str, str]:
    """Which registry fighter is in which corner.

    IMMAF titles list the red corner first ("A vs. B": A is red). Checked on the
    pilot fights against the broadcast score bar, which marks the red and blue
    names with coloured edges. Other sources need their own check.
    """
    return {"red": fighter_a_id, "blue": fighter_b_id}


def live_shots(
    tracks: pd.DataFrame, shots: pd.DataFrame, sampled_frames: pd.DataFrame, min_frac: float
) -> pd.DataFrame:
    """A shot is live when most of its sampled frames show at least two people.

    `tracks` holds only boxes already big enough and on the canvas. Crowd shots,
    graphics and close-ups of one person fail this test.
    """
    people = tracks.groupby("frame")["track_id"].nunique()
    s = sampled_frames.assign(n_people=sampled_frames["frame"].map(people).fillna(0))
    frac = s.assign(two=s["n_people"] >= 2).groupby("shot_id")["two"].mean()
    out = shots[["shot_id", "start", "end"]].copy()
    out["two_people_frac"] = out["shot_id"].map(frac).fillna(0.0)
    out["live"] = out["two_people_frac"] >= min_frac
    return out
