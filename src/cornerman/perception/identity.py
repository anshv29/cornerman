"""Who is who: label each tracklet as fighter A, fighter B, the referee, or other.

A tracker gives short tracklets that break at every camera cut, occlusion and
clinch. Identity glues them back together across the whole fight:

1. Summarise each tracklet by its average colour features (appearance.py).
2. Cluster tracklets into three groups with a cannot-link rule: two tracklets
   seen in the same frame are two different people, so they may not share a
   group. This constraint is what makes plain colour clustering reliable
   enough; it rules out the easy mistake of merging both fighters when their
   shorts look alike under bad lighting.
3. Name the groups by behaviour, not colour: the two fighters are the pair
   that stay closest to each other; the third group is the referee.

Fighter A and B are only consistent within a fight here. Linking them to the
names in fights.csv (for example by corner colour) comes later.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROLES = ("fighter_a", "fighter_b", "referee")
OTHER = "other"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("app_")]


@dataclass(frozen=True)
class Tracklets:
    ids: np.ndarray
    features: np.ndarray
    """One L2-normalised mean feature row per tracklet."""
    weights: np.ndarray
    """Number of frames each tracklet covers."""
    conflicts: list[set[int]]
    """conflicts[i] = indices of tracklets that share a frame with tracklet i."""


def summarise(tracks: pd.DataFrame, min_frames: int) -> Tracklets:
    cols = feature_columns(tracks)
    counts = tracks.groupby("track_id").size()
    keep = counts[counts >= min_frames].index
    t = tracks[tracks["track_id"].isin(keep)]
    means = t.groupby("track_id")[cols].mean()
    feats = means.to_numpy(dtype=float)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.where(norms > 0, norms, 1.0)
    ids = means.index.to_numpy()
    pos = {tid: i for i, tid in enumerate(ids)}
    conflicts: list[set[int]] = [set() for _ in ids]
    for _, group in t.groupby("frame")["track_id"]:
        members = [pos[x] for x in group.unique()]
        for i in members:
            conflicts[i].update(m for m in members if m != i)
    weights = counts.loc[ids].to_numpy(dtype=float)
    return Tracklets(ids=ids, features=feats, weights=weights, conflicts=conflicts)


def _seeds(tl: Tracklets, k: int) -> list[int]:
    """Start from k heavy tracklets that were on screen together, so surely different people."""
    order = list(np.argsort(-tl.weights, kind="stable"))
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
    return [int(c) for c in chosen]


def cluster(tl: Tracklets, k: int, max_dist: float, n_iter: int = 20) -> np.ndarray:
    """Cannot-link k-means. Label per tracklet in 0..k-1, or -1 for 'other'."""
    n = len(tl.ids)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    centroids = tl.features[_seeds(tl, k)]
    labels = np.full(n, -1, dtype=np.int64)
    order = np.argsort(-tl.weights, kind="stable")
    for _ in range(n_iter):
        new = np.full(n, -1, dtype=np.int64)
        dist = np.linalg.norm(tl.features[:, None, :] - centroids[None], axis=2)
        for i in (int(x) for x in order):
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
                m = np.average(tl.features[members], axis=0, weights=tl.weights[members])
                centroids[c] = m / max(float(np.linalg.norm(m)), 1e-12)
    return labels


def _pair_distance(tracks: pd.DataFrame, a: int, b: int) -> float:
    """Median distance between two groups' box centres, in units of box height."""
    ta = tracks[tracks["group"] == a].groupby("frame")[["cx", "cy", "h"]].mean()
    tb = tracks[tracks["group"] == b].groupby("frame")[["cx", "cy", "h"]].mean()
    both = ta.join(tb, lsuffix="_a", rsuffix="_b", how="inner")
    if both.empty:
        return float("inf")
    d = np.hypot(both["cx_a"] - both["cx_b"], both["cy_a"] - both["cy_b"])
    return float(np.median(d / ((both["h_a"] + both["h_b"]) / 2)))


def name_groups(tracks: pd.DataFrame, groups: Sequence[int]) -> dict[int, str]:
    """Map group labels to roles. `tracks` needs frame, group, x1, y1, x2, y2."""
    t = tracks.assign(
        cx=(tracks["x1"] + tracks["x2"]) / 2,
        cy=(tracks["y1"] + tracks["y2"]) / 2,
        h=tracks["y2"] - tracks["y1"],
    )
    gs = list(groups)
    if len(gs) < 2:
        return {g: ROLES[0] for g in gs}
    pairs = [(a, b) for i, a in enumerate(gs) for b in gs[i + 1 :]]
    fa, fb = min(pairs, key=lambda p: (_pair_distance(t, *p), p))
    # Fighter A is whoever is further left the first time both are on screen.
    xa = t[t["group"] == fa].groupby("frame")["cx"].mean()
    xb = t[t["group"] == fb].groupby("frame")["cx"].mean()
    common = xa.index.intersection(xb.index)
    if len(common) and xa[common[0]] > xb[common[0]]:
        fa, fb = fb, fa
    names = {fa: "fighter_a", fb: "fighter_b"}
    for g in gs:
        names.setdefault(g, "referee")
    return names


def assign_roles(tracks: pd.DataFrame, min_frames: int, max_dist: float) -> pd.DataFrame:
    """One row per tracklet: track_id, role, n_frames, distance to its group's centre."""
    tl = summarise(tracks, min_frames)
    labels = cluster(tl, k=len(ROLES), max_dist=max_dist)
    by_track = dict(zip(tl.ids, labels, strict=True))
    groups = sorted({int(g) for g in labels if g >= 0})
    t = tracks.assign(group=tracks["track_id"].map(by_track).fillna(-1).astype(int))
    names = name_groups(t[t["group"] >= 0], groups)

    rows: list[dict[str, object]] = []
    for i, tid in enumerate(tl.ids):
        g = int(labels[i])
        members = labels == g
        own = np.asarray(tl.features[i], dtype=float)
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
    short = sorted(set(all_ids) - {int(x) for x in tl.ids})
    extra: list[dict[str, object]] = [
        {"track_id": t_, "role": OTHER, "n_frames": 0, "distance": np.nan} for t_ in short
    ]
    rows.extend(extra)
    return pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)


def live_shots(
    tracks: pd.DataFrame, shots: pd.DataFrame, sampled_frames: pd.DataFrame, min_frac: float
) -> pd.DataFrame:
    """A shot is live when most of its sampled frames show at least two people.

    `tracks` holds only boxes already big enough to be on the mat. Crowd shots,
    graphics and close-ups of one person fail this test.
    """
    people = tracks.groupby("frame")["track_id"].nunique()
    s = sampled_frames.assign(n_people=sampled_frames["frame"].map(people).fillna(0))
    frac = s.assign(two=s["n_people"] >= 2).groupby("shot_id")["two"].mean()
    out = shots[["shot_id", "start", "end"]].copy()
    out["two_people_frac"] = out["shot_id"].map(frac).fillna(0.0)
    out["live"] = out["two_people_frac"] >= min_frac
    return out
