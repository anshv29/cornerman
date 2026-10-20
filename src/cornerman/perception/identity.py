"""Who is who: label every tracked box as the red corner, the blue corner, the referee, or other.

Runs after pose and crop_features, without the video. It is the method fixed
and tested in identity round 3 (DECISIONS.md; experiments/identity_v3.py,
FROZEN). Inputs per fight: the tracks, and per saved person crop (about one
per tracked person per second) an appearance vector, kit colour counts and a
merged-box flag (crop_features.py).

1. A per-fight embedding. A linear projection of the appearance vectors (plus
   a kit hue histogram) is trained with a triplet loss: two crops of one
   tracklet are the same person, crops of two tracklets on screen at the same
   time are different people.
2. People. Tracklets with at least two usable crops (body found, not a merged
   box) are clustered with a cannot-link rule (two tracklets in the same frame
   can't share a group), and look-alike groups that never co-occur are merged.
3. The two fighters. A fighter-versus-official classifier, trained across
   fights on labels from behaviour (on the ground or kicking = fighter; standing
   still for a long time = not), scores each group. Fighter-like groups that are
   almost never on screen together are folded into one person (tracker ID
   switches split a fighter). The fighters are the pair of fighter-like people
   that do appear together and have the most screen time; red is the one with
   more red than blue kit pixels. Of the rest, the group most often on screen
   with the fighters is the referee.
4. Roles per crop, not per tracklet. When fighters tangle, the tracker swaps
   IDs, so one tracklet can hold both fighters. Each crop is scored against the
   red and blue people and the nearest other person, and a Viterbi pass along
   each tracklet pays SWITCH to change person: one odd crop doesn't flip, a
   real ID switch does. Every box then takes the role of the nearest crop in
   its tracklet.

A per-fight confidence score (0 to 1) from label-free checks flags fights where
identity is likely wrong, so later models can leave them out.

Corners link to names through `corner_fighters`: IMMAF titles list the red
corner first (checked against the broadcast score bar, DECISIONS.md).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ("red", "blue", "referee")
FIGHTERS = ROLES[:2]
OTHER = "other"
MERGE_DIST = 0.5
"""Groups whose unit-length centres are closer than this and never co-occur are one person."""


@dataclass(frozen=True)
class Tracklets:
    ids: np.ndarray
    features: np.ndarray
    weights: np.ndarray
    """Number of frames each tracklet covers."""
    conflicts: list[set[int]]
    """conflicts[i] = indices of tracklets that share a frame with tracklet i."""


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


@dataclass(frozen=True)
class IdentityParams:
    """The frozen settings of identity round 3. Changing any of them means re-testing."""

    k: int = 8
    """Groups for the cannot-link clustering (more than the people present, then merged)."""
    max_dist: float = 10.0
    merge_dist: float = MERGE_DIST
    fighter_prob: float = 0.5
    """Groups whose mean fighter score reaches this are fighter-like."""
    track_min_prob: float = 0.35
    """A tracklet scoring below this is taken out of a fighter group."""
    switch: float = 2.0
    """Viterbi cost of changing person along a tracklet (scores are 10 x cosine)."""
    hue_weight: float = 1.0
    steps: int = 400
    margin: float = 0.3
    flag_below: float = 0.5
    """Fights with a confidence below this are flagged (see config.identity_flag_below)."""


@dataclass(frozen=True)
class Probe:
    """Logistic regression: fighter score = sigmoid(features @ w + b)."""

    w: np.ndarray
    b: float

    def score(self, features: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(features @ self.w + self.b)))

    def save(self, path: Path) -> None:
        np.savez(path, w=self.w.astype(np.float64), b=np.array(self.b))

    @staticmethod
    def load(path: Path) -> "Probe":
        z = np.load(path)
        return Probe(w=np.asarray(z["w"], dtype=np.float64), b=float(z["b"]))


def behaviour_labels(tracks: pd.DataFrame, poses: pd.DataFrame, crops: pd.DataFrame) -> np.ndarray:
    """Per crop: 1 = fighter (torso near horizontal, or kicking), 0 = an official, -1 = unknown.

    No human labels and no skin colour. Referees and officials essentially never
    lie down or kick; tracklets that barely move for at least 30 frames are
    standing officials or photographers. Checked on samples: about 75 to 80% of
    fighter labels and 95% of official labels are right (DECISIONS.md).
    """
    t = tracks.assign(
        cx=(tracks["x1"] + tracks["x2"]) / 2,
        cy=(tracks["y1"] + tracks["y2"]) / 2,
        h=tracks["y2"] - tracks["y1"],
    ).sort_values(["track_id", "frame"])
    g = t.groupby("track_id")
    dt = g["t"].diff()
    step = pd.Series(np.hypot(g["cx"].diff(), g["cy"].diff()), index=t.index)
    t = t.assign(v=(step / t["h"] / dt).where(dt <= 0.3))
    med = t.groupby("track_id")["v"].median()
    speed = dict(zip(med.index.tolist(), med.to_numpy(dtype=float).tolist(), strict=True))
    life = t.groupby("track_id").size()
    w = poses.pivot_table(index=["frame", "track_id"], columns="kp", values=["y", "conf"])
    c, y = w["conf"], w["y"]
    kicking = ((c[15] > 0.5) & (c[14] > 0.5) & (y[15] < y[14])) | (
        (c[16] > 0.5) & (c[13] > 0.5) & (y[16] < y[13])
    )
    kick = {(int(f), int(k)) for f, k in w.index[kicking.to_numpy()]}
    out = np.full(len(crops), -1)
    rows = zip(crops["track_id"], crops["frame"], crops["angle"], strict=True)
    for i, (tid, fr, ang) in enumerate(rows):
        if (int(fr), int(tid)) in kick or (not np.isnan(ang) and ang > 60):
            out[i] = 1
        elif speed.get(int(tid), 1.0) < 0.03 and int(life.get(tid, 0)) >= 30:
            out[i] = 0
    return out


def train_probe(x: np.ndarray, y: np.ndarray, steps: int = 2000) -> Probe:
    """L2-regularised logistic regression by gradient descent, classes balanced."""
    yf = y.astype(float)
    pos_w = float((y == 0).sum()) / max(int((y == 1).sum()), 1)
    sw = np.where(y == 1, pos_w, 1.0)
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        gr = (p - yf) * sw
        w -= 0.5 * (x.T @ gr / len(y) + 1e-3 * w)
        b -= 0.5 * float(gr.mean())
    return Probe(w=w, b=b)


def cooccurrence(tracks: pd.DataFrame) -> dict[int, set[int]]:
    """track_id -> the tracklets seen in at least one frame with it."""
    out: dict[int, set[int]] = {}
    for _, g in tracks.groupby("frame")["track_id"]:
        ids = [int(i) for i in g.unique()]
        for a in ids:
            out.setdefault(a, set()).update(b for b in ids if b != a)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-6)


def fight_embedding(
    x: np.ndarray,
    track_id: np.ndarray,
    use: np.ndarray,
    cooc: dict[int, set[int]],
    params: IdentityParams,
) -> np.ndarray:
    """Unit-length 64-d embedding of every crop, from a projection trained on `use` crops."""
    import torch

    rng = np.random.default_rng(0)
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    by_track = {int(t): np.flatnonzero(use & (track_id == t)) for t in np.unique(track_id[use])}
    by_track = {t: ix for t, ix in by_track.items() if len(ix) >= 2}
    anchors = [t for t in by_track if any(n in by_track for n in cooc.get(t, ()))]
    proj = torch.nn.Linear(x.shape[1], 64, bias=False)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    xt = torch.tensor(x, dtype=torch.float32)
    for _ in range(params.steps):
        if not anchors:
            break
        a_ix: list[int] = []
        p_ix: list[int] = []
        n_ix: list[int] = []
        for t in rng.choice(anchors, 128):
            i, j = rng.choice(by_track[int(t)], 2, replace=False)
            negs = [n for n in cooc[int(t)] if n in by_track]
            a_ix.append(int(i))
            p_ix.append(int(j))
            n_ix.append(int(rng.choice(by_track[int(rng.choice(negs))])))
        f = torch.nn.functional.normalize(proj(xt), dim=1)
        loss = torch.nn.functional.triplet_margin_with_distance_loss(
            f[a_ix],
            f[p_ix],
            f[n_ix],
            distance_function=lambda u, v: 1 - (u * v).sum(1),
            margin=params.margin,
        )
        opt.zero_grad()
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        opt.step()  # pyright: ignore[reportUnknownMemberType]
    with torch.no_grad():
        return torch.nn.functional.normalize(proj(xt), dim=1).numpy()


@dataclass
class Assignment:
    crop_roles: pd.DataFrame
    """track_id, frame, role (red, blue, referee, other), gap (cosine gap red vs blue)."""
    stats: dict[str, float] = field(default_factory=lambda: {})


def assign(
    tracks: pd.DataFrame,
    crops: pd.DataFrame,
    features: np.ndarray,
    hue: np.ndarray,
    probe: Probe,
    params: IdentityParams,
) -> Assignment:
    """Roles per crop for one fight. `tracks` are the boxes in live shots.

    `crops` has crop_features.COLUMNS in the same order as `features` and `hue`.
    """
    tid = crops["track_id"].to_numpy(dtype=np.int64)
    usable = (crops["has_mask"] & ~crops["merged"]).to_numpy(dtype=bool)
    cooc = cooccurrence(tracks)
    emb = fight_embedding(
        np.hstack([features, params.hue_weight * np.sqrt(hue)]), tid, usable, cooc, params
    )
    prob = probe.score(features)

    # People: cannot-link clustering of tracklets with at least two usable crops.
    by_track = {int(t): np.flatnonzero(usable & (tid == t)) for t in np.unique(tid[usable])}
    by_track = {t: ix for t, ix in by_track.items() if len(ix) >= 2}
    ids = np.array(sorted(by_track), dtype=np.int64)
    feat = (
        np.stack([_unit(emb[by_track[int(t)]].mean(0)) for t in ids])
        if len(ids)
        else np.zeros((0, 64))
    )
    pos = {int(t): i for i, t in enumerate(ids)}
    conflicts = [{pos[n] for n in cooc.get(int(t), ()) if n in pos} for t in ids]
    counts = tracks.groupby("track_id").size()
    tl = Tracklets(
        ids=ids,
        features=feat,
        weights=counts.reindex(ids.tolist()).to_numpy(dtype=float),
        conflicts=conflicts,
    )
    labels = merge_groups(tl, cluster(tl, k=params.k, max_dist=params.max_dist), params.merge_dist)
    grp = {int(t): int(g) for t, g in zip(ids, labels, strict=True) if g >= 0}
    g_of = np.array([grp.get(int(t), -1) for t in tid])
    use = usable & (g_of >= 0)
    df = pd.DataFrame(
        {
            "group": g_of,
            "prob": prob,
            "red": crops["red"].to_numpy(),
            "blue": crops["blue"].to_numpy(),
            "n": crops["npx"].to_numpy(),
        }
    )[use]
    gs = df.groupby("group").agg(
        prob=("prob", "mean"), red=("red", "sum"), blue=("blue", "sum"), n=("n", "sum")
    )
    prob_of = dict(zip(gs.index.tolist(), gs["prob"].to_numpy(dtype=float).tolist(), strict=True))
    if len(gs) < 2:
        roles = pd.DataFrame({"track_id": tid, "frame": crops["frame"], "role": OTHER, "gap": 0.0})
        return Assignment(roles, {"failed": 1.0})
    tg = tracks.assign(group=tracks["track_id"].map(grp)).dropna(subset=["group"])
    frames_of: dict[int, set[int]] = {
        int(f["group"].iloc[0]): set(f["frame"].tolist()) for _, f in tg.groupby("group")
    }
    for g in gs.index:
        frames_of.setdefault(int(g), set())

    def overlap(x: int, y: int) -> float:
        """Share of the smaller group's frames in which both groups are on screen."""
        shared = len(frames_of[x] & frames_of[y])
        return shared / max(1, min(len(frames_of[x]), len(frames_of[y])))

    group_ix = df["group"].to_numpy()
    centre = {int(g): _unit(emb[use][group_ix == g].mean(0)) for g in gs.index}
    # Fold fighter-like groups that are almost never on screen together (one fighter
    # split by tracker ID switches); biggest first, each into the most similar person.
    fighterish = sorted(
        (int(g) for g in gs.index if prob_of[g] >= params.fighter_prob),
        key=lambda g: -len(frames_of[g]),
    )
    people: list[int] = []
    for g in fighterish:
        fits = [p for p in people if overlap(g, p) < 0.05]
        if not fits:
            people.append(g)
            continue
        p = max(fits, key=lambda q: float(centre[g] @ centre[q]))
        for t, gg in list(grp.items()):
            if gg == g:
                grp[t] = p
        gs.loc[p, ["red", "blue", "n"]] += gs.loc[g, ["red", "blue", "n"]]
        frames_of[p] |= frames_of[g]
        gs = gs.drop(index=g)
    size = {int(g): len(frames_of[int(g)]) for g in gs.index}
    pairs = [(x, y) for i, x in enumerate(people) for y in people[i + 1 :] if overlap(x, y) >= 0.1]
    stats: dict[str, float] = {}
    if pairs:
        ranked = sorted(pairs, key=lambda xy: -(size[xy[0]] + size[xy[1]]))
        a, b = ranked[0]
        runner_up = size[ranked[1][0]] + size[ranked[1][1]] if len(ranked) > 1 else 0
        stats["pair_margin"] = 1 - runner_up / max(1, size[a] + size[b])
    else:
        top = sorted((int(g) for g in gs.index), key=lambda g: -prob_of[g])
        if len(top) < 2:
            roles = pd.DataFrame(
                {"track_id": tid, "frame": crops["frame"], "role": OTHER, "gap": 0.0}
            )
            return Assignment(roles, {"failed": 1.0})
        a, b = top[0], top[1]
        stats["pair_margin"] = 0.0
    share_s = (gs["red"] - gs["blue"]) / gs["n"].clip(lower=1)
    share = dict(zip(gs.index.tolist(), share_s.to_numpy(dtype=float).tolist(), strict=True))
    red, blue = (a, b) if share[a] >= share[b] else (b, a)
    stats["colour_margin"] = abs(share[a] - share[b])
    stats["pair_overlap"] = overlap(a, b)
    stats["fighter_prob_min"] = min(prob_of[a], prob_of[b])
    names = {red: "red", blue: "blue"}
    rest = [int(g) for g in gs.index if g not in (a, b)]
    referee = None
    if rest:
        both = frames_of[a] | frames_of[b]
        referee = max(rest, key=lambda g: len(frames_of[g] & both))
        names[referee] = "referee"
    track_role = {t: names.get(g, OTHER) for t, g in grp.items()}
    tp = pd.DataFrame({"t": tid, "p": prob}).groupby("t")["p"].mean()
    track_prob = dict(zip(tp.index.tolist(), tp.to_numpy(dtype=float).tolist(), strict=True))
    for t in list(track_role):
        if track_role[t] in FIGHTERS and track_prob.get(t, 0.0) < params.track_min_prob:
            track_role[t] = OTHER

    # Roles per crop, smoothed along each tracklet.
    r_of = np.array([track_role.get(int(t), "") for t in tid])
    if not all((usable & (r_of == k)).any() for k in FIGHTERS):
        roles = pd.DataFrame({"track_id": tid, "frame": crops["frame"], "role": OTHER, "gap": 0.0})
        return Assignment(roles, {**stats, "failed": 1.0})
    centres = {k: _unit(emb[usable & (r_of == k)].mean(0)) for k in FIGHTERS}
    others = {g: _unit(emb[use][group_ix == g].mean(0)) for g in rest}
    roles = per_crop_roles(crops, emb, centres, others, referee, params.switch)
    return Assignment(roles, stats)


def per_crop_roles(
    crops: pd.DataFrame,
    emb: np.ndarray,
    centres: dict[str, np.ndarray],
    others: dict[int, np.ndarray],
    referee: int | None,
    switch: float,
) -> pd.DataFrame:
    """Red, blue or other per crop: Viterbi along each tracklet, paying `switch` per change.

    Scores are 10 x cosine similarity to the red person, the blue person and the
    nearest other person. Crops that end up "other" and are nearest the referee's
    group are named referee afterwards, which leaves red and blue unchanged.
    """
    other_ids = list(others)
    if other_ids:
        sim_o = np.stack([emb @ others[g] for g in other_ids], 1)
        nearest = np.array(other_ids)[sim_o.argmax(1)]
        best_o = sim_o.max(1)
    else:
        nearest = np.full(len(emb), -1)
        best_o = np.full(len(emb), -1.0)
    sims = np.stack([emb @ centres["red"], emb @ centres["blue"], best_o], 1)
    score = sims * 10
    names = np.array(["red", "blue", OTHER])
    out = np.empty(len(crops), dtype=object)
    order = crops.reset_index(drop=True).sort_values("frame")
    for _, g in order.groupby("track_id"):
        ix = g.index.to_numpy()
        e = score[ix]
        dp = e[0].copy()
        back = np.zeros((len(ix), 3), dtype=np.int64)
        for k in range(1, len(ix)):
            trans = dp[:, None] - switch * (1 - np.eye(3))
            back[k] = trans.argmax(0)
            dp = trans.max(0) + e[k]
        path = [int(dp.argmax())]
        for k in range(len(ix) - 1, 0, -1):
            path.append(int(back[k][path[-1]]))
        out[ix] = names[path[::-1]]
    if referee is not None:
        out[(out == OTHER) & (nearest == referee)] = "referee"
    return pd.DataFrame(
        {
            "track_id": crops["track_id"].to_numpy(dtype=np.int64),
            "frame": crops["frame"].to_numpy(dtype=np.int64),
            "role": out.astype(str),
            "gap": np.abs(sims[:, 0] - sims[:, 1]),
        }
    )


def box_roles(tracks: pd.DataFrame, crop_roles: pd.DataFrame, fill: bool = True) -> pd.DataFrame:
    """frame, track_id, role for every box: the role of the nearest crop of its tracklet.

    A tracklet with no crop at all is "other" (or left out, with fill=False).
    """
    t = tracks[["frame", "track_id"]].sort_values("frame")
    r = crop_roles[["frame", "track_id", "role"]].sort_values("frame")
    m = pd.merge_asof(t, r, on="frame", by="track_id", direction="nearest")
    m = m.assign(role=m["role"].fillna(OTHER)) if fill else m.dropna(subset=["role"])
    return m.sort_values(["frame", "track_id"]).reset_index(drop=True)


def tracklet_roles(boxes: pd.DataFrame) -> pd.DataFrame:
    """One row per tracklet: its most common role, frame count, and how mixed it is.

    `distance` is the share of the tracklet's boxes with a different role (0 = pure).
    The box-level table is the one to use; this is a summary.
    """
    counts = boxes.groupby(["track_id", "role"]).size().rename("n").reset_index()
    total = counts.groupby("track_id")["n"].transform("sum")
    top = counts.assign(total=total).sort_values(["track_id", "n"], ascending=[True, False])
    top = top.drop_duplicates("track_id")
    return pd.DataFrame(
        {
            "track_id": top["track_id"].to_numpy(dtype=np.int64),
            "role": top["role"].to_numpy(dtype=str),
            "n_frames": top["total"].to_numpy(dtype=np.int64),
            "distance": (1 - top["n"] / top["total"]).to_numpy(dtype=float),
        }
    ).reset_index(drop=True)


def label_free_checks(
    tracks: pd.DataFrame,
    boxes: pd.DataFrame,
    crop_roles: pd.DataFrame,
    fighting: set[int],
) -> dict[str, float]:
    """Checks that need no labels, over fighting frames (`tracks`: live-shot boxes).

    - both: share of fighting frames with both red and blue boxed;
    - dup: share of frames where one corner is on two boxes (two people, same corner);
    - closest: in frames with red, blue and someone else, how often red and blue are
      the closest pair (fighters fight each other; an official in a fighter's row breaks it);
    - rb_share: share of fighting-time crops given a corner;
    - gap: median cosine gap between red and blue over those crops;
    - switch_rate: role changes along tracklets per crop.
    """
    t = tracks[tracks["frame"].isin(fighting)].merge(boxes, on=["frame", "track_id"])
    t = t.assign(cx=(t["x1"] + t["x2"]) / 2)
    rb = t[t["role"].isin(FIGHTERS)]
    per = rb.groupby(["frame", "role"]).size().unstack(fill_value=0)
    per = per.reindex(columns=list(FIGHTERS), fill_value=0)
    stats = {
        "both": float(((per["red"] > 0) & (per["blue"] > 0)).sum()) / max(1, len(fighting)),
        "dup": float(((per["red"] > 1) | (per["blue"] > 1)).sum()) / max(1, len(per)),
    }
    hits = total = 0
    for _, g in t.groupby("frame"):
        present = set(g["role"])
        if len(g) < 3 or not set(FIGHTERS) <= present:
            continue
        r = float(g.loc[g["role"] == "red", "cx"].iloc[0])
        b = float(g.loc[g["role"] == "blue", "cx"].iloc[0])
        o = g.loc[~g["role"].isin(FIGHTERS), "cx"].to_numpy(dtype=float)
        total += 1
        if len(o):  # three boxes all given a corner count as a miss
            hits += abs(r - b) <= float(np.minimum(np.abs(o - r), np.abs(o - b)).min())
    stats["closest"] = hits / total if total else float("nan")
    in_fight = crop_roles[crop_roles["frame"].isin(fighting)]
    mine = in_fight[in_fight["role"].isin(FIGHTERS)]
    stats["rb_share"] = len(mine) / max(1, len(in_fight))
    stats["gap"] = float(mine["gap"].median()) if len(mine) else 0.0
    o = crop_roles.sort_values(["track_id", "frame"])
    same = o["track_id"].eq(o["track_id"].shift())
    stats["switch_rate"] = float((same & o["role"].ne(o["role"].shift())).sum()) / max(1, len(o))
    stats["fight_min"] = len(fighting) / 600.0
    return stats


def confidence(stats: Mapping[str, float]) -> float:
    """Identity confidence for a fight in [0, 1]: the mean of four checks, each scaled to [0, 1].

    The ranges come from what a clean fight should look like, not from a fit; only
    the flag threshold was chosen on the dev fights. On those 30 fights the score
    separates right from wrong with an AUC of 0.74 (DECISIONS.md).
    """

    def scaled(x: float, lo: float, hi: float) -> float:
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    closest = stats.get("closest", float("nan"))
    return float(
        np.mean(
            [
                1 - scaled(stats.get("dup", 1.0), 0.0, 0.1),
                scaled(stats.get("fighter_prob_min", 0.0), 0.6, 0.85),
                scaled(stats.get("rb_share", 0.0), 0.3, 0.7),
                scaled(0.3 if np.isnan(closest) else closest, 0.2, 0.6),
            ]
        )
    )


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
