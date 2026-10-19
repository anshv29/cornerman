"""Experiment: identity round 3, and the method that goes into the pipeline.

Follows experiments/identity_fighters.py (20 of 30 held-out fights right). Its
failures were mixing, not red/blue swaps: ground scrambles put both fighters in
one row, officials and post-fight shots got into fighter rows. Ideas tried here,
judged on the same 30 fights (DECISIONS.md has the numbers):

1. Fighting time only. Fighting time can't come from the stored
   fighting.parquet (it depends on the old identity), so a first pass uses
   everyone's motion and a second pass the chosen fighters' motion
   (`two_pass`). Using fighting time to choose montage and confidence crops
   helped; also restricting grouping to it (`fighting`) was a tie.
2. Standing, separate boxes only, with the rest filled in afterwards from the
   tracklets around them (`upright`, `separate`, `fill`). Worse: too many
   standing crops in exchanges overlap another box, and the fill-in step put
   one fighter into the other's row.
3. A referee detector from behaviour (`separations`, the person stepping in
   between two people who were in contact). The labels were only 35 to 50%
   referees on samples (coaches outside the fence, post-fight crowds), too
   noisy to train on. Not used.
4. What worked: roles per crop instead of per tracklet (`per_crop`). Tracker
   ID switches in tangles put both fighters in one tracklet, so a label per
   tracklet must be wrong for part of it.

FROZEN is the method fixed before the 24 test fights were looked at. A per-fight
confidence score from label-free checks flags fights below FLAG_BELOW.

    uv run python experiments/identity_v3.py refs OUT_DIR FIGHT_ID...   # referee label montage
    uv run python experiments/identity_v3.py run OUT_DIR TAG TRAIN_FILE TEST_FILE [OPTIONS_JSON]
    uv run python experiments/identity_v3.py test OUT_DIR TRAIN_FILE TEST_FILE   # frozen method
    uv run python experiments/identity_v3.py sheets OUT_DIR TAG ROLES_DIR FIGHT_ID...

Montages contain fighters: OUT_DIR must be outside the repo (e.g. the scratchpad).
Dev verdicts are in identity_v3_dev_verdicts.csv.
"""

import json
import sys
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from cornerman.perception.activity import fighting_frames

sys.path.insert(0, str(Path(__file__).parent))
from identity_fighters import CACHE, HUE_WEIGHT, labels, train_probe

from cornerman.perception.identity import Tracklets, cluster, merge_groups

UPRIGHT_MAX = 45.0
"""Torso angle from vertical (degrees) above which a crop counts as on the ground."""
OVERLAP_MAX = 0.3
"""Crops sharing more than this share of their box with another person's box are left out."""
SEP_BEFORE_S = 2.0
SEP_AFTER_S = 2.0
STRICT_SEP = False
"""True: also require contact before and distance after (fewer, not cleaner, labels)."""
SEP_APART = 0.6
"""After a separation the two people are at least this many box heights apart (centres)."""


@cache
def tracks_of(fid: str) -> pd.DataFrame:
    """Boxes in live shots, with centre, height and the largest overlap with another box."""
    d = Path(f"data/derived/{fid}/tracking")
    t = pd.read_parquet(
        d / "tracks.parquet",
        columns=["frame", "t", "shot_id", "track_id", "x1", "y1", "x2", "y2", "score"],
    )
    live = pd.read_parquet(d / "live.parquet")
    t = t[t.shot_id.isin(live.loc[live.live, "shot_id"])].reset_index(drop=True)
    t["cx"] = (t.x1 + t.x2) / 2
    t["h"] = t.y2 - t.y1
    area = (t.x2 - t.x1) * t.h
    ov = np.zeros(len(t))
    for _, g in t.groupby("frame", sort=False):
        b = g[["x1", "y1", "x2", "y2"]].to_numpy()
        if len(b) > 1:
            iw = np.clip(
                np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]),
                0,
                None,
            )
            ih = np.clip(
                np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]),
                0,
                None,
            )
            inter = iw * ih
            np.fill_diagonal(inter, 0)
            a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            ov[g.index.to_numpy()] = inter.max(1) / np.maximum(a, 1)
    t["overlap"] = ov
    t["area"] = area
    return t


FIGHTING_OVERRIDE: dict[str, pd.DataFrame] = {}
"""Second pass: fighting frames from the first pass's fighters replace the role-free ones."""


def fighting_free(fid: str) -> pd.DataFrame:
    if fid in FIGHTING_OVERRIDE:
        return FIGHTING_OVERRIDE[fid]
    return _fighting_free(fid)


@cache
def _fighting_free(fid: str) -> pd.DataFrame:
    """Fighting frames from everyone's motion, so no roles are needed."""
    d = Path(f"data/derived/{fid}/tracking")
    fr = pd.read_parquet(d / "frames.parquet")
    live = pd.read_parquet(d / "live.parquet")
    shots = set(live.loc[live.live, "shot_id"].tolist())
    b = tracks_of(fid).assign(role=lambda x: x.track_id)
    return fighting_frames(fr, b, shots, 10.0, 0.2, 8.0, 15.0)


def separations(fid: str) -> pd.DataFrame:
    """(frame, track_id) of people stepping in between two others who were in contact.

    At frame f, person R's centre is between A's and B's and R's box reaches into
    both. In the SEP_BEFORE_S before, A and B overlapped horizontally while R was
    not between them; in the SEP_AFTER_S after, A and B end up at least SEP_APART
    box heights apart. That is a referee breaking a clinch or stopping the fight.
    """
    t = tracks_of(fid)
    fight = fighting_free(fid)
    live = set(fight.loc[fight.fighting, "frame"].tolist())
    by_frame = {int(f): g for f, g in t.groupby("frame")}
    times = {int(f): float(g.t.iloc[0]) for f, g in by_frame.items()}
    per_track = {int(k): g.set_index("frame") for k, g in t.groupby("track_id")}
    out: set[tuple[int, int]] = set()
    for f, g in by_frame.items():
        if not 3 <= len(g) <= 4 or f not in live:
            continue
        rows = list(g.itertuples())
        for r in rows:
            others = [o for o in rows if o.track_id != r.track_id]
            for i, a in enumerate(others):
                for b in others[i + 1 :]:
                    lo, hi = sorted((a.cx, b.cx))
                    if not (lo < r.cx < hi and r.x1 < min(a.x2, b.x2) and r.x2 > max(a.x1, b.x1)):
                        continue
                    # Same depth as the pair: similar height and feet on a similar line
                    # (a coach outside the fence, close to the camera, is much bigger).
                    h = (a.h + b.h) / 2
                    if not 0.75 <= r.h / h <= 1.33 or abs(r.y2 - (a.y2 + b.y2) / 2) > 0.25 * h:
                        continue
                    if not STRICT_SEP or (
                        before_contact(per_track, a.track_id, b.track_id, r.track_id, times[f])
                        and after_apart(per_track, a.track_id, b.track_id, times[f])
                    ):
                        out.add((f, int(r.track_id)))
    return pd.DataFrame(sorted(out), columns=["frame", "track_id"])


def before_contact(per: dict[int, pd.DataFrame], a: int, b: int, r: int, t0: float) -> bool:
    pa, pb, pr = per[a], per[b], per[r]
    w = pa[(pa.t >= t0 - SEP_BEFORE_S) & (pa.t < t0)].join(
        pb[["x1", "x2", "cx"]], rsuffix="_b", how="inner"
    )
    if w.empty:
        return False
    w = w.join(pr[["cx"]], rsuffix="_r", how="left")
    touch = (w.x1 < w.x2_b) & (w.x2 > w.x1_b)
    lo, hi = np.minimum(w.cx, w.cx_b), np.maximum(w.cx, w.cx_b)
    outside = w.cx_r.isna() | (w.cx_r <= lo) | (w.cx_r >= hi)
    return bool((touch & outside).any())


def after_apart(per: dict[int, pd.DataFrame], a: int, b: int, t0: float) -> bool:
    pa, pb = per[a], per[b]
    w = pa[(pa.t > t0) & (pa.t <= t0 + SEP_AFTER_S)].join(
        pb[["cx", "h"]], rsuffix="_b", how="inner"
    )
    if w.empty:
        return False
    return bool(((w.cx - w.cx_b).abs() / np.maximum(w.h, w.h_b) >= SEP_APART).any())


def fighting_from_roles(fid: str, roles: pd.DataFrame) -> pd.DataFrame:
    """The pipeline's fighting filter, on the boxes of the fighters this pass found."""
    d = Path(f"data/derived/{fid}/tracking")
    fr = pd.read_parquet(d / "frames.parquet")
    live = pd.read_parquet(d / "live.parquet")
    shots = set(live.loc[live.live, "shot_id"].tolist())
    t = tracks_of(fid).sort_values("frame")
    if "frame" in roles:
        t = pd.merge_asof(
            t, roles.sort_values("frame"), on="frame", by="track_id", direction="nearest"
        )
    else:
        t = t.merge(roles, on="track_id")
    b = t[t.role.isin(["red", "blue"])].sort_values(
        ["frame", "role", "score"], ascending=[True, True, False]
    )
    b = b.drop_duplicates(["frame", "role"])
    return fighting_frames(fr, b, shots, 10.0, 0.2, 8.0, 15.0)


def crop_table(fid: str) -> pd.DataFrame:
    """Per cached crop: features index, track, frame, and the clean-crop tests."""
    z = np.load(CACHE / f"{fid}.npz")
    c = pd.DataFrame(
        {
            "i": np.arange(len(z["track_id"])),
            "track_id": z["track_id"].astype(int),
            "frame": z["frame"].astype(int),
            "angle": z["angle"],
            "merged": z["merged"],
            "has_mask": z["has_mask"],
        }
    )
    t = tracks_of(fid)[["frame", "track_id", "overlap"]]
    c = c.merge(t, on=["frame", "track_id"], how="left")
    fight = fighting_free(fid).set_index("frame")["fighting"]
    c["fighting"] = c.frame.map(fight).fillna(False).astype(bool)
    c["upright"] = c.angle < UPRIGHT_MAX
    c["separate"] = c.overlap.fillna(1.0) <= OVERLAP_MAX
    return c


def montage_rows(fid: str, picks: dict[str, pd.DataFrame], n: int = 12) -> np.ndarray:
    rows = []
    for name, df in picks.items():
        pick = df.sample(min(n, len(df)), random_state=0) if len(df) else df
        tiles = [
            cv2.resize(cv2.imread(f"data/crops/{fid}/{t:06d}_{f:07d}.jpg"), (60, 120))
            for t, f in zip(pick.track_id, pick.frame, strict=True)
        ]
        tiles += [np.zeros((120, 60, 3), np.uint8)] * (n - len(tiles))
        lab = np.zeros((120, 110, 3), np.uint8)
        cv2.putText(lab, fid[3:], (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(lab, name, (2, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.hstack([lab, *tiles]))
    return np.vstack(rows)


def save_sheets(sheets: list[np.ndarray], out: Path, tag: str, per_page: int = 4) -> None:
    for i in range(0, len(sheets), per_page):
        sep = np.full((6, sheets[0].shape[1], 3), 255, np.uint8)
        page = np.vstack([x for s_ in sheets[i : i + per_page] for x in (s_, sep)])
        cv2.imwrite(str(out / f"sheet_{tag}_{i // per_page:02d}.jpg"), page)


def crops_at(fid: str, keys: pd.DataFrame) -> pd.DataFrame:
    """The cached crops nearest in time to each (frame, track_id) of the same tracklet."""
    c = crop_table(fid)
    out = []
    for tid, g in keys.groupby("track_id"):
        mine = c[c.track_id == tid]
        if mine.empty:
            continue
        for f in g.frame:
            j = (mine.frame - f).abs()
            if j.min() <= 15:  # within 1.5 s at 10 fps
                out.append(mine.loc[j.idxmin()])
    return pd.DataFrame(out).drop_duplicates("i") if out else c.iloc[:0]


FROZEN: dict[str, object] = {
    "fighting": False,
    "upright": False,
    "separate": False,
    "fill": False,
    "per_crop": True,
    "two_pass": True,
    "switch": 2.0,
    "prob_weight": 0.0,
}
"""The method fixed on 2026-10-18 before the 24 test fights were looked at (variant b)."""
FLAG_BELOW = 0.4
"""Fights whose confidence is below this are flagged (set on the 30 dev fights)."""
FIGHTER_PROB = 0.5
TRACK_MIN_PROB = 0.35
CONT_GAP_S = 3.0
"""A tracklet continues another if it starts within this long after the other ends, nearby."""
CONT_DIST = 0.6
"""... and its first box centre is within this many box heights of the other's last."""


def embed(z: dict[str, np.ndarray], tracks: pd.DataFrame, use: np.ndarray) -> np.ndarray:
    """Per-fight projection of every crop, trained with a triplet loss on the `use` crops.

    Same as identity_fighters.groups: crops of one tracklet are one person,
    tracklets on screen together are two people.
    """
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    tid = z["track_id"]
    cooc = cooccurrence(tracks)
    by_track = {int(t): np.flatnonzero(use & (tid == t)) for t in np.unique(tid[use])}
    by_track = {t: ix for t, ix in by_track.items() if len(ix) >= 2}
    anchors = [t for t in by_track if any(n in by_track for n in cooc.get(t, ()))]
    x_np = np.hstack([z["feat"], HUE_WEIGHT * np.sqrt(z["hue"])])
    proj = torch.nn.Linear(x_np.shape[1], 64, bias=False)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    x = torch.tensor(x_np, dtype=torch.float32)
    for _ in range(400):
        if not anchors:
            break
        a_ix, p_ix, n_ix = [], [], []
        for t in rng.choice(anchors, 128):
            i, j = rng.choice(by_track[int(t)], 2, replace=False)
            negs = [n for n in cooc[int(t)] if n in by_track]
            a_ix.append(i)
            p_ix.append(j)
            n_ix.append(rng.choice(by_track[int(rng.choice(negs))]))
        f = torch.nn.functional.normalize(proj(x), dim=1)
        loss = torch.nn.functional.triplet_margin_with_distance_loss(
            f[a_ix], f[p_ix], f[n_ix], distance_function=lambda u, v: 1 - (u * v).sum(1), margin=0.3
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.nn.functional.normalize(proj(x), dim=1).numpy()


@cache
def _cooc(fid: str) -> dict[int, set[int]]:
    return cooccurrence(tracks_of(fid))


def cooccurrence(tracks: pd.DataFrame) -> dict[int, set[int]]:
    cooc: dict[int, set[int]] = {}
    for _, g in tracks.groupby("frame")["track_id"]:
        ids = [int(i) for i in g.unique()]
        for a in ids:
            cooc.setdefault(a, set()).update(b for b in ids if b != a)
    return cooc


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-6)


def assign_v3(
    fid: str, w: np.ndarray, b: float, opts: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Track -> role for one fight, plus the numbers the confidence score is built from."""
    tracks = tracks_of(fid)
    z = dict(np.load(CACHE / f"{fid}.npz"))
    c = crop_table(fid)
    clean = (c.has_mask & ~c.merged).to_numpy().copy()
    if opts.get("upright", True):
        clean &= c.upright.to_numpy()
    if opts.get("separate", True):
        clean &= c.separate.to_numpy()
    if opts.get("fighting", True):
        clean &= c.fighting.to_numpy()
    emb = embed(z, tracks, clean)
    tid = z["track_id"].astype(int)
    prob = 1 / (1 + np.exp(-(z["feat"] @ w + b)))

    # Group tracklets with at least 2 clean crops (cannot-link clustering, as before).
    by_track = {int(t): np.flatnonzero(clean & (tid == t)) for t in np.unique(tid[clean])}
    by_track = {t: ix for t, ix in by_track.items() if len(ix) >= 2}
    ids = np.array(sorted(by_track))
    feat = (
        np.stack([unit(emb[by_track[t]].mean(0)) for t in ids]) if len(ids) else np.zeros((0, 64))
    )
    cooc = _cooc(fid)
    pos = {int(t): i for i, t in enumerate(ids)}
    conf = [{pos[n] for n in cooc.get(int(t), ()) if n in pos} for t in ids]
    counts = tracks.groupby("track_id").size()
    tl = Tracklets(
        ids=ids, features=feat, weights=counts.reindex(ids).to_numpy(float), conflicts=conf
    )
    lab = merge_groups(tl, cluster(tl, k=8, max_dist=10.0), max_dist=0.5)
    grp = {int(t): int(g) for t, g in zip(ids, lab, strict=True) if g >= 0}

    g_of = np.array([grp.get(int(t), -1) for t in tid])
    use = clean & (g_of >= 0)
    df = pd.DataFrame(
        {"group": g_of, "prob": prob, "red": z["red"], "blue": z["blue"], "n": z["npx"]}
    )[use]
    gs = df.groupby("group").agg(
        prob=("prob", "mean"), red=("red", "sum"), blue=("blue", "sum"), n=("n", "sum")
    )
    fight_frames = set(fighting_free(fid).query("fighting").frame.tolist())
    tf = tracks[tracks.frame.isin(fight_frames)] if opts.get("fighting", True) else tracks
    frames_of = {
        int(g): set(f.frame)
        for g, f in tf.assign(group=tf.track_id.map(grp)).dropna(subset=["group"]).groupby("group")
    }
    for g in gs.index:
        frames_of.setdefault(int(g), set())

    def overlap(x: int, y: int) -> float:
        shared = len(frames_of[int(x)] & frames_of[int(y)])
        return shared / max(1, min(len(frames_of[int(x)]), len(frames_of[int(y)])))

    centre = {int(g): unit(emb[use][df.group.to_numpy() == g].mean(0)) for g in gs.index}
    fighterish = sorted(
        [g for g in gs.index if gs.loc[g, "prob"] >= FIGHTER_PROB],
        key=lambda g: -len(frames_of[int(g)]),
    )
    people: list[int] = []
    for g in fighterish:
        fits = [p for p in people if overlap(g, p) < 0.05]
        if fits:
            p = max(fits, key=lambda q: float(centre[int(g)] @ centre[int(q)]))
            for t_, gg in list(grp.items()):
                if gg == g:
                    grp[t_] = int(p)
            gs.loc[p, ["red", "blue", "n"]] += gs.loc[g, ["red", "blue", "n"]]
            frames_of[int(p)] |= frames_of[int(g)]
            gs = gs.drop(index=g)
        else:
            people.append(g)
    size = {int(g): len(frames_of[int(g)]) for g in gs.index}
    pairs = [(x, y) for i, x in enumerate(people) for y in people[i + 1 :] if overlap(x, y) >= 0.1]
    stats: dict[str, float] = {}
    if pairs:
        ranked = sorted(pairs, key=lambda xy: -(size[xy[0]] + size[xy[1]]))
        a, bb = ranked[0]
        stats["pair_margin"] = (
            1.0
            if len(ranked) == 1
            else 1 - (size[ranked[1][0]] + size[ranked[1][1]]) / (size[a] + size[bb])
        )
    else:
        top = sorted(gs.index, key=lambda g: -gs.loc[g, "prob"])
        if len(top) < 2:
            return pd.DataFrame({"track_id": sorted(set(tid)), "role": "other"}), {"failed": 1.0}
        a, bb = top[0], top[1]
        stats["pair_margin"] = 0.0
    shares = (gs.red - gs.blue) / gs.n.clip(lower=1)
    red, blue = (a, bb) if shares[a] >= shares[bb] else (bb, a)
    stats["colour_margin"] = float(abs(shares[a] - shares[bb]))
    stats["pair_overlap"] = overlap(a, bb)
    stats["fighter_prob_min"] = float(min(gs.loc[a, "prob"], gs.loc[bb, "prob"]))
    names = {int(red): "red", int(blue): "blue"}
    rest = [g for g in gs.index if g not in (a, bb)]
    if rest:
        both = frames_of[int(a)] | frames_of[int(bb)]
        names[int(max(rest, key=lambda g: len(frames_of[int(g)] & both)))] = "referee"
    role = {t: names.get(g, "other") for t, g in grp.items()}
    track_prob = pd.Series(prob).groupby(tid).mean()
    for t in list(role):
        if (
            role[t] in ("red", "blue")
            and track_prob.get(t, 0) < TRACK_MIN_PROB
            and len(by_track.get(t, [])) > 0
        ):
            role[t] = "other"

    # Fill in tracklets with no clean crops (ground, merged, tangled) from their surroundings.
    if opts.get("fill", True):
        fill_in(fid, tracks, role, emb, tid, prob, centre_of_roles(emb, tid, clean, role), stats)
    all_ids = sorted({int(t) for t in tracks.track_id})
    out = pd.DataFrame({"track_id": all_ids, "role": [role.get(t, "other") for t in all_ids]})
    if opts.get("per_crop", False):
        others = [
            unit(emb[use][df.group.to_numpy() == g].mean(0)) for g in gs.index if g not in (a, bb)
        ]
        out = per_crop(
            z,
            c,
            emb,
            centre_of_roles(emb, tid, clean, role),
            others,
            float(opts.get("switch", 2.0)),
            prob,
            float(opts.get("prob_weight", 0.0)),
        )
        stats.update(label_free(fid, tracks, out, fight_frames))
        rb = out.merge(c[["track_id", "frame", "fighting"]], on=["track_id", "frame"])
        rb = rb[rb.fighting & rb.role.isin(["red", "blue"])]
        stats["gap"] = float(rb.gap.median()) if len(rb) else 0.0
        o = out.sort_values(["track_id", "frame"])
        sw = (o.role != o.groupby("track_id").role.shift()) & o.track_id.eq(o.track_id.shift())
        stats["switch_rate"] = float(sw.sum() / max(1, len(o)))
        stats["rb_share"] = float(len(rb) / max(1, int(c.fighting.sum())))
        return out[["track_id", "frame", "role"]], stats
    stats.update(label_free(fid, tracks, out, fight_frames))
    return out, stats


def per_crop(
    z: dict[str, np.ndarray],
    c: pd.DataFrame,
    emb: np.ndarray,
    centres: dict[str, np.ndarray],
    others: list[np.ndarray],
    switch: float,
    prob: np.ndarray,
    prob_weight: float,
) -> pd.DataFrame:
    """Role per crop: nearest of red, blue or anyone else, smoothed along each tracklet.

    Tracker ID switches in tangles put two people in one tracklet, so a label per
    tracklet must be wrong for part of it. Here each crop scores against the red
    and blue centres and the nearest other person; a Viterbi pass along the
    tracklet's crops (in time order) pays `switch` for changing person, so one
    odd-looking crop doesn't flip, but a real ID switch does.
    """
    if len(centres) < 2:
        return pd.DataFrame({"track_id": c.track_id, "frame": c.frame, "role": "other"})
    sims = np.stack(
        [
            emb @ centres["red"],
            emb @ centres["blue"],
            np.max(np.stack([emb @ o for o in others]), 0) if others else np.full(len(emb), -1.0),
        ],
        1,
    )
    names = np.array(["red", "blue", "other"])
    score = sims * 10  # cosine to a log-score scale
    # The fighter classifier's vote: a crop that doesn't look like a fighter leans to "other".
    p = np.clip(prob, 0.02, 0.98)
    score[:, :2] += prob_weight * np.log(p)[:, None]
    score[:, 2] += prob_weight * np.log(1 - p)
    out = np.empty(len(c), dtype=object)
    for _, g in c.sort_values("frame").groupby("track_id"):
        ix = g.i.to_numpy()
        e = score[ix]
        n = len(ix)
        dp = e[0].copy()
        back = np.zeros((n, 3), int)
        for k in range(1, n):
            trans = dp[:, None] - switch * (1 - np.eye(3))
            back[k] = trans.argmax(0)
            dp = trans.max(0) + e[k]
        path = [int(dp.argmax())]
        for k in range(n - 1, 0, -1):
            path.append(int(back[k][path[-1]]))
        out[g.index.to_numpy()] = names[path[::-1]]
    res = pd.DataFrame({"track_id": c.track_id, "frame": c.frame, "role": out})
    # How decisive the red/blue choice is per crop (cosine gap), kept for the confidence score.
    res["gap"] = np.abs(sims[c.i.to_numpy(), 0] - sims[c.i.to_numpy(), 1])
    return res


def centre_of_roles(
    emb: np.ndarray, tid: np.ndarray, clean: np.ndarray, role: dict[int, str]
) -> dict[str, np.ndarray]:
    r = np.array([role.get(int(t), "") for t in tid])
    return {
        k: unit(emb[clean & (r == k)].mean(0)) for k in ("red", "blue") if (clean & (r == k)).any()
    }


def fill_in(
    fid: str,
    tracks: pd.DataFrame,
    role: dict[int, str],
    emb: np.ndarray,
    tid: np.ndarray,
    prob: np.ndarray,
    centres: dict[str, np.ndarray],
    stats: dict[str, float],
) -> None:
    """Give red or blue to fighter-looking tracklets that had no clean crops.

    Evidence, per candidate role: the tracklet continues one that just ended
    nearby in the same shot, or is continued by one that starts nearby (strong);
    it is on screen with a tracklet that already has the role (rules it out);
    and its crops look like the role (weak). Biggest tracklets first, so a pair
    of tangled tracklets on the ground ends up as one red and one blue.
    """
    if len(centres) < 2:
        return
    spans = tracks.groupby("track_id").agg(
        shot=("shot_id", "first"), t0=("t", "min"), t1=("t", "max"), n=("t", "size")
    )
    first = tracks.sort_values("t").drop_duplicates("track_id").set_index("track_id")
    last = tracks.sort_values("t").drop_duplicates("track_id", keep="last").set_index("track_id")
    cooc = _cooc(fid)
    todo = [
        int(t)
        for t in spans.sort_values("n", ascending=False).index
        if int(t) not in role and spans.n[t] >= 5
    ]
    filled = 0
    for t in todo:
        mine = tid == t
        p = float(prob[mine].mean()) if mine.any() else 0.0
        score = {"red": 0.0, "blue": 0.0}
        sh, t0, t1 = spans.loc[t, ["shot", "t0", "t1"]]
        same = spans[(spans.shot == sh) & (spans.index != t)]
        prev = same[(same.t1 <= t0) & (same.t1 >= t0 - CONT_GAP_S)]
        nxt = same[(same.t0 >= t1) & (same.t0 <= t1 + CONT_GAP_S)]
        f = first.loc[t]
        l_ = last.loc[t]
        for o in prev.index:
            r = role.get(int(o))
            lo = last.loc[o]
            if r in score and np.hypot(lo.cx - f.cx, lo.y2 - f.y2) / max(lo.h, f.h) < CONT_DIST:
                score[r] += 1.0
        for o in nxt.index:
            r = role.get(int(o))
            fo = first.loc[o]
            if r in score and np.hypot(fo.cx - l_.cx, fo.y2 - l_.y2) / max(fo.h, l_.h) < CONT_DIST:
                score[r] += 1.0
        continued = any(v > 0 for v in score.values())
        if not continued and p < FIGHTER_PROB:
            continue
        if mine.any():
            m = unit(emb[mine].mean(0))
            for k in score:
                score[k] += 0.5 * float(m @ centres[k])
        for o in cooc.get(t, ()):
            if role.get(int(o)) in score:
                score[role[int(o)]] = -np.inf
        best = max(score, key=lambda k: score[k])
        if np.isfinite(score[best]):
            role[t] = best
            filled += 1
    stats["filled"] = float(filled)


def label_free(
    fid: str, tracks: pd.DataFrame, roles: pd.DataFrame, fight_frames: set[int]
) -> dict[str, float]:
    """Checks that need no labels, over fighting frames.

    both: share of fighting frames where red and blue are both boxed.
    dup: share of frames where one corner has two boxes (two people given the same corner).
    closest: in frames with red, blue and someone else, how often red and blue are the
      closest pair (fighters fight each other; an official in a fighter's row breaks this).
    """
    if "frame" in roles:
        # Per-crop roles: carry each crop's role to the boxes around it in its tracklet.
        t = tracks[tracks.frame.isin(fight_frames)].sort_values("frame")
        r = roles.sort_values("frame")
        t = pd.merge_asof(t, r, on="frame", by="track_id", direction="nearest").dropna(
            subset=["role"]
        )
    else:
        t = tracks[tracks.frame.isin(fight_frames)].merge(roles, on="track_id")
    rb = t[t.role.isin(["red", "blue"])]
    per = (
        rb.groupby(["frame", "role"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["red", "blue"], fill_value=0)
    )
    both = float(((per.red > 0) & (per.blue > 0)).sum() / max(1, len(fight_frames)))
    dup = float(((per.red > 1) | (per.blue > 1)).sum() / max(1, len(per)))
    hits = tot = 0
    for _, g in t.groupby("frame"):
        if len(g) < 3 or not {"red", "blue"} <= set(g.role):
            continue
        r = g[g.role == "red"].iloc[0]
        bl = g[g.role == "blue"].iloc[0]
        d_rb = abs(r.cx - bl.cx)
        others = g[~g.role.isin(["red", "blue"])]
        d_o = np.minimum((others.cx - r.cx).abs(), (others.cx - bl.cx).abs()).min()
        tot += 1
        hits += d_rb <= d_o
    return {
        "both": both,
        "dup": dup,
        "closest": hits / tot if tot else np.nan,
        "fight_min": len(fight_frames) / 600,
    }


def keys(roles: pd.DataFrame) -> list[str]:
    return ["track_id", "frame"] if "frame" in roles else ["track_id"]


def purity(fid: str, roles: pd.DataFrame) -> float:
    """Proxy check: share of coloured red/blue crops whose own pixels agree with their corner.

    Only crops with at least 30 red-or-blue pixels count, so it mostly speaks for
    colourful kits; it ranks variants, it doesn't replace judging montages.
    """
    z = np.load(CACHE / f"{fid}.npz")
    c = crop_table(fid).assign(red=z["red"], blue=z["blue"]).merge(roles, on=keys(roles))
    c = c[c.fighting & ~c.merged & c.role.isin(["red", "blue"]) & (c.red + c.blue >= 30)]
    if c.empty:
        return np.nan
    agree = np.where(c.role == "red", c.red > c.blue, c.blue > c.red)
    return float(agree.mean())


def confidence(stats: dict[str, float]) -> float:
    """Per-fight identity confidence in [0, 1]: the mean of four label-free checks, each in [0, 1].

    The ranges were set by reasoning about what a clean fight looks like, not fitted;
    only FLAG_BELOW was chosen on the dev fights.
    - dup: share of frames where one corner is on two boxes at once (0 good, 0.1 or more bad);
    - fighter_prob_min: the fighter classifier's mean score of the less fighter-like
      of the two chosen people (0.6 bad, 0.85 good);
    - rb_share: share of fighting-time crops given a corner (0.3 bad, 0.7 good);
    - closest: in frames with a third person, how often red and blue are the closest pair
      (0.2 bad, 0.6 good).
    """

    def sc(x: float, lo: float, hi: float) -> float:
        return float(np.clip((x - lo) / (hi - lo), 0, 1))

    closest = stats.get("closest", np.nan)
    return float(
        np.mean(
            [
                1 - sc(stats.get("dup", 1.0), 0, 0.1),
                sc(stats.get("fighter_prob_min", 0.0), 0.6, 0.85),
                sc(stats.get("rb_share", 0.0), 0.3, 0.7),
                sc(0.3 if np.isnan(closest) else closest, 0.2, 0.6),
            ]
        )
    )


def run(out: Path, tag: str, train: list[str], test: list[str], opts: dict[str, object]) -> None:
    w, b = train_probe(train)
    sheets, rows = [], []
    for fid in test:
        FIGHTING_OVERRIDE.pop(fid, None)
        roles, stats = assign_v3(fid, w, b, opts)
        if opts.get("two_pass", False):
            FIGHTING_OVERRIDE[fid] = fighting_from_roles(fid, roles)
            roles, stats = assign_v3(fid, w, b, opts)
        roles.to_parquet(out / f"roles_{tag}_{fid}.parquet")
        stats["purity"] = purity(fid, roles)
        stats["confidence"] = confidence(stats)
        stats["flagged"] = float(stats["confidence"] < FLAG_BELOW)
        rows.append({"fight_id": fid, **stats})
        c = crop_table(fid).merge(roles, on=keys(roles))
        c = c[c.fighting & ~c.merged]
        sheets.append(montage_rows(fid, {"red": c[c.role == "red"], "blue": c[c.role == "blue"]}))
        print(fid, {k: round(v, 3) for k, v in stats.items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / f"stats_{tag}.csv", index=False)
    if not opts.get("no_sheets"):
        save_sheets(sheets, out, tag)


if __name__ == "__main__":
    cmd, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    if cmd == "refs":
        sheets, stats = [], {}
        for fid in sys.argv[3:]:
            sep = separations(fid)
            c = crops_at(fid, sep)
            lab = labels(fid, {k: v for k, v in np.load(CACHE / f"{fid}.npz").items()})
            stats[fid] = {
                "events": len(sep),
                "tracks": int(sep.track_id.nunique()),
                "crops": len(c),
            }
            fighters = crop_table(fid)[lab == 1]
            sheets.append(montage_rows(fid, {"sep": c, "fighter": fighters}))
        save_sheets(sheets, out_dir, "refs")
        print(json.dumps(stats, indent=1))
    elif cmd == "sheets":
        # Montages from saved roles: sheets OUT_DIR TAG ROLES_DIR FIGHT_ID...
        tag, roles_dir = sys.argv[3], Path(sys.argv[4])
        sheets = []
        for fid in sys.argv[5:]:
            roles = pd.read_parquet(roles_dir / f"roles_{tag}_{fid}.parquet")
            FIGHTING_OVERRIDE[fid] = fighting_from_roles(fid, roles)
            c = crop_table(fid).merge(roles, on=keys(roles))
            c = c[c.fighting & ~c.merged]
            sheets.append(
                montage_rows(fid, {"red": c[c.role == "red"], "blue": c[c.role == "blue"]})
            )
        save_sheets(sheets, out_dir, tag)
    elif cmd == "run":
        tag = sys.argv[3]
        train_ids = Path(sys.argv[4]).read_text().split()
        test_ids = Path(sys.argv[5]).read_text().split()
        options = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
        run(out_dir, tag, train_ids, test_ids, options)
    elif cmd == "test":
        # The frozen method: test OUT_DIR TRAIN_FILE TEST_FILE
        train_ids = Path(sys.argv[3]).read_text().split()
        test_ids = Path(sys.argv[4]).read_text().split()
        run(out_dir, "test", train_ids, test_ids, FROZEN)
