"""Experiment: fighter-versus-everyone-else classifier for identity, across fights.

Follows experiments/identity_embedding.py, which separated people well but often
picked the wrong two groups as the fighters. Here a classifier trained across
fights decides which groups are fighters. Runs on the laptop from local crops
and poses; no video, no Kaggle.

    uv run python experiments/identity_fighters.py features FIGHT_ID...   # cache per-crop features
    uv run python experiments/identity_fighters.py run OUT_DIR             # train, assign, montages

Labels come from behaviour, not from people (no human labels, no skin colour):
- fighter: torso close to horizontal (on the canvas); referees and officials
  essentially never lie down;
- not a fighter: tracklets that barely move for their whole life (standing
  officials and photographers).
The classifier is a logistic regression on body-masked DINOv2 features, trained
on some fights and applied to others. Per fight, crops are grouped per person
with the per-fight embedding from identity_embedding.py; the two large groups
with the highest mean fighter score that appear together are the fighters;
merged clinch boxes add evidence (groups whose crops look like the fight's
merged crops); red and blue are decided by pooled red and blue pixel share.

Features and montages contain fighters' images: OUT_DIR must be outside the
repo (e.g. the scratchpad). Feature caches go to data/identity_cache/ (gitignored).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

from cornerman.perception.identity import Tracklets, cluster, merge_groups

CACHE = Path("data/identity_cache")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
MERGED_RATIO = 1.6
DEBUG = False
HUE_WEIGHT = 1.0
"""Weight of the kit hue histogram next to the DINOv2 feature in the per-fight embedding."""
FIGHTER_PROB = 0.5
TRACK_MIN_PROB = 0.35
"""Tracklets below this fighter probability are taken out of the red and blue groups."""
"""Groups whose mean fighter probability reaches this count as fighter-like."""


def load(fid: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(f"data/derived/{fid}")
    tracks = pd.read_parquet(d / "tracking/tracks.parquet")
    poses = pd.read_parquet(d / "pose/poses.parquet")
    files = sorted(Path(f"data/crops/{fid}").glob("*.jpg"))
    meta = pd.DataFrame(
        [(int(c.name[:6]), int(c.name[7:14]), str(c)) for c in files],
        columns=["track_id", "frame", "path"],
    )
    meta = meta.merge(
        tracks[["frame", "track_id", "t", "x1", "y1", "x2", "y2"]], on=["frame", "track_id"]
    )
    wh = (meta.x2 - meta.x1) / (meta.y2 - meta.y1)
    meta["merged"] = wh > MERGED_RATIO * wh.median()
    return tracks, poses, meta


def pose_lookup(poses: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    """(frame, track) -> (17, 3) array of x, y, conf."""
    p = poses.sort_values(["frame", "track_id", "kp"])
    arr = p[["x", "y", "conf"]].to_numpy().reshape(-1, 17, 3)
    keys = p[["frame", "track_id"]].to_numpy()[::17]
    return {(int(f), int(t)): a for (f, t), a in zip(keys, arr, strict=True)}


def body_mask(kp: np.ndarray | None, row: pd.Series, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    if kp is None:
        return m
    s = shape[0] / (row.y2 - row.y1)
    xy = ((kp[:, :2] - [row.x1, row.y1]) * s).astype(np.int32)
    c = kp[:, 2]
    if min(c[[5, 6, 11, 12]]) > 0.3:
        cv2.fillPoly(m, [xy[[5, 6, 12, 11]]], 1)
    for a, b in LIMBS:
        if min(c[a], c[b]) > 0.3:
            cv2.line(m, tuple(xy[a]), tuple(xy[b]), 1, max(2, int(0.08 * shape[0])))
    return m


def torso_angle(kp: np.ndarray | None) -> float:
    """Degrees between the torso (hip midpoint to shoulder midpoint) and vertical."""
    if kp is None or min(kp[[5, 6, 11, 12], 2]) <= 0.3:
        return np.nan
    v = (kp[5, :2] + kp[6, :2]) / 2 - (kp[11, :2] + kp[12, :2]) / 2
    return float(np.degrees(np.arctan2(abs(v[0]), -v[1])))


def colour_counts(img: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """Red and blue kit pixels in the body mask (dark red allowed; light canvas blue not)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[mask > 0]
    if len(hsv) < 20:
        return 0, 0, 0
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1], hsv[:, 2]
    red = (s >= 70) & (v >= 35) & ((h <= 6) | (h >= 170))
    blue = (s >= 120) & (v >= 40) & (v <= 170) & (h >= 100) & (h <= 140)
    return int(red.sum()), int(blue.sum()), len(hsv)


def hue_hist(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """18-bin hue histogram of strongly coloured body pixels, as a share of all body pixels.

    Light canvas blue (bright, less saturated) is left out, like in colour_counts.
    """
    out = np.zeros(18, np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[mask > 0]
    if len(hsv) < 20:
        return out
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1], hsv[:, 2]
    strong = (s >= 70) & (v >= 35) & ~((v > 170) & (h >= 90) & (h <= 130))
    out += np.bincount(h[strong] * 18 // 180, minlength=18)[:18] / len(hsv)
    return out


def features(fid: str, model: AutoModel) -> None:
    """Per crop: body-masked DINOv2 feature, torso angle, kit colour counts. Cached."""
    out = CACHE / f"{fid}.npz"
    if out.exists():
        return
    _, poses, meta = load(fid)
    pk = pose_lookup(poses)
    feats, keep, angle, red, blue, npx, hues = [], [], [], [], [], [], []
    batch_x, batch_w = [], []

    def flush() -> None:
        if not batch_x:
            return
        x = torch.tensor(np.stack(batch_x), dtype=torch.float32, device=DEV)
        with torch.no_grad():
            patches = model(pixel_values=x).last_hidden_state[:, 1:, :].cpu().numpy()
        w = np.stack(batch_w)
        feats.append((patches * w[:, :, None]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-6))
        batch_x.clear()
        batch_w.clear()

    for row in meta.itertuples():
        img = cv2.imread(row.path)
        kp = pk.get((row.frame, row.track_id))
        m = body_mask(kp, row, img.shape[:2])
        angle.append(torso_angle(kp))
        r, b, n = colour_counts(img, m)
        hues.append(hue_hist(img, m))
        red.append(r)
        blue.append(b)
        npx.append(n)
        h, w = img.shape[:2]
        s = max(h, w)
        oy, ox = (s - h) // 2, (s - w) // 2
        sq = np.zeros((s, s, 3), np.uint8)
        sq[oy : oy + h, ox : ox + w] = img[:, :, ::-1]
        msq = np.zeros((s, s), np.float32)
        msq[oy : oy + h, ox : ox + w] = m
        grid = cv2.resize(msq, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
        keep.append(grid.sum() >= 3)
        x = cv2.resize(sq, (224, 224)).astype(np.float32) / 255
        batch_x.append(((x - MEAN) / STD).transpose(2, 0, 1))
        batch_w.append(grid)
        if len(batch_x) == 64:
            flush()
    flush()
    f = np.concatenate(feats) if feats else np.zeros((0, 384))
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-6)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        feat=f.astype(np.float32),
        has_mask=np.array(keep),
        angle=np.array(angle),
        red=np.array(red),
        blue=np.array(blue),
        npx=np.array(npx),
        hue=np.array(hues, dtype=np.float32).reshape(-1, 18),
        track_id=meta.track_id.to_numpy(),
        frame=meta.frame.to_numpy(),
        merged=meta.merged.to_numpy(),
    )


def labels(fid: str, z: dict[str, np.ndarray]) -> np.ndarray:
    """Per crop: 1 = fighter (on the ground or kicking), 0 = never moves, -1 = unknown."""
    tracks, poses, _ = load(fid)
    t = tracks.assign(
        cx=(tracks.x1 + tracks.x2) / 2, cy=(tracks.y1 + tracks.y2) / 2, h=tracks.y2 - tracks.y1
    ).sort_values(["track_id", "frame"])
    g = t.groupby("track_id")
    dt = g["t"].diff()
    t["v"] = (np.hypot(g.cx.diff(), g.cy.diff()) / t.h / dt).where(dt <= 0.3)
    speed = t.groupby("track_id")["v"].median()
    life = t.groupby("track_id").size()
    w = poses.pivot_table(index=["frame", "track_id"], columns="kp", values=["y", "conf"])
    c, y = w["conf"], w["y"]
    kick_idx = w.index[
        ((c[15] > 0.5) & (c[14] > 0.5) & (y[15] < y[14]))
        | ((c[16] > 0.5) & (c[13] > 0.5) & (y[16] < y[13]))
    ]
    kick = set(map(tuple, kick_idx.to_frame().to_numpy().astype(int)))
    out = np.full(len(z["track_id"]), -1)
    for i, (tid, fr, ang) in enumerate(zip(z["track_id"], z["frame"], z["angle"], strict=True)):
        if (int(fr), int(tid)) in kick or (not np.isnan(ang) and ang > 60):
            out[i] = 1
        elif speed.get(tid, 1.0) < 0.03 and life.get(tid, 0) >= 30:
            out[i] = 0
    return out


def train_probe(fights: list[str]) -> tuple[np.ndarray, float]:
    """Logistic regression (numpy, L2) on body features of labelled crops from these fights."""
    xs, ys = [], []
    for fid in fights:
        z = dict(np.load(CACHE / f"{fid}.npz"))
        lab = labels(fid, z)
        ok = (lab >= 0) & z["has_mask"]
        xs.append(z["feat"][ok])
        ys.append(lab[ok])
    x, y = np.concatenate(xs), np.concatenate(ys).astype(float)
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)  # balance the classes
    sw = np.where(y == 1, pos_w, 1.0)
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(2000):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        gr = (p - y) * sw
        w -= 0.5 * (x.T @ gr / len(y) + 1e-3 * w)
        b -= 0.5 * gr.mean()
    print(
        f"probe trained on {len(fights)} fights: {int((y == 1).sum())} fighter, "
        f"{int((y == 0).sum())} not-fighter crops",
        flush=True,
    )
    return w, b


def groups(fid: str, z: dict[str, np.ndarray], tracks: pd.DataFrame, k: int = 8) -> dict[int, int]:
    """Per-fight projection (triplet loss) and cannot-link clustering: track -> group."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    use = z["has_mask"] & ~z["merged"]
    tid = z["track_id"]
    cooc: dict[int, set[int]] = {}
    for _, g in tracks.groupby("frame")["track_id"]:
        ids = [int(i) for i in g.unique()]
        for a in ids:
            cooc.setdefault(a, set()).update(b for b in ids if b != a)
    by_track = {int(t): np.flatnonzero(use & (tid == t)) for t in np.unique(tid[use])}
    by_track = {t: ix for t, ix in by_track.items() if len(ix) >= 2}
    anchors = [t for t in by_track if any(n in by_track for n in cooc.get(t, ()))]
    proj = torch.nn.Linear(
        z["feat"].shape[1] + (18 if HUE_WEIGHT > 0 and "hue" in z else 0), 64, bias=False
    )
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    x_np = z["feat"]
    if HUE_WEIGHT > 0 and "hue" in z:
        x_np = np.hstack([x_np, HUE_WEIGHT * np.sqrt(z["hue"])])
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
        emb = torch.nn.functional.normalize(proj(x), dim=1).numpy()
    ids = np.array(sorted(by_track))
    feat = np.stack([emb[by_track[t]].mean(0) for t in ids]) if len(ids) else np.zeros((0, 64))
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-6)
    pos = {int(t): i for i, t in enumerate(ids)}
    conf = [{pos[n] for n in cooc.get(int(t), ()) if n in pos} for t in ids]
    counts = tracks.groupby("track_id").size()
    tl = Tracklets(
        ids=ids, features=feat, weights=counts.reindex(ids).to_numpy(float), conflicts=conf
    )
    lab = merge_groups(tl, cluster(tl, k=k, max_dist=10.0), max_dist=0.5)
    return {int(t): int(g) for t, g in zip(ids, lab, strict=True) if g >= 0}


def assign(fid: str, w: np.ndarray, b: float, merged_weight: float) -> pd.DataFrame:
    """Track -> role (red, blue, referee, other) for one fight, using the probe."""
    tracks, _, _ = load(fid)
    z = dict(np.load(CACHE / f"{fid}.npz"))
    grp = groups(fid, z, tracks)
    g_of = np.array([grp.get(int(t), -1) for t in z["track_id"]])
    use = z["has_mask"] & ~z["merged"] & (g_of >= 0)
    prob = 1 / (1 + np.exp(-(z["feat"] @ w + b)))
    df = pd.DataFrame(
        {"group": g_of, "prob": prob, "red": z["red"], "blue": z["blue"], "n": z["npx"]}
    )[use]
    gs = df.groupby("group").agg(
        prob=("prob", "mean"),
        red=("red", "sum"),
        blue=("blue", "sum"),
        n=("n", "sum"),
        crops=("prob", "size"),
    )
    # Merged clinch boxes hold both fighters: groups that look like them are likelier fighters.
    mm = z["has_mask"] & z["merged"]
    if mm.any() and merged_weight > 0:
        centre = z["feat"][mm].mean(0)
        centre = centre / np.linalg.norm(centre)
        sim = pd.Series(z["feat"][use] @ centre, index=df.index).groupby(df.group).mean()
        gs["score"] = gs.prob + merged_weight * sim
    else:
        gs["score"] = gs.prob
    t_frames = tracks.assign(group=tracks.track_id.map(grp)).dropna(subset=["group"])
    frames_of = {int(g): set(f.frame) for g, f in t_frames.groupby("group")}
    gs["frames"] = pd.Series({g: len(frames_of.get(int(g), ())) for g in gs.index})

    def overlap(x: int, y: int) -> float:
        """Share of the smaller group's frames in which both groups are on screen."""
        shared = len(frames_of[int(x)] & frames_of[int(y)])
        return shared / max(1, min(len(frames_of[int(x)]), len(frames_of[int(y)])))

    if DEBUG:
        print(gs.assign(share=(gs.red - gs.blue) / gs.n.clip(lower=1)).round(3).to_string())
    # Fold fighter-like groups first: a fighter split by tracker ID switches shows up as
    # several groups that are almost never on screen together. Biggest first; each joins
    # the most similar existing person it barely overlaps with, or starts a new one.
    centre = {g: z["feat"][use][df.group.to_numpy() == g].mean(0) for g in gs.index}
    fighterish = sorted(
        [g for g in gs.index if gs.loc[g, "score"] >= FIGHTER_PROB], key=lambda g: -gs.frames[g]
    )
    people: list[int] = []
    for g in fighterish:
        fits = [p for p in people if overlap(g, p) < 0.05]
        if fits:
            p = max(fits, key=lambda q: float(centre[g] @ centre[q]))
            for t_, gg in list(grp.items()):
                if gg == g:
                    grp[t_] = int(p)
            gs.loc[p, ["red", "blue", "n"]] += gs.loc[g, ["red", "blue", "n"]]
            frames_of[int(p)] |= frames_of[int(g)]
            gs = gs.drop(index=g)
        else:
            people.append(g)
    gs["frames"] = pd.Series({g: len(frames_of.get(int(g), ())) for g in gs.index})
    # The two fighters: the pair of fighter-like people with the most screen time that
    # do appear together; if fewer than two, fall back to the highest scores.
    pairs = [(x, y) for i, x in enumerate(people) for y in people[i + 1 :] if overlap(x, y) >= 0.1]
    if pairs:
        a, bb = max(pairs, key=lambda xy: gs.frames[xy[0]] + gs.frames[xy[1]])
    else:
        top = sorted(gs.index, key=lambda g: -gs.loc[g, "score"])
        a, bb = top[0], top[1]
    if DEBUG:
        print(fid, "people", people, "fighters", a, bb)
    shares = (gs.red - gs.blue) / gs.n.clip(lower=1)
    red, blue = (a, bb) if shares[a] >= shares[bb] else (bb, a)
    names = {int(red): "red", int(blue): "blue"}
    rest = [g for g in gs.index if g not in (a, bb)]
    if rest:
        fighters = frames_of[int(a)] | frames_of[int(bb)]
        names[int(max(rest, key=lambda g: len(frames_of[int(g)] & fighters)))] = "referee"
    roles = pd.DataFrame(
        {"track_id": list(grp), "role": [names.get(g, "other") for g in grp.values()]}
    )
    # A tracklet the classifier itself calls a non-fighter doesn't stay in a fighter group,
    # even if clustering put it there (a referee or official stepping in close).
    track_prob = (
        pd.DataFrame({"track_id": z["track_id"], "prob": prob})[use]
        .groupby("track_id")["prob"]
        .mean()
    )
    low = roles.track_id.map(track_prob).fillna(0) < TRACK_MIN_PROB
    roles.loc[roles.role.isin(["red", "blue"]) & low, "role"] = "other"
    return roles


def montage(fid: str, roles: pd.DataFrame, out: Path, n: int = 12) -> np.ndarray:
    """Two rows (red, blue) of n random crops each, merged boxes left out."""
    z = dict(np.load(CACHE / f"{fid}.npz"))
    meta = pd.DataFrame({"track_id": z["track_id"], "frame": z["frame"], "merged": z["merged"]})
    meta = meta.merge(roles, on="track_id")
    rows = []
    for role in ("red", "blue"):
        mine = meta[(meta.role == role) & ~meta.merged]
        pick = mine.sample(min(n, len(mine)), random_state=0) if len(mine) else mine
        tiles = [
            cv2.resize(cv2.imread(f"data/crops/{fid}/{t:06d}_{f:07d}.jpg"), (60, 120))
            for t, f in zip(pick.track_id, pick.frame, strict=True)
        ]
        tiles += [np.zeros((120, 60, 3), np.uint8)] * (n - len(tiles))
        lab = np.zeros((120, 110, 3), np.uint8)
        cv2.putText(lab, fid[3:], (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(lab, role, (2, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.hstack([lab, *tiles]))
    return np.vstack(rows)


def assign_two_sides(fid: str, w: np.ndarray, b: float) -> pd.DataFrame:
    """Red and blue from co-occurrence: fighter-like tracklets seen together are different people.

    Among fighter-like tracklets, an edge joins any two on screen at the same time.
    With two fighters that graph is (nearly) two-colourable, so within each connected
    piece the split follows from the overlaps alone. Appearance (mean body feature)
    is only used to line pieces up with each other, and colour only to decide which
    side is red. Odd cycles (a third person scored as a fighter) are resolved by
    putting a tracklet on the side it conflicts with least.
    """
    tracks, _, _ = load(fid)
    z = dict(np.load(CACHE / f"{fid}.npz"))
    use = z["has_mask"] & ~z["merged"]
    prob = 1 / (1 + np.exp(-(z["feat"] @ w + b)))
    crops = pd.DataFrame(
        {"track_id": z["track_id"], "prob": prob, "red": z["red"], "blue": z["blue"], "n": z["npx"]}
    )[use]
    per = crops.groupby("track_id").agg(
        prob=("prob", "mean"), red=("red", "sum"), blue=("blue", "sum"), n=("n", "sum"),
        crops=("prob", "size"),
    )  # fmt: skip
    fighters = per[(per.prob >= FIGHTER_PROB) & (per.crops >= 2)].index.to_list()
    feat = {t: z["feat"][use][crops.track_id.to_numpy() == t].mean(0) for t in fighters}
    frames = {
        int(t): set(g.frame) for t, g in tracks[tracks.track_id.isin(fighters)].groupby("track_id")
    }
    size = {t: len(frames[t]) for t in fighters}
    nbrs: dict[int, set[int]] = {t: set() for t in fighters}
    for i, a in enumerate(fighters):
        for c in fighters[i + 1 :]:
            if frames[a] & frames[c]:
                nbrs[a].add(c)
                nbrs[c].add(a)
    side: dict[int, int] = {}
    comps: list[list[int]] = []
    for start in sorted(fighters, key=lambda t: -size[t]):
        if start in side:
            continue
        side[start] = 0
        comp, queue = [start], [start]
        while queue:
            t = queue.pop()
            for n in nbrs[t]:
                if n not in side:
                    # The side this tracklet conflicts with least among already-placed neighbours.
                    on0 = sum(1 for m in nbrs[n] if side.get(m) == 0)
                    on1 = sum(1 for m in nbrs[n] if side.get(m) == 1)
                    side[n] = 1 if on0 > on1 else 0 if on1 > on0 else 1 - side[t]
                    comp.append(n)
                    queue.append(n)
        comps.append(comp)

    def mean_of(ts: list[int]) -> np.ndarray:
        if not ts:
            return np.zeros(z["feat"].shape[1])
        m = np.average([feat[t] for t in ts], axis=0, weights=[size[t] for t in ts])
        return m / max(float(np.linalg.norm(m)), 1e-6)

    # Line up each piece with the biggest one by appearance (flip its sides if that fits better).
    anchor = comps[0] if comps else []
    a0 = mean_of([t for t in anchor if side[t] == 0])
    a1 = mean_of([t for t in anchor if side[t] == 1])
    for comp in comps[1:]:
        c0 = mean_of([t for t in comp if side[t] == 0])
        c1 = mean_of([t for t in comp if side[t] == 1])
        keep = float(c0 @ a0 + c1 @ a1)
        flip = float(c0 @ a1 + c1 @ a0)
        if flip > keep:
            for t in comp:
                side[t] = 1 - side[t]
    sums = (
        per.loc[fighters]
        .assign(side=[side[t] for t in fighters])
        .groupby("side")[["red", "blue", "n"]]
        .sum()
    )
    share = (sums.red - sums.blue) / sums.n.clip(lower=1)
    red_side = int(share.idxmax()) if len(share) == 2 else 0
    names = {t: ("red" if side[t] == red_side else "blue") for t in fighters}
    all_ids = sorted({int(t) for t in z["track_id"]})
    if DEBUG:
        print(
            fid,
            "fighter tracklets",
            len(fighters),
            "pieces",
            len(comps),
            "share",
            share.round(3).to_dict(),
        )
    return pd.DataFrame({"track_id": all_ids, "role": [names.get(t, "other") for t in all_ids]})


def run(
    out: Path,
    train: list[str],
    test: list[str],
    merged_weight: float,
    tag: str,
    method: str = "groups",
) -> None:
    w, b = train_probe(train)
    sheets = []
    for fid in test:
        if method == "sides":
            roles = assign_two_sides(fid, w, b)
        else:
            roles = assign(fid, w, b, merged_weight)
        roles.to_parquet(out / f"roles_{tag}_{fid}.parquet")
        sheets.append(montage(fid, roles, out))
    for i in range(0, len(sheets), 4):
        sep = np.full((6, sheets[0].shape[1], 3), 255, np.uint8)
        page = np.vstack([x for s_ in sheets[i : i + 4] for x in (s_, sep)])
        cv2.imwrite(str(out / f"sheet_{tag}_{i // 4:02d}.jpg"), page)


if __name__ == "__main__":
    if sys.argv[1] == "features":
        mdl = AutoModel.from_pretrained("facebook/dinov2-small", revision="ed25f3a31f01")
        mdl = mdl.eval().to(DEV)
        for fight in sys.argv[2:]:
            features(fight, mdl)
            print("features", fight, flush=True)
    elif sys.argv[1] == "loo":
        # Leave one fight out: train on the others, assign the held-out one.
        out_dir = Path(sys.argv[2])
        fights = sys.argv[4:]
        for held in fights:
            run(
                out_dir, [f for f in fights if f != held], [held], float(sys.argv[3]), f"loo_{held}"
            )
    elif sys.argv[1] == "split":
        # Train on fights listed in TRAIN_FILE, assign the rest of the cached fights.
        out_dir = Path(sys.argv[2])
        train_ids = Path(sys.argv[4]).read_text().split()
        test_ids = Path(sys.argv[5]).read_text().split()
        method = sys.argv[7] if len(sys.argv) > 7 else "groups"
        run(out_dir, train_ids, test_ids, float(sys.argv[3]), sys.argv[6], method)
