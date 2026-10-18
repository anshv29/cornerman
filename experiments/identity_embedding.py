"""Experiment (2026-10-18): identity from a per-fight appearance embedding.

Not part of the pipeline: it didn't reach the target (red and blue right on 4 of 5
pilot fights). See DECISIONS.md, "Offline identity experiments".

Runs on the laptop from local crops and poses, no video and no Kaggle:
    uv run python experiments/identity_embedding.py FIGHT_ID OUT_DIR \
        [MERGED=1.6] [MERGE_DIST=0.5] [K=8] [MAX_DIST=10] [NAMING=engaged|closest]
Writes OUT_DIR/emb_roles_{fight}.parquet, a montage of 12 crops per role, and a
montage of 8 crops per raw group. Montages contain fighters: keep OUT_DIR outside
the repo (e.g. the scratchpad).

Steps: DINOv2-small patch features pooled over the pose body mask; a linear
projection trained per fight with a triplet loss (same tracklet = same person,
tracklets on screen together = different people); cannot-link clustering of
tracklets; groups named red, blue, referee by which pair are the fighters
(NAMING) and pooled red and blue pixel shares between them.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

from cornerman.perception.identity import Tracklets, cluster, merge_groups

fid, out = sys.argv[1], Path(sys.argv[2])
NAMING = sys.argv[7] if len(sys.argv) > 7 else "engaged"
d = Path(f"data/derived/{fid}")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
rng = np.random.default_rng(0)
torch.manual_seed(0)

tracks = pd.read_parquet(d / "tracking/tracks.parquet")
poses = pd.read_parquet(d / "pose/poses.parquet")
crop_files = sorted(Path(f"data/crops/{fid}").glob("*.jpg"))
meta = pd.DataFrame(
    [(int(c.name[:6]), int(c.name[7:14]), c) for c in crop_files],
    columns=["track_id", "frame", "path"],
)
meta = meta.merge(tracks[["frame", "track_id", "x1", "y1", "x2", "y2"]], on=["frame", "track_id"])
meta["wh"] = (meta.x2 - meta.x1) / (meta.y2 - meta.y1)
MERGED = float(sys.argv[3]) if len(sys.argv) > 3 else 1.6
meta["merged"] = meta.wh > MERGED * meta.wh.median()
print(fid, "crops", len(meta), "merged-looking", int(meta.merged.sum()))

# 1. DINOv2 patch features pooled over the pose body mask.

model = AutoModel.from_pretrained("facebook/dinov2-small", revision="ed25f3a31f01").eval().to(dev)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


pk = {k: g.sort_values("kp") for k, g in poses.groupby(["frame", "track_id"])}
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]


def body_mask(row, shape) -> np.ndarray:
    """Torso polygon plus thick limb segments, from the pose, in crop pixels."""
    m = np.zeros(shape, np.uint8)
    k = pk.get((row.frame, row.track_id))
    if k is None or len(k) < 17:
        return m
    s = shape[0] / (row.y2 - row.y1)
    xy = ((k[["x", "y"]].to_numpy() - [row.x1, row.y1]) * s).astype(np.int32)
    c = k.conf.to_numpy()
    if min(c[[5, 6, 11, 12]]) > 0.3:
        cv2.fillPoly(m, [xy[[5, 6, 12, 11]]], 1)
    for a, b in LIMBS:
        if min(c[a], c[b]) > 0.3:
            cv2.line(m, tuple(xy[a]), tuple(xy[b]), 1, max(2, int(0.08 * shape[0])))
    return m


def prep(row) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(row.path))[:, :, ::-1]
    m = body_mask(row, img.shape[:2])
    h, w = img.shape[:2]
    s = max(h, w)
    oy, ox = (s - h) // 2, (s - w) // 2
    sq = np.zeros((s, s, 3), np.uint8)
    sq[oy : oy + h, ox : ox + w] = img
    msq = np.zeros((s, s), np.float32)
    msq[oy : oy + h, ox : ox + w] = m
    x = cv2.resize(sq, (224, 224)).astype(np.float32) / 255
    grid = cv2.resize(msq, (16, 16), interpolation=cv2.INTER_AREA)
    return ((x - mean) / std).transpose(2, 0, 1), grid.reshape(-1)


emb, has = [], []
with torch.no_grad():
    for i in range(0, len(meta), 64):
        batch = [prep(r) for r in meta.iloc[i : i + 64].itertuples()]
        x = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=dev)
        patches = model(pixel_values=x).last_hidden_state[:, 1:, :].cpu().numpy()  # (n, 256, 384)
        wts = np.stack([b[1] for b in batch])
        has.extend(wts.sum(1) >= 3)
        pooled = (patches * wts[:, :, None]).sum(1) / np.maximum(wts.sum(1, keepdims=True), 1e-6)
        emb.append(pooled)
E = np.concatenate(emb)
E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
meta["merged"] = meta["merged"] | ~np.array(has)  # no usable body mask: leave out like merged boxes
print("crops with a body mask", int(np.sum(has)))

# 2. Per-fight projection: same tracklet = same person; co-occurring tracklets = different people.
counts = tracks.groupby("track_id").size()
cooc: dict[int, set[int]] = {}
for _, g in tracks.groupby("frame")["track_id"]:
    ids = list(g.unique())
    for a in ids:
        cooc.setdefault(a, set()).update(b for b in ids if b != a)
train = meta[~meta.merged].reset_index(drop=True)
Et = E[~meta.merged.to_numpy()]
by_track = {t: np.flatnonzero(train.track_id.to_numpy() == t) for t in train.track_id.unique()}
anchors = [
    t for t, ix in by_track.items() if len(ix) >= 2 and any(n in by_track for n in cooc.get(t, ()))
]
print("tracklets usable for training", len(anchors))
W = torch.nn.Linear(E.shape[1], 64, bias=False)
opt = torch.optim.Adam(W.parameters(), lr=1e-3)
X = torch.tensor(Et, dtype=torch.float32)
for _step in range(400):
    a_ix, p_ix, n_ix = [], [], []
    for t in rng.choice(anchors, 128):
        i, j = rng.choice(by_track[t], 2, replace=False)
        negs = [n for n in cooc[t] if n in by_track]
        k = rng.choice(by_track[int(rng.choice(negs))])
        a_ix.append(i)
        p_ix.append(j)
        n_ix.append(k)
    f = torch.nn.functional.normalize(W(X), dim=1)
    loss = torch.nn.functional.triplet_margin_with_distance_loss(
        f[a_ix], f[p_ix], f[n_ix], distance_function=lambda u, v: 1 - (u * v).sum(1), margin=0.3
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
print("final triplet loss", round(float(loss), 3))
with torch.no_grad():
    F = torch.nn.functional.normalize(W(torch.tensor(E, dtype=torch.float32)), dim=1).numpy()

# 3. Tracklet embeddings (non-merged crops only), cannot-link clustering, merge look-alikes.
keep_t = [t for t in meta.track_id.unique() if ((meta.track_id == t) & ~meta.merged).sum() >= 2]
ids = np.array(sorted(keep_t))
feat = np.stack([F[((meta.track_id == t) & ~meta.merged).to_numpy()].mean(0) for t in ids])
feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
pos = {t: i for i, t in enumerate(ids)}
conf = [set(pos[n] for n in cooc.get(t, ()) if n in pos) for t in ids]
tl = Tracklets(ids=ids, features=feat, weights=counts.reindex(ids).to_numpy(float), conflicts=conf)
labels = cluster(
    tl,
    k=int(sys.argv[5]) if len(sys.argv) > 5 else 6,
    max_dist=float(sys.argv[6]) if len(sys.argv) > 6 else 10.0,
)

labels = merge_groups(tl, labels, max_dist=float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)


# 4. Name groups by pooled red/blue share in pose-masked torso and thighs.
def body_counts(row: pd.Series) -> tuple[int, int, int]:
    k = pk.get((row.frame, row.track_id))
    if k is None or len(k) < 17:
        return 0, 0, 0
    img = cv2.imread(str(row.path))
    s = img.shape[0] / (row.y2 - row.y1)
    xy = (k[["x", "y"]].to_numpy() - [row.x1, row.y1]) * s
    c = k.conf.to_numpy()
    m = np.zeros(img.shape[:2], np.uint8)
    if min(c[[5, 6, 11, 12]]) > 0.3:
        cv2.fillPoly(m, [xy[[5, 6, 12, 11]].astype(np.int32)], 1)
    for hip, knee in ((11, 13), (12, 14)):
        if min(c[hip], c[knee]) > 0.3:
            dd = xy[knee] - xy[hip]
            n = np.array([-dd[1], dd[0]])
            n = n / (np.linalg.norm(n) + 1e-6) * 0.07 * img.shape[0]
            cv2.fillPoly(
                m,
                [np.array([xy[hip] + n, xy[knee] + n, xy[knee] - n, xy[hip] - n]).astype(np.int32)],
                1,
            )
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[m > 0]
    if len(hsv) < 20:
        return 0, 0, 0
    h, sat, v = hsv[:, 0].astype(int), hsv[:, 1], hsv[:, 2]
    red = (sat >= 70) & (v >= 35) & ((h <= 5) | (h >= 155))
    blue = (sat >= 120) & (v >= 40) & (v <= 170) & (h >= 100) & (h <= 130)
    return int(red.sum()), int(blue.sum()), len(hsv)


bc = meta.apply(body_counts, axis=1, result_type="expand")
bc.columns = ["red", "blue", "n"]
meta = pd.concat([meta, bc], axis=1)
group_of = {int(t): int(lab_) for t, lab_ in zip(ids, labels, strict=True)}
meta["group"] = meta.track_id.map(group_of).fillna(-1).astype(int)
g = meta[(meta.group >= 0) & ~meta.merged].groupby("group")[["red", "blue", "n"]].sum()
g["red_share"] = g.red / g.n
g["blue_share"] = g.blue / g.n
g["frames"] = pd.Series({gr: float(tl.weights[labels == gr].sum()) for gr in g.index})
tt = tracks.assign(
    cx=(tracks.x1 + tracks.x2) / 2, cy=(tracks.y1 + tracks.y2) / 2, h=tracks.y2 - tracks.y1
).sort_values(["track_id", "frame"])
gg = tt.groupby("track_id")
dt = gg["t"].diff()
tt["v"] = (np.hypot(gg.cx.diff(), gg.cy.diff()) / tt.h / dt).where(dt <= 0.3)
speed = tt.groupby("track_id")["v"].median()
g["speed"] = pd.Series(
    {
        gr: float(
            np.average(speed.reindex(ids[labels == gr]).fillna(0), weights=tl.weights[labels == gr])
        )
        for gr in g.index
    }
)
g["n_tracks"] = pd.Series({gr: int((labels == gr).sum()) for gr in g.index})
print(g.round(4).to_string())
names: dict[int, str] = {}
# Fighters: among groups with real screen time, the pair that is closest (median
# distance in box heights) whenever both are on screen, counting only pairs that
# appear together in at least 20% of the smaller group's frames.
tg = tt.assign(group=tt.track_id.map(group_of)).dropna(subset=["group"])
tg = tg[tg.group >= 0]
per = tg.groupby(["frame", "group"])[["cx", "cy", "h"]].mean().reset_index()
frames_of = {gr: set(per[per.group == gr].frame) for gr in g.index}
big = [gr for gr in g.index if g.loc[gr, "frames"] >= 0.25 * g.frames.max()]
pairs = {}
for i, a in enumerate(big):
    for b in big[i + 1 :]:
        A = per[per.group == a].set_index("frame")
        B = per[per.group == b].set_index("frame")
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        if len(j) < 0.2 * min(len(frames_of[a]), len(frames_of[b])):
            continue
        d = np.hypot(j.cx_a - j.cx_b, j.cy_a - j.cy_b) / ((j.h_a + j.h_b) / 2)
        pairs[(a, b)] = float(np.median(d))
print(
    "median pair distance:",
    {k: round(v, 2) for k, v in sorted(pairs.items(), key=lambda kv: kv[1])[:4]},
)
if NAMING == "engaged":
    counts_close = {}
    for x, y in [(x, y) for i, x in enumerate(sorted(g.index)) for y in sorted(g.index)[i + 1 :]]:
        A = per[per.group == x].set_index("frame")
        B = per[per.group == y].set_index("frame")
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        d = np.hypot(j.cx_a - j.cx_b, j.cy_a - j.cy_b) / ((j.h_a + j.h_b) / 2)
        counts_close[(x, y)] = int((d < 1.5).sum())
    a, b = max(counts_close, key=counts_close.get)
else:
    a, b = min(pairs, key=pairs.get)
sa = g.loc[a, "red_share"] - g.loc[a, "blue_share"]
sb = g.loc[b, "red_share"] - g.loc[b, "blue_share"]
red, blue = (a, b) if sa >= sb else (b, a)
names[int(red)] = "red"
names[int(blue)] = "blue"
rest = [x for x in g.index if x not in (a, b)]
if rest:
    F = frames_of[a] | frames_of[b]
    names[int(max(rest, key=lambda x: len(frames_of[x] & F)))] = "referee"
meta["role"] = meta.group.map(names).fillna("other")
meta.loc[meta.merged, "role"] = "merged"
roles = pd.DataFrame({"track_id": ids, "role": [names.get(int(lab_), "other") for lab_ in labels]})
roles.to_parquet(out / f"emb_roles_{fid}.parquet")

# 5. Montage: 12 random crops per role.
rows = []
for role in ["red", "blue", "referee", "other", "merged"]:
    mine = meta[meta.role == role]
    pick = mine.sample(min(12, len(mine)), random_state=0) if len(mine) else mine
    tiles = [cv2.resize(cv2.imread(str(p)), (60, 120)) for p in pick.path] + [
        np.zeros((120, 60, 3), np.uint8)
    ] * (12 - len(pick))
    lab = np.zeros((120, 90, 3), np.uint8)
    cv2.putText(lab, role, (2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(np.hstack([lab, *tiles]))
cv2.imwrite(str(out / f"emb_montage_{fid}.jpg"), np.vstack(rows))
print({r: int((meta.role == r).sum()) for r in ["red", "blue", "referee", "other", "merged"]})

# group_montage: 8 crops per raw group, for diagnosis
rows = []
for gr in sorted(meta.group.unique()):
    if gr < 0:
        continue
    mine = meta[(meta.group == gr) & ~meta.merged]
    pick = mine.sample(min(8, len(mine)), random_state=1) if len(mine) else mine
    tiles = [cv2.resize(cv2.imread(str(pp)), (60, 120)) for pp in pick.path] + [
        np.zeros((120, 60, 3), np.uint8)
    ] * (8 - len(pick))
    lab = np.zeros((120, 70, 3), np.uint8)
    cv2.putText(
        lab,
        f"g{gr} {names.get(gr, '')}",
        (2, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    rows.append(np.hstack([lab, *tiles]))
cv2.imwrite(str(out / f"groups_{fid}.jpg"), np.vstack(rows))
