"""Blind by-eye check of the clean-skeleton rule (perception/tangle.py), 2026-10-26.

    uv run python experiments/clean_skeleton_check.py sample
    uv run python experiments/clean_skeleton_check.py score VERDICTS_CSV

The tangle stage keeps a fighter's skeleton (from the masks stage: ViTPose with the
other people greyed out) only if it is clean: at least 8 confident keypoints, at
least 70% inside its own mask, at most 15% on the other fighter's. This checks
whether the kept ones really are clean and the dropped ones really aren't.

sample: skeletons of the fighter pair in shots that aren't live (the shots the masks
stage was added for), from every fight with tangle output: 20 the rule keeps and 20
it drops that still have 8 or more confident keypoints (fewer is unusable anyway, so
judging those would flatter the rule). Seed 0. Each video is downloaded, checked to
be the one processed, the frames saved full size, and the video deleted. One tile
per skeleton, cropped to the pair, only that skeleton drawn (no masks, which are
what the rule looks at), shuffled and numbered; the key is read only by `score`.
A tile is "clean" if every drawn point sits on one fighter's body, roughly where
that joint is, and nothing is on the other fighter; "bad" otherwise; "cant_judge"
if the frame isn't two fighters (a referee, a close-up) or is too blurred.
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.config import PerceptionConfig
from cornerman.remote import videos

DERIVED = Path("data/derived")
OUT = Path("data/clean_skeleton_check")
N_EACH, MIN_KPS, SEED = 20, 8, 0
TILE_W = 360
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]  # fmt: skip


def candidates(fid: str, kp_min: float) -> pd.DataFrame:
    d = DERIVED / fid
    sk = pd.read_parquet(d / "tangle" / "skeletons.parquet")
    inst = pd.read_parquet(d / "masks" / "instances.parquet")
    frames = pd.read_parquet(d / "tracking" / "frames.parquet", columns=["frame", "shot_id"])
    live = pd.read_parquet(d / "tracking" / "live.parquet", columns=["shot_id", "live"])
    not_live = frames.merge(live, on="shot_id")
    not_live = set(not_live.loc[~not_live["live"], "frame"])
    c = sk[sk["in_pair"] & sk["frame"].isin(list(not_live))].merge(inst, on=["frame", "inst"])
    c["n_kps"] = [int((np.asarray(k) >= kp_min).sum()) for k in c["kc"]]
    c = c[c["clean"] | (c["n_kps"] >= MIN_KPS)]
    return c.assign(fight_id=fid)


def sample() -> None:
    p = PerceptionConfig()
    fights = sorted(f.parent.parent.name for f in DERIVED.glob("*/tangle/manifest.json"))
    pool = pd.concat([candidates(f, p.kp_min_conf) for f in fights], ignore_index=True)
    pick = (
        pd.concat(
            [g.sample(min(N_EACH, len(g)), random_state=SEED) for _, g in pool.groupby("clean")]
        )
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    pick.insert(0, "n", range(1, len(pick) + 1))
    print(f"{len(pool)} candidates from {pool.fight_id.nunique()} fights; picked {len(pick)} "
          f"from {pick.fight_id.nunique()} fights")  # fmt: skip
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "tiles").mkdir(parents=True)
    pick[["n", "fight_id", "frame", "inst", "clean", "n_kps"]].to_csv(OUT / "key.csv", index=False)
    urls = pd.read_csv("data/fights.csv").set_index("fight_id")["url"]
    for fid, g in pick.groupby("fight_id"):
        video = Path("data/raw/skeleton_check") / f"{fid}.mp4"
        try:
            videos.download(str(urls[fid]), video, 720)
            info = json.loads((DERIVED / str(fid) / "ingest" / "video.json").read_text())
            if videos.matches_processed(video, info) is None:
                print(fid, "different video, skipped")
                continue
            inst = pd.read_parquet(DERIVED / str(fid) / "masks" / "instances.parquet")
            pairs = pd.read_parquet(DERIVED / str(fid) / "tangle" / "pairs.parquet")
            cap = cv2.VideoCapture(str(video))
            for r in g.itertuples():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
                ok, img = cap.read()
                if ok:
                    pr = pairs[pairs["frame"] == r.frame].iloc[0]
                    both = inst[(inst["frame"] == r.frame) & inst["inst"].isin([pr.a, pr.b])]
                    tile = _tile(img, r, both, p.kp_min_conf)
                    cv2.imwrite(str(OUT / "tiles" / f"{r.n:02d}.jpg"), tile)
            cap.release()
            print(fid, "ok", flush=True)
        finally:
            video.unlink(missing_ok=True)
    _pages()


def _tile(img: np.ndarray, r, both: pd.DataFrame, kp_min: float) -> np.ndarray:
    kx, ky, kc = np.asarray(r.kx), np.asarray(r.ky), np.asarray(r.kc)
    for a, b in EDGES:
        if kc[a] >= kp_min and kc[b] >= kp_min:
            pa, pb = (int(kx[a]), int(ky[a])), (int(kx[b]), int(ky[b]))
            cv2.line(img, pa, pb, (0, 255, 255), 2, cv2.LINE_AA)
    for j in range(len(kc)):
        if kc[j] >= kp_min:
            cv2.circle(img, (int(kx[j]), int(ky[j])), 4, (255, 0, 255), -1, cv2.LINE_AA)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = both.x1.min(), both.y1.min(), both.x2.max(), both.y2.max()
    m = 0.2 * max(x2 - x1, y2 - y1)
    x1, y1 = int(max(0, x1 - m)), int(max(0, y1 - m))
    x2, y2 = int(min(w, x2 + m)), int(min(h, y2 + m))
    c = img[y1:y2, x1:x2]
    c = cv2.resize(c, (TILE_W, int(TILE_W * c.shape[0] / c.shape[1])))
    cv2.putText(c, str(r.n), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return c


def _pages() -> None:
    tiles = [cv2.imread(str(t)) for t in sorted((OUT / "tiles").glob("*.jpg"))]
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
    rows = [np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)]
    for k in range(0, len(rows), 3):
        cv2.imwrite(str(OUT / f"page_{k // 3:02d}.jpg"), np.vstack(rows[k : k + 3]))


def score(verdicts: Path) -> None:
    v = pd.read_csv(verdicts).merge(pd.read_csv(OUT / "key.csv"), on="n")
    v = v[v["verdict"] != "cant_judge"]
    for kept, g in v.groupby("clean"):
        right = (g["verdict"] == ("clean" if kept else "bad")).sum()
        print(f"rule {'keeps' if kept else 'drops'}: {right} of {len(g)} judged right")
    print(v.groupby(["clean", "verdict"]).size().to_string())


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample()
    else:
        score(Path(sys.argv[2]))
