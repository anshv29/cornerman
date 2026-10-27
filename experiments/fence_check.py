"""Blind by-eye check of the cage-pressure rule (labeller G, perception/fence.py), 2026-10-27.

    uv run python experiments/fence_check.py sample OUT_DIR
    uv run python experiments/fence_check.py score OUT_DIR VERDICTS_CSV
    uv run python experiments/fence_check.py sample2 OUT_DIR   (round 2, below)

Round 2 (2026-10-27): after round 1 the far fence was the main source of false calls.
Two fixes were set by reasoning (no far-fence calls; vertical distances x2 for
perspective) and compared on round 1's 40 moments (dev): x2 kept 8 of 10 calls right
against 10 of 20. Round 2 tests x2 on a fresh sample, seed 1, round 1's moments left
out, with the same tiles, the same two sets and the same verdict rules.

Silver has 1 cage clip, so G gets its own check. Thumbnails (arena stage, every
5 s) at moments where G's conditions hold (both fighters boxed in a live shot,
standing, not touching, at striking range):
- called: 20 where G calls one fighter pressured against the fence;
- not called: 20 where G says neutral or abstains but a fighter's feet are within
  twice `fence_near` of a real mat edge (the hard negatives).
Seed 0, shuffled, numbered only. Each tile shows the red and blue fighters' boxes,
their feet points and the mat edges G treats as fence (green). The key is read
after the verdicts are written.

Verdict per tile: red_pressured, blue_pressured, neutral, or cant_judge.
"Pressured" means a fighter's back is at or within a step of the fence with the
opponent in front at striking range, cutting off the open mat. Agent-judged, blind
to the set but not to the footage; a sanity check, not an unbiased accuracy.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.config import PerceptionConfig
from cornerman.contracts.perception import fighter_boxes, population_fights
from cornerman.events.moments import fight_moments
from cornerman.perception.fence import real_edges

DERIVED = Path("data/derived")
THUMBS = Path("data/thumbs")
N_EACH = 20
SCALE = 1.5


def pool(p: PerceptionConfig) -> pd.DataFrame:
    rows = []
    for fid in population_fights(DERIVED):
        if not (DERIVED / fid / "fighting" / "manifest.json").exists():
            continue
        have = {int(x.stem) for x in (THUMBS / fid).glob("*.jpg")}
        m = fight_moments(DERIVED / fid, p)
        m = m[m["frame"].isin(list(have))]
        near = m[["red_fence", "blue_fence"]].min(axis=1) <= 2 * p.fence_near
        lo, hi = p.fence_gap
        ready = m["live"].astype(bool) & ~m["contact"].astype(bool) & m["gap"].between(lo, hi)
        called = m["G"].isin(["red-pressuring", "blue-pressuring"])
        for kind, sel in (("called", called), ("not_called", ready & near & ~called)):
            for r in m[sel].itertuples():
                rows.append({"fight_id": fid, "frame": int(r.frame), "kind": kind, "G": r.G})
    return pd.DataFrame(rows)


def draw(fid: str, frame: int, n: int) -> np.ndarray:
    import json

    video = json.loads((DERIVED / fid / "ingest" / "video.json").read_text())
    w, h = float(video["width"]), float(video["height"])
    img = cv2.imread(str(THUMBS / fid / f"{frame:07d}.jpg"))
    img = cv2.resize(img, (int(320 * SCALE), int(180 * SCALE)))
    sx, sy = img.shape[1] / w, img.shape[0] / h
    hull = pd.read_parquet(DERIVED / fid / "arena" / "hull.parquet").set_index("frame")
    if frame in hull.index:
        for x0, y0, x1, y1 in real_edges(list(hull.loc[frame, "x"]), list(hull.loc[frame, "y"]),
                                         w, h):  # fmt: skip
            cv2.line(img, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)),
                     (0, 220, 0), 2)  # fmt: skip
    tracks = pd.read_parquet(DERIVED / fid / "tracking" / "tracks.parquet")
    tracks = tracks[tracks["frame"] == frame]
    roles = pd.read_parquet(DERIVED / fid / "identity" / "roles.parquet")
    live = pd.read_parquet(DERIVED / fid / "tracking" / "live.parquet")
    for r in fighter_boxes(tracks, roles, live).itertuples():
        colour = (0, 0, 255) if r.role == "red" else (255, 80, 0)
        cv2.rectangle(img, (int(r.x1 * sx), int(r.y1 * sy)), (int(r.x2 * sx), int(r.y2 * sy)),
                      colour, 2)  # fmt: skip
        cv2.circle(img, (int((r.x1 + r.x2) / 2 * sx), int(r.y2 * sy)), 4, (0, 255, 255), -1)
    cv2.putText(img, str(n), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def sample(
    out: Path, p: PerceptionConfig | None = None, seed: int = 0, skip: Path | None = None
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pl = pool(p or PerceptionConfig())
    if skip is not None:
        old = pd.read_csv(skip)[["fight_id", "frame"]]
        pl = pl.merge(old, how="left", indicator=True)
        pl = pl[pl["_merge"] == "left_only"].drop(columns="_merge")
    print(pl.groupby("kind").size().to_string())
    pick = (
        pd.concat([g.sample(min(N_EACH, len(g)), random_state=seed) for _, g in pl.groupby("kind")])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    pick.insert(0, "n", range(1, len(pick) + 1))
    pick.to_csv(out / "key.csv", index=False)
    tiles = [draw(r.fight_id, r.frame, r.n) for r in pick.itertuples()]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 2)
    rows_img = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    for k in range(0, len(rows_img), 3):
        cv2.imwrite(str(out / f"page_{k // 3:02d}.jpg"), np.vstack(rows_img[k : k + 3]))


def score(out: Path, verdicts: Path) -> None:
    v = pd.read_csv(verdicts).merge(pd.read_csv(out / "key.csv"), on="n")
    judged = v[v["verdict"] != "cant_judge"]
    print(f"can't judge: {(v['verdict'] == 'cant_judge').sum()} of {len(v)}")
    called = judged[judged["kind"] == "called"]
    want = called["G"].map({"red-pressuring": "blue_pressured", "blue-pressuring": "red_pressured"})
    right = int((called["verdict"] == want).sum())
    neutral = int((called["verdict"] == "neutral").sum())
    flipped = len(called) - right - neutral
    print(f"called: {right} of {len(called)} right (direction and all); judged neutral "
          f"{neutral}, the other fighter pressured {flipped}")  # fmt: skip
    nc = judged[judged["kind"] == "not_called"]
    print(f"not called: {(nc['verdict'] == 'neutral').sum()} of {len(nc)} judged neutral")
    print(v.groupby(["kind", "verdict"]).size().to_string())


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(Path(sys.argv[2]))
    elif sys.argv[1] == "sample2":
        sample(Path(sys.argv[2]), PerceptionConfig(fence_ky=2.0), seed=1,
               skip=Path("data/fence_check/key.csv"))  # fmt: skip
    else:
        score(Path(sys.argv[2]), Path(sys.argv[3]))
