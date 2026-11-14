"""Blind by-eye check of the strike rule (perception/strikes.py), 2026-11-14.

    uv run python experiments/strike_check.py sample OUT_DIR
    uv run python experiments/strike_check.py score OUT_DIR VERDICTS_CSV

The silver-set clips (3 s, 10 fps, red and blue boxes drawn) are the only video
left on the laptop, so the check uses moments inside them:
- detected: 20 strikes the rule found;
- not detected: 20 moments where the rule could have fired (live, apart, at
  striking range) with no detection within 0.5 s.
At most 2 per fight, seed 0, shuffled, numbered only. Each tile is a strip of 5
frames, 0.1 s apart, centred on the moment. The key is read after the verdicts
are written.

Verdict per tile: which boxed fighters throw a punch, kick, knee or elbow inside
the strip: red, blue, both, none, or cant_judge. Agent-judged, blind to the set
but not to the footage; a sanity check, not an unbiased accuracy.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CLIPS = Path("data/gold/clips")
N_EACH = 20
PER_FIGHT = 2
FRAMES = (-0.2, -0.1, 0.0, 0.1, 0.2)
WIDTH = 288


def pool() -> pd.DataFrame:
    s = pd.read_parquet("data/states/strikes.parquet")
    m = pd.read_parquet("data/states/moments.parquet")
    c = pd.read_parquet("data/gold/clips.parquet")[["clip_id", "fight_id", "t0", "t1"]]
    inside = s.merge(c, on="fight_id")
    inside = inside[(inside["t"] >= inside["t0"] + 0.2) & (inside["t"] <= inside["t1"] - 0.3)]
    det = inside.assign(kind="detected")[["fight_id", "clip_id", "t", "t0", "kind", "actor"]]
    m = m.drop(columns=["clip_id"], errors="ignore")
    ready = m[m["live"].astype(bool) & ~m["contact"].astype(bool) & (m["gap"] <= 1.5)]
    cand = ready.merge(c, on="fight_id")
    cand = cand[(cand["t"] >= cand["t0"] + 0.2) & (cand["t"] <= cand["t1"] - 0.3)]
    near = cand.merge(s[["fight_id", "t"]].rename(columns={"t": "ts"}), on="fight_id", how="left")
    hit = near[(near["ts"] - near["t"]).abs() <= 0.5].groupby(["fight_id", "t"]).size()
    keys = pd.MultiIndex.from_arrays([cand["fight_id"], cand["t"]])
    cand = cand[~keys.isin(hit.index)]
    miss = cand.assign(kind="not_detected", actor=None)[det.columns]
    return pd.concat([det, miss], ignore_index=True)


def strip(clip_id: str, t: float, t0: float, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(CLIPS / f"{clip_id}.mp4"))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
    frames: list[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    tiles: list[np.ndarray] = []
    for dt in FRAMES:
        i = int(np.clip(round((t + dt - t0) * fps), 0, len(frames) - 1))
        f = frames[i]
        tiles.append(cv2.resize(f, (WIDTH, round(f.shape[0] * WIDTH / f.shape[1]))))
    out = np.hstack(tiles)
    cv2.putText(out, str(n), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return out


def sample(out: Path, seed: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pl = pool()
    print(pl.groupby("kind").size().to_string())
    picks = []
    for _, g in pl.groupby("kind"):
        g = g.sample(frac=1, random_state=seed).groupby("fight_id").head(PER_FIGHT)
        picks.append(g.head(N_EACH))
    pick = pd.concat(picks).sample(frac=1, random_state=seed).reset_index(drop=True)
    pick.insert(0, "n", range(1, len(pick) + 1))
    pick.to_csv(out / "key.csv", index=False)
    tiles = [strip(r.clip_id, r.t, r.t0, r.n) for r in pick.itertuples()]
    for k in range(0, len(tiles), 4):
        cv2.imwrite(str(out / f"page_{k // 4:02d}.jpg"), np.vstack(tiles[k : k + 4]))


def score(out: Path, verdicts: Path) -> None:
    v = pd.read_csv(verdicts).merge(pd.read_csv(out / "key.csv"), on="n")
    print(pd.crosstab(v["kind"], v["verdict"]).to_string())
    d = v[(v["kind"] == "detected") & (v["verdict"] != "cant_judge")]
    right = d.apply(lambda r: r["verdict"] in (r["actor"], "both"), axis=1).sum()
    other = (d["verdict"].isin(["red", "blue"]) & (d["verdict"] != d["actor"])).sum()
    none = (d["verdict"] == "none").sum()
    print(f"detected: {right} of {len(d)} judged show the named fighter striking, "
          f"{other} only the other fighter, {none} no strike")  # fmt: skip
    nd = v[(v["kind"] == "not_detected") & (v["verdict"] != "cant_judge")]
    print(f"not detected: {(nd['verdict'] == 'none').sum()} of {len(nd)} judged show no strike")


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(Path(sys.argv[2]))
    else:
        score(Path(sys.argv[2]), Path(sys.argv[3]))
