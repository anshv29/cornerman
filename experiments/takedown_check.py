"""Blind by-eye check of takedowns read off state changes (events/actions.py), 2026-11-14.

    uv run python experiments/takedown_check.py sample OUT_DIR
    uv run python experiments/takedown_check.py score OUT_DIR VERDICTS_CSV

Only thumbnails (arena stage, every 5 s) are left for most fight time, so each tile
pairs the thumbnail just before a moment (0.5 to 5 s before) with the one just after
(0.5 to 5 s after):
- takedown: 20 takedowns the rules found (standing or clinch to ground);
- control: 20 moments in the middle of a held standing position, which should show
  no change to the ground.
At most 2 per fight, seed 0, shuffled, numbered only. The key is read after the
verdicts are written.

Verdict per tile: "down" (standing or clinch in the first frame, ground in the
second), "same" (no change to the ground), or cant_judge. Agent-judged, blind to the
set, not to the footage; a sanity check, not an unbiased accuracy. A thumbnail up to
5 s away can miss a quick takedown and stand-up, so "same" on a real takedown is
possible.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.events.actions import held_runs

THUMBS = Path("data/thumbs")
N_EACH = 20
PER_FIGHT = 2
SCALE = 1.5


def _thumbs(fid: str) -> pd.DataFrame:
    frames = sorted(int(f.stem) for f in (THUMBS / fid).glob("*.jpg"))
    m = pd.read_parquet(Path("data/derived") / fid / "tracking" / "frames.parquet")
    t = m.set_index("frame")["t"]
    return pd.DataFrame({"frame": frames, "t": [float(t.get(f, np.nan)) for f in frames]}).dropna()


def _pair(th: pd.DataFrame, t: float) -> tuple[int, int] | None:
    before = th[(th["t"] <= t - 0.5) & (th["t"] >= t - 5)]
    after = th[(th["t"] >= t + 0.5) & (th["t"] <= t + 5)]
    if before.empty or after.empty:
        return None
    return int(before["frame"].iloc[-1]), int(after["frame"].iloc[0])


def pool() -> pd.DataFrame:
    pred = pd.read_parquet("data/states/predictions.parquet")
    acts = pd.read_parquet("data/states/actions.parquet")
    rows: list[dict[str, object]] = []
    for fid, g in pred.groupby("fight_id"):
        fid = str(fid)
        th = _thumbs(fid)
        if th.empty:
            continue
        g = g.sort_values("t").reset_index(drop=True)
        for t in acts.loc[(acts["fight_id"] == fid) & (acts["action"] == "shoot-takedown"), "t"]:
            pr = _pair(th, float(t))
            if pr:
                rows.append({"fight_id": fid, "t": float(t), "kind": "takedown",
                             "before": pr[0], "after": pr[1]})  # fmt: skip
        tt = g["t"].to_numpy(float)
        for h in held_runs(g["pred_family"].to_numpy(), tt):
            if h.family != "standing" or tt[h.end - 1] - tt[h.start] < 12:
                continue
            mid = float((tt[h.start] + tt[h.end - 1]) / 2)
            pr = _pair(th, mid)
            if pr:
                rows.append({"fight_id": fid, "t": mid, "kind": "control",
                             "before": pr[0], "after": pr[1]})  # fmt: skip
    return pd.DataFrame(rows)


def tile(fid: str, a: int, b: int, n: int) -> np.ndarray:
    imgs = []
    for f in (a, b):
        img = cv2.imread(str(THUMBS / fid / f"{f:07d}.jpg"))
        imgs.append(cv2.resize(img, (int(320 * SCALE), int(180 * SCALE))))
    out = np.hstack(imgs)
    cv2.putText(out, str(n), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
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
    tiles = [tile(r.fight_id, r.before, r.after, r.n) for r in pick.itertuples()]
    for k in range(0, len(tiles), 4):
        cv2.imwrite(str(out / f"page_{k // 4:02d}.jpg"), np.vstack(tiles[k : k + 4]))


def score(out: Path, verdicts: Path) -> None:
    v = pd.read_csv(verdicts).merge(pd.read_csv(out / "key.csv"), on="n")
    print(pd.crosstab(v["kind"], v["verdict"]).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(Path(sys.argv[2]))
    else:
        score(Path(sys.argv[2]), Path(sys.argv[3]))
