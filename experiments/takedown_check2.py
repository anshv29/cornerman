"""Blind by-eye check of the stricter takedown rule (events/actions.py), round 2, 2026-11-14.

    uv run python experiments/takedown_check2.py sample OUT_DIR
    uv run python experiments/takedown_check2.py score OUT_DIR VERDICTS_CSV

Round 1 (takedown_check.py) found 4 of 17 judged rule takedowns real. Ansh then
asked for standing held 2 s before, ground held 2 s after, stable states and a real
height drop. That rule finds only 9 takedowns in the 53 fights, so the check takes:
- takedown: every takedown the rule finds (precision);
- near_miss: standing-to-ground changes that pass everything but the height drop
  (unknown or not enough), up to 12 (does the drop guard throw real ones away?);
- control: 10 moments in the middle of a held standing run.
At most 2 per fight for near misses and controls, seed 1, shuffled, numbered only.
Moments within 10 s of anything shown in round 1 are left out. Same tiles and
verdicts as round 1: "down", "same" or "cant_judge". Agent-judged, blind to the kind,
not to the footage; a sanity check, not an unbiased accuracy.
"""

import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.events import actions as act

sys.path.insert(0, str(Path(__file__).parent))
from takedown_check import _pair, _thumbs, tile

ROUND1 = Path("data/takedown_check/key.csv")
N_NEAR = 12
N_CONTROL = 10
PER_FIGHT = 2
SEED = 1
AWAY_S = 10.0


def pool() -> pd.DataFrame:
    pred = pd.read_parquet("data/states/predictions.parquet")
    seen = pd.read_csv(ROUND1) if ROUND1.exists() else pd.DataFrame(columns=["fight_id", "t"])
    rows: list[dict[str, object]] = []
    for fid, g in pred.groupby("fight_id"):
        fid = str(fid)
        th = _thumbs(fid)
        if th.empty:
            continue
        g = g.sort_values("t").reset_index(drop=True)
        t = g["t"].to_numpy(float)
        old = seen.loc[seen["fight_id"] == fid, "t"].to_numpy(float)

        def add(at: float, kind: str, fid: str = fid, old: np.ndarray = old,
                th: pd.DataFrame = th) -> None:  # fmt: skip
            if len(old) and np.abs(old - at).min() < AWAY_S:
                return
            pr = _pair(th, at)
            if pr:
                rows.append({"fight_id": fid, "t": at, "kind": kind,
                             "before": pr[0], "after": pr[1]})  # fmt: skip

        held = act.held_runs(g["pred_family"].to_numpy(), t)
        for prev, nxt in itertools.pairwise(held):
            if prev.family != "standing" or nxt.family != "ground":
                continue
            before = act._window(g, prev.end, act.LOOK_S, forward=False)
            after = act._window(g, nxt.start, act.LOOK_S, forward=True)
            stable = (
                t[nxt.start] - t[prev.end - 1] <= act.MAX_GAP_S
                and act._family_p(g, before, "standing") >= act.STABLE
                and act._family_p(g, after, "ground") >= act.STABLE
            )
            if not stable:
                continue
            drop = act._height_drop(g, before, after, nxt.start)
            add(float(t[nxt.start]), "takedown" if drop <= act.DROP else "near_miss")
        for h in held:
            if h.family == "standing" and t[h.end - 1] - t[h.start] >= 12:
                add(float((t[h.start] + t[h.end - 1]) / 2), "control")
    return pd.DataFrame(rows)


def sample(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pl = pool()
    print(pl.groupby("kind").size().to_string())
    picks = [pl[pl["kind"] == "takedown"]]
    for kind, n in (("near_miss", N_NEAR), ("control", N_CONTROL)):
        g = pl[pl["kind"] == kind].sample(frac=1, random_state=SEED)
        picks.append(g.groupby("fight_id").head(PER_FIGHT).head(n))
    pick = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
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
