"""By-eye check of identity per shot (identity_segments), blind to the rule's decision.

Round 1 (dev), rule v1 committed first: a shot of a flagged fight is usable if
it has at least 3 s of fighting and its per-shot confidence is at least 0.5.
Sampled 30 shots the rule keeps and 15 it rejects (seed 0). Its verdicts led
to rule v2, which also needs both fighters boxed in at least 30% of the shot's
fighting frames.

Round 2 (test), rule v2 committed first: 30 shots v2 keeps and 10 that v1 kept
but v2 rejects (seed 1), none of them from round 1.

Both rounds shuffle the shots and draw one block per shot, labelled only by its
sample number:

- ref red, ref blue: crops of the red and blue corner from the fight's
  highest-scoring other shot, so a swap within the shot can be seen;
- red, blue: crops of the shot itself (fighting time, clean crops only: body
  found and not a merged box).

The key (which shots the rule keeps) is written to a separate file, read only
after every block is judged. Judging rule, fixed with the sampling: a shot is
right if in its red row and its blue row at least 3 of 4 crops (6 of 8) are one
person, the two people differ, neither is an official, and the red person is
the ref red person (no swap). A row with fewer than 3 crops can't be judged.

    uv run python experiments/identity_segments_check.py sample OUT_DIR          # round 1
    uv run python experiments/identity_segments_check.py sample2 OUT_DIR ROUND1_KEY
    uv run python experiments/identity_segments_check.py score OUT_DIR VERDICTS_CSV

Sheets and the key stay in OUT_DIR (local; crops are footage).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.contracts.perception import identity_quality

DERIVED = Path("data/derived")
N_KEPT, N_REJECTED, N_DROPPED_BY_V2 = 30, 15, 10
V1_FLAG_BELOW = 0.5
MIN_S = 3.0
TILES, REF_TILES = 8, 6
PER_PAGE = 3


def shots() -> pd.DataFrame:
    q = identity_quality(DERIVED)
    out = []
    for fid in q.loc[q["flagged"], "fight_id"]:
        seg = pd.read_parquet(DERIVED / fid / "identity_segments" / "segments.parquet")
        out.append(seg.assign(fight_id=fid))
    s = pd.concat(out, ignore_index=True)
    return s[s["fight_s"] >= MIN_S].reset_index(drop=True)


def crops_of(fid: str) -> pd.DataFrame:
    d = DERIVED / fid
    crops = pd.read_parquet(d / "crop_features" / "crops.parquet")
    roles = pd.read_parquet(d / "identity" / "roles.parquet")
    frames = pd.read_parquet(d / "tracking" / "frames.parquet")[["frame", "shot_id"]]
    fighting = pd.read_parquet(d / "identity" / "fighting.parquet")
    fight = set(fighting.loc[fighting["fighting"], "frame"])
    c = crops.merge(roles, on=["frame", "track_id"]).merge(frames, on="frame")
    return c[c["frame"].isin(fight) & c["has_mask"] & ~c["merged"]]


def row(fid: str, crops: pd.DataFrame, label: str, n: int, seed: int) -> np.ndarray:
    pick = crops.sample(min(n, len(crops)), random_state=seed) if len(crops) else crops
    tiles = [
        cv2.resize(cv2.imread(str(Path("data/crops") / fid / f"{t:06d}_{f:07d}.jpg")), (60, 120))
        for t, f in zip(pick["track_id"], pick["frame"], strict=True)
    ]
    tiles += [np.zeros((120, 60, 3), np.uint8)] * (TILES - len(tiles))
    lab = np.zeros((120, 90, 3), np.uint8)
    cv2.putText(lab, label, (2, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.hstack([lab, *tiles])


def sample(out: Path, pick: pd.DataFrame, s: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pick = pick.reset_index(drop=True)
    pick.insert(0, "n", range(1, len(pick) + 1))
    pick.to_csv(out / "key.csv", index=False)
    blocks: list[np.ndarray] = []
    cache: dict[str, pd.DataFrame] = {}
    for r in pick.itertuples():
        fid = str(r.fight_id)
        c = cache.setdefault(fid, crops_of(fid))
        seg = s[(s["fight_id"] == fid) & (s["shot_id"] != r.shot_id)]
        ref_shot = int(seg.sort_values("score").iloc[-1]["shot_id"])
        ref, mine = c[c["shot_id"] == ref_shot], c[c["shot_id"] == r.shot_id]
        blocks.append(
            np.vstack(
                [
                    row(fid, ref[ref["role"] == "red"], f"{r.n} ref red", REF_TILES, 0),
                    row(fid, ref[ref["role"] == "blue"], f"{r.n} ref blue", REF_TILES, 0),
                    row(fid, mine[mine["role"] == "red"], f"{r.n} red", TILES, 0),
                    row(fid, mine[mine["role"] == "blue"], f"{r.n} blue", TILES, 0),
                ]
            )
        )
    for i in range(0, len(blocks), PER_PAGE):
        sep = np.full((8, blocks[0].shape[1], 3), 255, np.uint8)
        page = np.vstack([x for b in blocks[i : i + PER_PAGE] for x in (b, sep)])
        cv2.imwrite(str(out / f"page_{i // PER_PAGE:02d}.jpg"), page)


def score(out: Path, verdicts_csv: Path) -> None:
    key = pd.read_csv(out / "key.csv")
    v = pd.read_csv(verdicts_csv).merge(key[["n", "usable", "fight_s"]], on="n")
    judged = v[v["verdict"] != "cant_judge"]
    for usable, g in judged.groupby("usable"):
        right = g["verdict"] == "right"
        by_min = g.loc[right, "fight_s"].sum() / g["fight_s"].sum()
        print(
            f"rule {'keeps' if usable else 'rejects'}: {int(right.sum())} of {len(g)} right "
            f"({right.mean():.0%}), by minutes {by_min:.0%}"
        )
    print(f"can't judge: {int((v['verdict'] == 'cant_judge').sum())}")
    print(v.groupby(["usable", "verdict"]).size().to_string())


def round1(out: Path) -> None:
    """Round 1 as it was run, under rule v1 (score and length only)."""
    s = shots()
    v1 = s["score"] >= V1_FLAG_BELOW
    kept = s[v1].sample(N_KEPT, random_state=0)
    rejected = s[~v1].sample(N_REJECTED, random_state=0)
    sample(out, pd.concat([kept, rejected]).sample(frac=1, random_state=0), s)


def round2(out: Path, round1_key: Path) -> None:
    s = shots()
    seen = pd.read_csv(round1_key)[["fight_id", "shot_id"]]
    fresh = s.merge(seen, how="left", indicator=True).query("_merge == 'left_only'")
    fresh = fresh.drop(columns="_merge")
    v1 = fresh["score"] >= V1_FLAG_BELOW
    kept = fresh[fresh["usable"]].sample(N_KEPT, random_state=1)
    dropped = fresh[v1 & ~fresh["usable"]].sample(N_DROPPED_BY_V2, random_state=1)
    sample(out, pd.concat([kept, dropped]).sample(frac=1, random_state=1), s)


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        round1(Path(sys.argv[2]))
    elif sys.argv[1] == "sample2":
        round2(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        score(Path(sys.argv[2]), Path(sys.argv[3]))
