"""Checks for fighting time v2 (the fighting stage), 2026-10-26.

    uv run python experiments/fighting_check.py report
    uv run python experiments/fighting_check.py fallback
    uv run python experiments/fighting_check.py sample OUT_DIR
    uv run python experiments/fighting_check.py score OUT_DIR VERDICTS_CSV

report: per fight, fighting minutes from motion only (v1) and v2, how much of
v2 the clock decided, minutes v2 removes before the first and after the last
running clock (walk-out, celebration), round time in shots that aren't live
(fighters merged into one box, so no pose there), and the rounds seen on the clock.

fallback: where the clock can't be read, v2 falls back to motion plus a crowd
check. To see how good that is, take the frames where the clock is known and
pretend it isn't: the clock's answer is the label, the fallback (without the
first/last-running bounds, which need the clock) is the prediction. Fights are
split in two by sorted id: dev (even) and test (odd). The thresholds in config
were set by reasoning before this was run; dev is only used to see whether
another setting is clearly better, and test is reported as it comes.

sample/score: a blind by-eye check on thumbnails (arena stage, every 5 s).
20 thumbnails v2 keeps as fighting and 20 that motion called fighting but v2
drops, shuffled, numbered only; the key is read after the verdicts. A frame is
"fighting" if the two fighters are engaged in a round: not walking out, being
introduced, celebrating, waiting for the decision, or on stools between rounds.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.contracts.perception import population_fights
from cornerman.perception.activity import RUNNING, STOPPED

DERIVED = Path("data/derived")
THUMBS = Path("data/thumbs")
N_EACH = 20


def fights() -> list[str]:
    return [f for f in population_fights(DERIVED) if (DERIVED / f / "fighting").exists()]


def tables(fid: str) -> tuple[pd.DataFrame, float]:
    v2 = pd.read_parquet(DERIVED / fid / "fighting" / "fighting.parquet")
    dt = float(v2["t"].diff().median())
    return v2, dt


def report() -> None:
    rows = []
    for fid in fights():
        v2, dt = tables(fid)
        clock = pd.read_parquet(DERIVED / fid / "arena" / "clock.parquet")
        run_t = v2.loc[v2["clock"] == RUNNING, "t"]
        before = after = 0.0
        if len(run_t):
            gone = v2["motion"] & ~v2["fighting"]
            before = float((gone & (v2["t"] < run_t.min())).sum() * dt / 60)
            after = float((gone & (v2["t"] > run_t.max())).sum() * dt / 60)
        rows.append(
            {
                "fight_id": fid,
                "v1_min": v2["motion"].sum() * dt / 60,
                "v2_min": v2["fighting"].sum() * dt / 60,
                "not_live_min": (v2["fighting"] & ~v2["live"]).sum() * dt / 60,
                "clock_read": clock["clock_s"].notna().mean(),
                "clock_decided": (v2.loc[v2["fighting"], "clock"] != "unknown").mean()
                if v2["fighting"].any()
                else 0.0,
                "walkout_min": before,
                "celebration_min": after,
                "rounds_seen": clock["round"].nunique(),
            }
        )
    r = pd.DataFrame(rows)
    print(r.round(2).to_string(index=False))
    print(
        f"\n{len(r)} fights: v1 {r.v1_min.sum():.1f} min, v2 {r.v2_min.sum():.1f} min; "
        f"of v2 outside live shots (no pose) {r.not_live_min.sum():.1f}; "
        f"walk-outs {r.walkout_min.sum():.1f}, celebrations {r.celebration_min.sum():.1f}; "
        f"clock read in {r.clock_read.median():.0%} of readings (median fight), "
        f"{(r.clock_read > 0.3).sum()} fights above 30%"
    )
    exp = r.rounds_seen.clip(lower=1) * 3
    print(f"v2 minutes / (rounds seen x 3 min), median: {(r.v2_min / exp).median():.2f}")


def fallback(max_people: int, window_s: float) -> None:
    for name, part in (("dev", fights()[0::2]), ("test", fights()[1::2])):
        tp = fp = fn = 0
        for fid in part:
            v2, dt = tables(fid)
            known = v2["clock"].isin([RUNNING, STOPPED])
            if known.sum() < 100:
                continue
            tracks = pd.read_parquet(DERIVED / fid / "tracking" / "tracks.parquet")
            people = v2["frame"].map(tracks.groupby("frame")["track_id"].nunique()).fillna(0)
            w = max(1, round(window_s / dt))
            crowd = people.rolling(w, center=True, min_periods=1).median()
            pred = v2["motion"] & (crowd <= max_people)
            label = v2["clock"] == RUNNING
            tp += int((pred & label & known).sum())
            fp += int((pred & ~label & known).sum())
            fn += int((~pred & label & known).sum())
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        print(
            f"{name}: max_people {max_people}, window {window_s:.0f}s: "
            f"precision {prec:.2f}, recall {rec:.2f} (frames, clock as label)"
        )


def sample(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fid in fights():
        v2, _ = tables(fid)
        have = {int(p.stem): p for p in (THUMBS / fid).glob("*.jpg")}
        v2 = v2[v2["frame"].isin(list(have))]
        for kind, sel in (("kept", v2["fighting"]), ("dropped", v2["motion"] & ~v2["fighting"])):
            for f in v2.loc[sel, "frame"]:
                rows.append({"fight_id": fid, "frame": int(f), "kind": kind})
    pool = pd.DataFrame(rows)
    pick = (
        pd.concat([g.sample(min(N_EACH, len(g)), random_state=0) for _, g in pool.groupby("kind")])
        .sample(frac=1, random_state=0)
        .reset_index(drop=True)
    )
    pick.insert(0, "n", range(1, len(pick) + 1))
    pick.to_csv(out / "key.csv", index=False)
    tiles = []
    for r in pick.itertuples():
        img = cv2.imread(str(THUMBS / r.fight_id / f"{r.frame:07d}.jpg"))
        img = cv2.resize(img, (320, 180))
        cv2.putText(img, str(r.n), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(img)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 2)
    rows_img = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    for k in range(0, len(rows_img), 4):
        cv2.imwrite(str(out / f"page_{k // 4:02d}.jpg"), np.vstack(rows_img[k : k + 4]))


def score(out: Path, verdicts: Path) -> None:
    v = pd.read_csv(verdicts).merge(pd.read_csv(out / "key.csv"), on="n")
    v = v[v["verdict"] != "cant_judge"]
    for kind, g in v.groupby("kind"):
        fighting = (g["verdict"] == "fighting").sum()
        print(f"{kind}: {fighting} of {len(g)} judged fighting")
    print(v.groupby(["kind", "verdict"]).size().to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "report":
        report()
    elif cmd == "fallback":
        for people in (3, 4):
            for window in (5.0, 10.0):
                fallback(people, window)
    elif cmd == "sample":
        sample(Path(sys.argv[2]))
    else:
        score(Path(sys.argv[2]), Path(sys.argv[3]))
