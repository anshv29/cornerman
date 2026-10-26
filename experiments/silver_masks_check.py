"""Mask rules (tangle stage) against the silver set, per clip, 2026-10-26.

    uv run python experiments/silver_masks_check.py

For every silver clip (and, reported separately, every lower-tier clip) in a fight
with tangle output: the tangle stage's frames inside the clip.
- Posture: the majority call over frames with a touching pair, against the clip's
  state family (neutral, clinch and cage are standing; ground is ground).
- Top, on ground clips: each frame's top mask is matched to red or blue through the
  tracked boxes with corner roles (box IoU at least 0.5); the clip's call is the
  majority over frames where a colour was matched, against silver's top fighter.
Silver leans to easy clips (DECISIONS.md, silver second build), so these numbers
flatter the rules on hard ones; ViCoS (vicos_masks_check.py) is the other check.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DERIVED = Path("data/derived")
GOLD = Path("data/gold")


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area - inter, 1e-9)


def family(state: str) -> str:
    return "ground" if state.startswith("ground") else "standing"


def silver_top(red: str, blue: str) -> str | None:
    if red == "ground-top" and blue == "ground-bottom":
        return "red"
    if blue == "ground-top" and red == "ground-bottom":
        return "blue"
    return None


def clip_calls(fid: str, clips: pd.DataFrame) -> pd.DataFrame:
    d = DERIVED / fid
    pairs = pd.read_parquet(d / "tangle" / "pairs.parquet")
    inst = pd.read_parquet(d / "masks" / "instances.parquet")
    tracks = pd.read_parquet(
        d / "tracking" / "tracks.parquet", columns=["frame", "track_id", "x1", "y1", "x2", "y2"]
    )
    roles = pd.read_parquet(d / "identity" / "roles.parquet")
    boxes = tracks.merge(roles, on=["frame", "track_id"])
    boxes = boxes[boxes["role"].isin(["red", "blue"])]
    out = []
    for c in clips.itertuples():
        f = pairs[(pairs["t"] >= c.t0) & (pairs["t"] <= c.t1)]
        found = f[f["posture"] != "none"]
        post = found["posture"].mode().iloc[0] if len(found) else None
        colours = []
        for r in found[found["top"] >= 0].itertuples():
            box = inst[(inst["frame"] == r.frame) & (inst["inst"] == r.top)][
                ["x1", "y1", "x2", "y2"]
            ]
            cand = boxes[boxes["frame"] == r.frame]
            if box.empty or cand.empty:
                continue
            iou = _iou(box.to_numpy(float)[0], cand[["x1", "y1", "x2", "y2"]].to_numpy(float))
            if iou.max() >= 0.5:
                colours.append(str(cand["role"].iloc[int(iou.argmax())]))
        top = pd.Series(colours).mode().iloc[0] if colours else None
        if colours and (pd.Series(colours).value_counts() == len(colours) / 2).all():
            top = None  # a tie
        out.append({"clip_id": c.clip_id, "frames": len(f), "pair_frames": len(found),
                    "posture": post, "top": top})  # fmt: skip
    return pd.DataFrame(out)


def main() -> None:
    s = pd.read_parquet(GOLD / "silver.parquet")
    clips = pd.read_parquet(GOLD / "clips.parquet")[["clip_id", "t0", "t1"]]
    s = s.merge(clips, on="clip_id")
    have = [
        f for f in s["fight_id"].unique() if (DERIVED / f / "tangle" / "manifest.json").exists()
    ]
    s = s[s["fight_id"].isin(have)]
    calls = pd.concat([clip_calls(f, g) for f, g in s.groupby("fight_id")], ignore_index=True)
    s = s.merge(calls, on="clip_id")
    print(f"{s.fight_id.nunique()} fights with tangle output")
    for tier, col in (("silver", "silver"), ("lower tier", "prompts_only")):
        t = s[s[col]].copy()
        t["family"] = t["red_final"].map(family)
        t["state"] = np.where(t["red_final"].str.startswith("clinch"), "clinch", t["family"])
        print(f"\n{tier}: {len(t)} clips")
        for st, g in t.groupby("state"):
            has = g[g["posture"].notna()]
            ok = (has["posture"] == has["family"]).mean() if len(has) else float("nan")
            print(f"  {st}: {len(g)} clips, a pair in {len(has)}, posture right {ok:.0%}")
        g = t[t["family"] == "ground"].copy()
        g["truth"] = [
            silver_top(r, b) for r, b in zip(g["red_final"], g["blue_final"], strict=True)
        ]
        g = g[g["truth"].notna()]
        if len(g):
            right = (g["top"] == g["truth"]).sum()
            wrong = (g["top"].notna() & (g["top"] != g["truth"])).sum()
            print(f"  top on {len(g)} ground clips: right {right}, wrong {wrong}, "
                  f"unsure {g['top'].isna().sum()}")  # fmt: skip


if __name__ == "__main__":
    main()
