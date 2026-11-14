"""Blind by-eye check of the red-top lean in the state classifier, 2026-11-14.

    uv run python experiments/red_top_check.py sample OUT_DIR
    uv run python experiments/red_top_check.py score OUT_DIR VERDICTS_CSV

Why. Out of fold, the classifier calls red on top in 10% of live moments and blue
in 6%, while the labellers lean the other way (rules 151 red-top votes to 176
blue-top, masks 255 to 289, VLM 149 to 240). Most ground calls are made with no
directional vote at all, and the lean sits in moments where only one fighter is
boxed (usually one box around both bodies): with only red boxed the model says
red-top 1,371 times and blue-top 285; with only blue boxed, blue-top 678 and
red-top 586. Merged crops get the red role 72% of the time. The question is
whether that single box really is the fighter on top (then the lean is real) or
whether a merged box goes red whoever is on top (then it is an identity bias).

Sample. Thumbnails (arena stage, every 5 s) at live moments the classifier puts on
the ground with exactly one fighter boxed: 20 where that box is red, 20 where it
is blue, at most 2 per fight, seed 0, shuffled, numbered only. The box is drawn in
white so its role isn't shown. The key is read after the verdicts are written.

Verdicts per tile (two columns):
- top: red or blue (the corner colour of the fighter on top, from gloves, shin
  guards and shorts), none (nobody clearly on top, or not on the ground), or
  cant_judge;
- box: top, bottom or both (which bodies the white box covers), or other.
Agent-judged, blind to the role but not to the footage; a sanity check, not an
unbiased accuracy.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.contracts.perception import fighter_boxes

DERIVED = Path("data/derived")
THUMBS = Path("data/thumbs")
PREDICTIONS = Path("data/states/predictions.parquet")
N_EACH = 20
PER_FIGHT = 2
SCALE = 2.0


def pool() -> pd.DataFrame:
    p = pd.read_parquet(PREDICTIONS)
    p = p[p["live"].astype(bool) & (p["pred_family"] == "ground") & (p["n_fighter_boxes"] == 1)]
    red = p["red_speed"].notna() & p["blue_speed"].isna()
    blue = p["blue_speed"].notna() & p["red_speed"].isna()
    p = p.assign(box=np.where(red, "red", np.where(blue, "blue", "")))
    p = p[p["box"] != ""]
    have = {
        (f.parent.name, int(f.stem)) for fid in p["fight_id"].unique()
        for f in (THUMBS / fid).glob("*.jpg")
    }  # fmt: skip
    keep = [(f, int(fr)) in have for f, fr in zip(p["fight_id"], p["frame"], strict=True)]
    return p.loc[keep, ["fight_id", "frame", "box", "pred"]].reset_index(drop=True)


def draw(fid: str, frame: int, n: int) -> np.ndarray:
    video = json.loads((DERIVED / fid / "ingest" / "video.json").read_text())
    w, h = float(video["width"]), float(video["height"])
    img = cv2.imread(str(THUMBS / fid / f"{frame:07d}.jpg"))
    img = cv2.resize(img, (int(320 * SCALE), int(180 * SCALE)))
    sx, sy = img.shape[1] / w, img.shape[0] / h
    tracks = pd.read_parquet(DERIVED / fid / "tracking" / "tracks.parquet")
    tracks = tracks[tracks["frame"] == frame]
    roles = pd.read_parquet(DERIVED / fid / "identity" / "roles.parquet")
    live = pd.read_parquet(DERIVED / fid / "tracking" / "live.parquet")
    for r in fighter_boxes(tracks, roles, live).itertuples():
        cv2.rectangle(img, (int(r.x1 * sx), int(r.y1 * sy)), (int(r.x2 * sx), int(r.y2 * sy)),
                      (255, 255, 255), 2)  # fmt: skip
    cv2.putText(img, str(n), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return img


def sample(out: Path, seed: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pl = pool()
    print(pl.groupby("box").size().to_string())
    picks = []
    for _, g in pl.groupby("box"):
        g = g.sample(frac=1, random_state=seed).groupby("fight_id").head(PER_FIGHT)
        picks.append(g.head(N_EACH))
    pick = pd.concat(picks).sample(frac=1, random_state=seed).reset_index(drop=True)
    pick.insert(0, "n", range(1, len(pick) + 1))
    pick.to_csv(out / "key.csv", index=False)
    tiles = [draw(r.fight_id, r.frame, r.n) for r in pick.itertuples()]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 2)
    rows_img = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    for k in range(0, len(rows_img), 2):
        cv2.imwrite(str(out / f"page_{k // 2:02d}.jpg"), np.vstack(rows_img[k : k + 2]))


def score(out: Path, verdicts: Path) -> None:
    v = pd.read_csv(verdicts).merge(pd.read_csv(out / "key.csv"), on="n")
    print(pd.crosstab(v["box_y"] if "box_y" in v else v["box"], v["top"]).to_string())
    k = "box_y" if "box_y" in v else "box"
    judged = v[v["top"].isin(["red", "blue"])]
    for colour in ("red", "blue"):
        g = judged[judged[k] == colour]
        print(f"{colour} box: top is {colour} in {(g['top'] == colour).sum()} of {len(g)} judged")
    if "box_x" in v:
        print(pd.crosstab(v["box_y"], v["box_x"]).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(Path(sys.argv[2]))
    else:
        score(Path(sys.argv[2]), Path(sys.argv[3]))
