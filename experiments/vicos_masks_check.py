"""Mask rules (perception/tangle.py) against the ViCoS hand labels, 2026-10-26.

    uv run python experiments/vicos_masks_check.py fetch
    uv run python experiments/vicos_masks_check.py run
    uv run python experiments/vicos_masks_check.py report

Ansh (2026-10-26): validate the mask-based top/bottom rule against ViCoS as well as
silver. ViCoS is evaluation only and never uploaded (DECISIONS.md), so the images
are read straight out of the remote zip by range requests and the masks are
computed on the laptop.

fetch: 30 frames per position with both athletes' poses, seed 0 (16 positions).
run: RF-DETR Seg on each image, masks resized to masks_width as in the masks stage,
then the rules with the thresholds in config (fixed before this was run).
report: posture against ViCoS standing/ground; on ground frames with someone on
top, the top rule right, wrong or unsure. Each athlete is matched to a mask by
where its hand-checked keypoints fall. Labeller A (keypoint rules) on the same
frames for comparison, and the visible-area cue as an exploratory number only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cornerman.config import PerceptionConfig
from cornerman.gold.vicos import FOLDER, fetch_images, load, rules_on_frames, sample_frames
from cornerman.perception import models
from cornerman.perception.tangle import fighter_pair, posture, small_mask, top_of

OUT = Path("data/external/vicos_jiujitsu/masks_check")
IMAGES = FOLDER / "images"
PER_POSITION, SEED = 30, 0


def fetch() -> None:
    sample = sample_frames(load(), PER_POSITION, SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    sample.to_pickle(OUT / "sample.pkl")
    got = fetch_images(sample["image"].tolist(), IMAGES)
    print(f"{len(got)} of {len(sample)} images in {IMAGES}")


def _athlete_mask(pose: np.ndarray, masks: list[np.ndarray], scale: float) -> int | None:
    """Which of the masks holds most of this athlete's confident keypoints."""
    ok = pose[:, 2] > 0.3
    if ok.sum() < 3:
        return None
    hits = []
    for m in masks:
        h, w = m.shape
        xs = np.clip((pose[ok, 0] * scale).astype(int), 0, w - 1)
        ys = np.clip((pose[ok, 1] * scale).astype(int), 0, h - 1)
        hits.append(int(m[ys, xs].sum()))
    best = int(np.argmax(hits))
    return best if hits[best] > 0 else None


def run() -> None:
    import cv2

    p = PerceptionConfig()
    sample: pd.DataFrame = pd.read_pickle(OUT / "sample.pkl")
    rows = []
    for r in sample.itertuples():
        path = IMAGES / f"{r.image}.jpg"
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        _, _, full = models.segment_people([img], p.seg_model, p.det_threshold)[0]
        h, w = img.shape[:2]
        keep = [m for m in full if m.any() and np.ptp(np.nonzero(m)[0]) >= p.min_person_height * h]
        small = [small_mask(m, p.masks_width) for m in keep]
        scale = p.masks_width / w
        row: dict[str, object] = {"image": r.image, "position": r.position, "n_masks": len(small)}
        pair = fighter_pair(np.array(small), p.tangle_contact_px) if len(small) >= 2 else None
        if pair is not None:
            a, b = small[pair[0]], small[pair[1]]
            row["posture"] = posture(a, b, p.tangle_ground_ratio)
            top = top_of(a, b, p.tangle_top_margin)
            row["top_mask"] = None if top is None else (0, 1)[top]
            row["area_top"] = 0 if a.sum() > b.sum() else 1  # exploratory
            row["m1"] = _athlete_mask(np.asarray(r.pose1), [a, b], scale)
            row["m2"] = _athlete_mask(np.asarray(r.pose2), [a, b], scale)
        rows.append(row)
        if len(rows) % 50 == 0:
            print(len(rows), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)


def report() -> None:
    sample: pd.DataFrame = pd.read_pickle(OUT / "sample.pkl")
    res = pd.read_csv(OUT / "results.csv", dtype={"image": str})
    a = rules_on_frames(sample)[["image", "level", "truth_top", "rules"]]
    d = res.merge(a, on="image")
    n = len(d)
    has = d[d["posture"].notna()]
    print(f"{n} frames; a touching pair found in {len(has)} ({len(has) / n:.0%})")
    for level, g in d.groupby("level"):
        found = g["posture"].notna().mean()
        ok = (g["posture"] == level).sum() / max(1, g["posture"].notna().sum())
        print(f"  {level}: pair found {found:.0%}; posture right where found {ok:.0%}")
    # Top: ground frames with someone on top, a pair, and the two athletes on different masks.
    g = has[(has["level"] == "ground") & has["truth_top"].notna()]
    matched = g[g["m1"].notna() & g["m2"].notna() & (g["m1"] != g["m2"])]
    truth_mask = np.where(matched["truth_top"] == "red", matched["m1"], matched["m2"])
    said = matched["top_mask"]
    right = (said == truth_mask).sum()
    wrong = (said.notna() & (said != truth_mask)).sum()
    unsure = said.isna().sum()
    k = len(matched)
    print(
        f"top, {k} ground frames with someone on top and both athletes matched to masks "
        f"(of {len(g)} with a pair, {len(d[(d['level'] == 'ground') & d['truth_top'].notna()])} "
        f"in all): right {right / k:.0%}, wrong {wrong / k:.0%}, unsure {unsure / k:.0%}"
    )
    area = (matched["area_top"] == truth_mask).mean()
    print(f"  exploratory, larger visible area = top: right {area:.0%}")
    for pos, gg in matched.assign(t=truth_mask).groupby("position"):
        s = gg["top_mask"]
        print(
            f"    {pos}: {len(gg)} frames, right {(s == gg['t']).mean():.0%}, "
            f"wrong {(s.notna() & (s != gg['t'])).mean():.0%}"
        )
    # Labeller A (keypoints, hand-checked poses) on the same frames, for comparison.
    ga = d[(d["level"] == "ground") & d["truth_top"].notna() & d["rules"].notna()]
    said_a = ga["rules"].str.extract(r"ground-(red|blue)-top")[0]
    print(
        f"labeller A (keypoints) on the same ground frames: right "
        f"{(said_a == ga['truth_top']).mean():.0%}, wrong "
        f"{(said_a.notna() & (said_a != ga['truth_top'])).mean():.0%}"
    )


if __name__ == "__main__":
    {"fetch": fetch, "run": run, "report": report}[sys.argv[1]]()
