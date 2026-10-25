"""Spot check of the silver set, judged by the agent by eye.

Ansh doesn't label footage (hard rule 8), so the spot check is done by the
coding agent looking at one contact sheet per clip: four frames spread over the
clip, red and blue boxes already drawn in by the cutter. It is agent-judged and
not blind (the agent sees the label it is checking), so it is a sanity check
on precision, not an unbiased measurement.

The sample is small on purpose (hard rule 9 keeps frames out of the agent's
context except for a few): up to `per_group` clips per state pair and tier,
seeded. A state pair is the clip's (red, blue) states with sides ignored, so
ground-top/ground-bottom is one group whichever corner is on top.

Sheets go to data/gold/spot/ (gitignored, local only) because they are footage.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def pair_group(red: str, blue: str) -> str:
    return " / ".join(sorted((red, blue)))


def sample(w: pd.DataFrame, per_group: int, seed: int = 0) -> pd.DataFrame:
    """Up to per_group clips per (tier, state pair) from gold.silver.silver output."""
    tiers = [("silver", w[w["silver"]]), ("prompts_only", w[w["prompts_only"]])]
    out: list[pd.DataFrame] = []
    for tier, t in tiers:
        t = t.assign(
            tier=tier,
            group=[pair_group(r, b) for r, b in zip(t["red_final"], t["blue_final"], strict=True)],
        )
        for _, g in t.groupby("group"):
            out.append(g.sample(n=min(per_group, len(g)), random_state=seed))
    if not out:
        return pd.DataFrame()
    cols = ["tier", "group", "clip_id", "stratum", "red_final", "blue_final", "rules"]
    return pd.concat(out, ignore_index=True)[cols]


def contact_sheet(clip: Path, out: Path, frames: int = 4, height: int = 240) -> None:
    """A 2x2 grid of `frames` frames spread evenly over the clip, each `height` px high."""
    import cv2  # the gpu extra; imported here so sampling works without it

    from cornerman.perception.video import iter_frames

    images = [f for _, f in iter_frames(clip)]
    if not images:
        raise ValueError(f"no frames in {clip}")
    picks = np.linspace(0, len(images) - 1, frames).round().astype(int)
    tiles: list[np.ndarray] = []
    for i in picks:
        img = images[int(i)]
        width = round(img.shape[1] * height / img.shape[0])
        tiles.append(cv2.resize(img, (width, height)))
    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
