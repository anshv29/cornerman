"""Spot-check sampling: capped per tier and state pair, sides ignored, seeded."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cornerman.gold.spot import contact_sheet, pair_group, sample


def test_pair_group_ignores_sides() -> None:
    assert pair_group("ground-top", "ground-bottom") == pair_group("ground-bottom", "ground-top")


def test_sample_caps_each_tier_and_group() -> None:
    n = 6
    w = pd.DataFrame(
        {
            "clip_id": [f"c{i}" for i in range(3 * n)],
            "stratum": "close",
            "red_final": ["ground-top"] * n + ["ground-bottom"] * n + ["standing-neutral"] * n,
            "blue_final": ["ground-bottom"] * n + ["ground-top"] * n + ["standing-neutral"] * n,
            "rules": None,
            "silver": [True] * (2 * n) + [False] * n,
            "prompts_only": [False] * (2 * n) + [True] * n,
        }
    )
    s = sample(w, per_group=4)
    assert s.groupby(["tier", "group"]).size().to_dict() == {
        ("prompts_only", "standing-neutral / standing-neutral"): 4,
        ("silver", "ground-bottom / ground-top"): 4,
    }
    assert s.equals(sample(w, per_group=4))


def test_contact_sheet_is_a_2x2_grid(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    clip = tmp_path / "c.mp4"
    out = cv2.VideoWriter(str(clip), cv2.VideoWriter.fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):
        out.write(np.full((48, 64, 3), i * 8, np.uint8))
    out.release()
    contact_sheet(clip, tmp_path / "s.jpg", height=24)
    img = cv2.imread(str(tmp_path / "s.jpg"))
    assert img is not None and img.shape[:2] == (48, 64)
