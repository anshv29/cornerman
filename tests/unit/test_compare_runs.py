"""Comparing two perception runs of the same fight."""

from pathlib import Path

import numpy as np
import pandas as pd

from cornerman.eval.compare_runs import compare_fight


def _write(d: Path, track_offset: int, kp_shift: float, drop_box: bool = False) -> None:
    rows = [
        {"frame": f, "t": f / 10, "shot_id": 0, "track_id": tid + track_offset,
         "x1": x, "y1": 0.0, "x2": x + 50, "y2": 100.0, "score": 0.9}
        for f in range(20) for tid, x in ((1, 0.0), (2, 200.0))
    ]  # fmt: skip
    if drop_box:
        rows = rows[1:]
    for stage in ("tracking", "identity", "pose", "shots"):
        (d / stage).mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d / "tracking" / "tracks.parquet")
    pd.DataFrame(
        {"track_id": [1 + track_offset, 2 + track_offset], "role": ["fighter_a", "fighter_b"],
         "n_frames": [20, 20], "distance": [0.0, 0.0]}
    ).to_parquet(d / "identity" / "identity.parquet")  # fmt: skip
    poses = pd.DataFrame(
        [{"frame": f, "t": f / 10, "role": r, "kp": k, "x": 10.0 + k + kp_shift, "y": 5.0,
          "conf": 0.9} for f in range(20) for r in ("fighter_a", "fighter_b") for k in range(17)]
    )  # fmt: skip
    poses.to_parquet(d / "pose" / "poses.parquet")
    pd.DataFrame({"shot_id": [0], "start": [0.0], "end": [2.0]}).to_parquet(
        d / "shots" / "shots.parquet"
    )


def test_identical_runs_agree_even_with_new_track_ids(tmp_path: Path) -> None:
    _write(tmp_path / "a", 0, 0.0)
    _write(tmp_path / "b", 100, 0.0)
    m = compare_fight(tmp_path / "a", tmp_path / "b")
    assert m["matched_frac"] == 1.0
    assert m["same_track_frac"] == 1.0
    assert m["same_role_frac"] == 1.0
    assert m["kp_px_median"] == 0.0


def test_small_differences_are_measured(tmp_path: Path) -> None:
    _write(tmp_path / "a", 0, 0.0)
    _write(tmp_path / "b", 0, 1.5, drop_box=True)
    m = compare_fight(tmp_path / "a", tmp_path / "b")
    assert m["matched_frac"] == np.float64(39 / 40)
    assert m["kp_px_median"] == 1.5
    assert m["kp_within_2px"] == 1.0
