"""Choosing Kaggle batches and importing their outputs."""

import json
import shutil
from pathlib import Path

import pandas as pd

from cornerman.remote.batch import done_fights, import_outputs, select_fights


def _finish(derived: Path, fight: str) -> None:
    (derived / fight / "identity").mkdir(parents=True)
    (derived / fight / "identity" / "manifest.json").write_text("{}")


def test_select_skips_done_and_prefers_repeat_fighters(tmp_path: Path) -> None:
    fights = pd.DataFrame(
        {
            "fight_id": ["ft_a", "ft_b", "ft_c", "ft_d"],
            "fighter_a_id": ["F0001", "F0001", "F0003", "F0005"],
            "fighter_b_id": ["F0002", "F0004", "F0001", "F0006"],
            "url": ["u1", "u2", "u3", "u4"],
        }
    )
    _finish(tmp_path, "ft_a")
    chosen = select_fights(fights, tmp_path, n=2)
    assert chosen["fight_id"].tolist() == ["ft_b", "ft_c"]
    assert done_fights(tmp_path) == {"ft_a"}


def test_import_copies_only_finished_fights(tmp_path: Path) -> None:
    src = tmp_path / "out"
    _finish(src / "derived", "ft_done")
    (src / "derived" / "ft_half" / "tracking").mkdir(parents=True)
    (tmp_path / "c" / "ft_done").mkdir(parents=True)
    (tmp_path / "c" / "ft_done" / "000001_0000010.jpg").write_bytes(b"x")
    (src / "crops").mkdir(parents=True)
    shutil.make_archive(str(src / "crops" / "ft_done"), "zip", tmp_path / "c" / "ft_done")
    (src / "overlays").mkdir()
    (src / "overlays" / "ft_done.mp4").write_bytes(b"x")
    (src / "job_log.json").write_text(json.dumps([{"fight_id": "ft_done"}]))
    data = tmp_path / "data"
    (data / "derived" / "ft_done" / "crop_features").mkdir(parents=True)  # run on the laptop

    done = import_outputs(src, data / "derived", data / "overlays", data / "logs", data / "crops")
    assert done == ["ft_done"]
    assert (data / "derived" / "ft_done" / "identity" / "manifest.json").exists()
    assert not (data / "derived" / "ft_half").exists()
    assert (data / "derived" / "ft_done" / "crop_features").exists()
    assert (data / "overlays" / "ft_done.mp4").exists()
    assert (data / "logs" / "job_log_000.json").exists()
    assert (data / "crops" / "ft_done" / "000001_0000010.jpg").exists()
