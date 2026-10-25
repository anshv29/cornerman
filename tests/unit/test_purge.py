"""Purging fights removes every local trace and nothing else."""

import json
from pathlib import Path

import pandas as pd

from cornerman.pipeline.purge import purge


def _tree(root: Path) -> None:
    for fid in ("ft_gone", "ft_keep"):
        (root / "data/derived" / fid / "pose").mkdir(parents=True)
        (root / "data/crops" / fid).mkdir(parents=True)
        (root / "data/identity_cache").mkdir(parents=True, exist_ok=True)
        (root / "data/identity_cache" / f"{fid}.npz").write_bytes(b"x")
        (root / "data/gold/clips").mkdir(parents=True, exist_ok=True)
        (root / "data/gold/clips" / f"{fid}_000100.mp4").write_bytes(b"x")
    clips = pd.DataFrame({"clip_id": ["ft_gone_000100", "ft_keep_000100"],
                          "fight_id": ["ft_gone", "ft_keep"]})  # fmt: skip
    clips.to_parquet(root / "data/gold/clips.parquet")
    (root / "data/gold/vlm").mkdir(parents=True)
    (root / "data/gold/vlm/labels_000.jsonl").write_text(
        '{"clip_id": "ft_gone_000100"}\n{"clip_id": "ft_keep_000100"}\n'
    )
    (root / "data/logs").mkdir(parents=True)
    log = {"gpus": 2, "fights": [{"fight_id": "ft_gone"}, {"fight_id": "ft_keep"}]}
    (root / "data/logs/job_log_000.json").write_text(json.dumps(log))


def test_purge_removes_only_the_given_fights(tmp_path: Path) -> None:
    _tree(tmp_path)
    n = purge({"ft_gone"}, tmp_path)
    assert n == {"folders": 2, "files": 1, "clips": 1, "rows": 3}
    left = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if "gone" in p.name)
    assert left == []
    assert (tmp_path / "data/derived/ft_keep/pose").is_dir()
    assert pd.read_parquet(tmp_path / "data/gold/clips.parquet")["fight_id"].tolist() == ["ft_keep"]
    assert "ft_keep" in (tmp_path / "data/gold/vlm/labels_000.jsonl").read_text()
    log = json.loads((tmp_path / "data/logs/job_log_000.json").read_text())
    assert log["fights"] == [{"fight_id": "ft_keep"}] and log["gpus"] == 2
    assert purge({"ft_gone"}, tmp_path) == {"folders": 0, "files": 0, "clips": 0, "rows": 0}
