"""Golden test: the fixture fight must produce exactly the checked-in event log.

If a change to a stage is meant to change the output, regenerate the file with
`CORNERMAN_UPDATE_GOLDEN=1 uv run pytest tests/golden` and review the diff.
"""

import os
from pathlib import Path

import pandas as pd

from cornerman.config import load_settings
from cornerman.pipeline.runner import Status, run, stage_dir
from cornerman.pipeline.stages import PIPELINE

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "fixture_events.csv"


def _as_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, float_format="%.6f", lineterminator="\n")


def test_fixture_fight_matches_golden(tmp_path: Path) -> None:
    settings = load_settings(ROOT / "configs" / "fixture.yaml")
    settings.paths.fixtures = ROOT / "data" / "fixtures"
    settings.paths.derived = tmp_path / "derived"
    settings.paths.quarantine = tmp_path / "quarantine"

    first = run(PIPELINE, settings, ["fixture_fight"])
    assert {r.status for r in first} == {Status.RAN}
    second = run(PIPELINE, settings, ["fixture_fight"])
    assert {r.status for r in second} == {Status.CACHED}

    events = pd.read_parquet(stage_dir(settings, "fixture_fight", "events") / "events.parquet")
    actual = _as_csv(events)
    if os.environ.get("CORNERMAN_UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(actual)
    assert actual == GOLDEN.read_text()
