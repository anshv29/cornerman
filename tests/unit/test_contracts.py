import json
from pathlib import Path

import pandas as pd
import pytest

from cornerman.contracts.events import compatibility_violations, validate_events
from cornerman.pipeline.stage import QuarantineError


def good_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fight_id": ["x"] * 4,
            "t": [0.0, 0.0, 1.5, 1.5],
            "fighter": ["F0001", "F0002", "F0001", "F0002"],
            "state": ["standing-neutral", "standing-neutral", "ground-top", "ground-bottom"],
            "momentum": [False, False, True, False],
            "action": ["strike", None, "pass-guard", None],
            "landed": pd.array([True, None, None, None], dtype="boolean"),
            "confidence": [0.9, 0.8, 0.7, 0.7],
            "source": ["stub"] * 4,
        }
    )


def test_good_table_passes() -> None:
    validate_events(good_events())


def test_incompatible_states_fail() -> None:
    df = good_events()
    df.loc[3, "state"] = "ground-top"
    assert len(compatibility_violations(df)) == 1
    with pytest.raises(QuarantineError, match="compatible"):
        validate_events(df)


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("state", "dancing", "isin"),
        ("action", "taunt", "isin"),
        ("confidence", 1.5, "in_range"),
        ("fighter", "Real Name", "str_matches"),
    ],
)
def test_bad_values_fail(column: str, value: str | float, reason: str) -> None:
    df = good_events()
    df.loc[0, column] = value
    with pytest.raises(QuarantineError, match=reason):
        validate_events(df)


def test_landed_on_non_strike_fails() -> None:
    df = good_events()
    df.loc[2, "landed"] = True
    with pytest.raises(QuarantineError, match="landed"):
        validate_events(df)


def test_unsorted_time_fails() -> None:
    df = good_events().iloc[[2, 3, 0, 1]].reset_index(drop=True)
    with pytest.raises(QuarantineError, match="sorted"):
        validate_events(df)


def test_all_failures_reported_together() -> None:
    df = good_events()
    df.loc[0, "confidence"] = -1.0
    df.loc[1, "source"] = "guess"
    with pytest.raises(QuarantineError) as e:
        validate_events(df)
    assert "confidence" in str(e.value)
    assert "source" in str(e.value)


def test_usable_and_population_fights(tmp_path: Path) -> None:
    from cornerman.contracts.perception import population_fights, usable_fights

    for fid, conf in (("ft_good", 0.7), ("ft_bad", 0.3)):
        (tmp_path / fid / "identity").mkdir(parents=True)
        (tmp_path / fid / "identity" / "quality.json").write_text(
            json.dumps({"confidence": conf, "flagged": conf < 0.5, "checks": {"dup": 0.0}})
        )
    assert usable_fights(tmp_path) == ["ft_good"]
    assert population_fights(tmp_path) == ["ft_bad", "ft_good"]
    assert usable_fights(tmp_path / "none") == []
