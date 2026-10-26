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
    no_hold = tmp_path / "no_hold.csv"
    assert usable_fights(tmp_path, no_hold) == ["ft_good"]
    assert population_fights(tmp_path, no_hold) == ["ft_bad", "ft_good"]
    assert usable_fights(tmp_path / "none", no_hold) == []

    # A fight on hold (age not confirmed) is left out of both.
    hold = tmp_path / "hold.csv"
    hold.write_text("fight_id,reason\nft_good,division not shown\n")
    assert usable_fights(tmp_path, hold) == []
    assert population_fights(tmp_path, hold) == ["ft_bad"]


def test_usable_frames_keep_clean_shots_of_flagged_fights(tmp_path: Path) -> None:
    from cornerman.contracts.perception import usable_frames

    fighting = pd.DataFrame({"frame": range(0, 100, 5)}).assign(fighting=lambda d: d["frame"] < 90)
    segments = pd.DataFrame(
        {"shot_id": [0, 1], "start": [0, 50], "end": [45, 85], "usable": [True, False]}
    )
    for fid, conf in (("ft_good", 0.7), ("ft_bad", 0.3), ("ft_old", 0.3)):
        d = tmp_path / fid / "identity"
        d.mkdir(parents=True)
        (d / "quality.json").write_text(
            json.dumps({"confidence": conf, "flagged": conf < 0.5, "checks": {}})
        )
        fighting.to_parquet(d / "fighting.parquet")
        if fid != "ft_old":  # ft_old has no segments yet: nothing of it is usable
            (tmp_path / fid / "identity_segments").mkdir()
            segments.to_parquet(tmp_path / fid / "identity_segments" / "segments.parquet")
    frames = usable_frames(tmp_path, tmp_path / "no_hold.csv")
    assert sorted(frames) == ["ft_bad", "ft_good"]
    assert frames["ft_good"] == set(range(0, 90, 5))  # unflagged: all fighting frames
    assert frames["ft_bad"] == set(range(0, 50, 5))  # flagged: the usable shot only
    hold = tmp_path / "hold.csv"
    hold.write_text("fight_id,reason\nft_bad,division not shown\n")
    assert sorted(usable_frames(tmp_path, hold)) == ["ft_good"]
