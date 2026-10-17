"""Data contract for per-fight event tables (`events.parquet`).

One row per fighter per moment. Column checks catch bad values; table checks
catch things that are only wrong in combination, like two fighters both on
top at the same instant.
"""

from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from cornerman.pipeline.stage import QuarantineError
from cornerman.taxonomy import Action, State, compatible

SOURCES = ("stub", "rules", "vlm", "skeleton", "aggregate")


def compatibility_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Moments where both fighters have a state and the pair is impossible."""
    a, b = sorted(df["fighter"].unique())[:2]
    left = df.loc[df["fighter"] == a, ["t", "state"]]
    right = df.loc[df["fighter"] == b, ["t", "state"]]
    both = left.merge(right, on="t", suffixes=("_a", "_b"))
    ok = [
        compatible(State(x), State(y))
        for x, y in zip(both["state_a"], both["state_b"], strict=True)
    ]
    return both.loc[[not v for v in ok]]


def _states_compatible(df: pd.DataFrame) -> bool:
    return compatibility_violations(df).empty


def _landed_only_on_strikes(df: pd.DataFrame) -> pd.Series:
    return df["landed"].isna() | (df["action"] == Action.STRIKE.value)


def _one_fight(df: pd.DataFrame) -> bool:
    return df["fight_id"].nunique() == 1


def _two_fighters(df: pd.DataFrame) -> bool:
    return df["fighter"].nunique() == 2


def _time_sorted(df: pd.DataFrame) -> bool:
    return bool(df["t"].is_monotonic_increasing)


EVENT_SCHEMA = pa.DataFrameSchema(
    {
        "fight_id": pa.Column(str, pa.Check.str_length(min_value=1)),  # pyright: ignore[reportUnknownMemberType]
        "t": pa.Column(float, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "fighter": pa.Column(str, pa.Check.str_matches(r"^F\d{4,}$")),  # pyright: ignore[reportUnknownMemberType]
        "state": pa.Column(str, pa.Check.isin([s.value for s in State])),  # pyright: ignore[reportUnknownMemberType]
        "momentum": pa.Column(bool),
        "action": pa.Column(str, pa.Check.isin([a.value for a in Action]), nullable=True),  # pyright: ignore[reportUnknownMemberType]
        "landed": pa.Column("boolean", nullable=True),
        "confidence": pa.Column(float, pa.Check.in_range(0, 1)),  # pyright: ignore[reportUnknownMemberType]
        "source": pa.Column(str, pa.Check.isin(SOURCES)),  # pyright: ignore[reportUnknownMemberType]
    },
    checks=[
        pa.Check(_one_fight, error="one fight per table"),
        pa.Check(_two_fighters, error="exactly two fighters"),
        pa.Check(_time_sorted, error="rows sorted by t"),
        pa.Check(_landed_only_on_strikes, error="landed is only set on strikes"),
        pa.Check(_states_compatible, error="both fighters' states are compatible"),
    ],
    strict=True,
    ordered=True,
)


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an event table, raising QuarantineError with every failure listed."""
    try:
        return EVENT_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as e:
        cases = cast(pd.DataFrame, e.failure_cases)  # pyright: ignore[reportUnknownMemberType]
        failed = cases[["column", "check"]].drop_duplicates()
        reasons = "; ".join(
            f"{c}: {k}" for c, k in zip(failed["column"], failed["check"], strict=True)
        )
        raise QuarantineError(f"event contract failed: {reasons}") from e
