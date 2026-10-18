"""Data contracts for perception outputs: tracks, identity and poses."""

from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from cornerman.perception.identity import OTHER, ROLES
from cornerman.pipeline.stage import QuarantineError

_BOX = {c: pa.Column(float) for c in ("x1", "y1", "x2", "y2")}


def _box_ordered(df: pd.DataFrame) -> pd.Series:
    return (df["x2"] > df["x1"]) & (df["y2"] > df["y1"])


TRACKS_SCHEMA = pa.DataFrameSchema(
    {
        "frame": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "t": pa.Column(float, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "shot_id": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "track_id": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        **_BOX,
        "score": pa.Column(float, pa.Check.in_range(0, 1)),  # pyright: ignore[reportUnknownMemberType]
    },
    checks=[pa.Check(_box_ordered, error="boxes have positive size")],
    strict=False,  # plus the app_* feature columns
)

IDENTITY_SCHEMA = pa.DataFrameSchema(
    {
        "track_id": pa.Column(int, unique=True),
        "role": pa.Column(str, pa.Check.isin([*ROLES, OTHER])),  # pyright: ignore[reportUnknownMemberType]
        "n_frames": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "distance": pa.Column(float, nullable=True),
    },
    strict=True,
)

POSES_SCHEMA = pa.DataFrameSchema(
    {
        "frame": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "t": pa.Column(float, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "track_id": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "kp": pa.Column(int, pa.Check.in_range(0, 16)),  # pyright: ignore[reportUnknownMemberType]
        "x": pa.Column(float),
        "y": pa.Column(float),
        "conf": pa.Column(float, pa.Check.in_range(0, 1)),  # pyright: ignore[reportUnknownMemberType]
    },
    unique=["frame", "track_id", "kp"],
    strict=True,
)


def validate(schema: pa.DataFrameSchema, df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate a table, raising QuarantineError with every failure listed."""
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as e:
        cases = cast(pd.DataFrame, e.failure_cases)  # pyright: ignore[reportUnknownMemberType]
        failed = cases[["column", "check"]].drop_duplicates()
        reasons = "; ".join(
            f"{c}: {k}" for c, k in zip(failed["column"], failed["check"], strict=True)
        )
        raise QuarantineError(f"{name} contract failed: {reasons}") from e
