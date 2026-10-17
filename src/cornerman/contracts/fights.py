"""Data contracts for the local fights table and fighter registry."""

import pandas as pd
import pandera.pandas as pa

_FIGHTER_ID = r"^F\d{4,}$"


def _fighters_differ(df: pd.DataFrame) -> pd.Series:
    return df["fighter_a_id"] != df["fighter_b_id"]


FIGHTS_SCHEMA = pa.DataFrameSchema(
    {
        "fight_id": pa.Column(str, pa.Check.str_matches(r"^ft_[0-9a-f]{10}$"), unique=True),  # pyright: ignore[reportUnknownMemberType]
        "source": pa.Column(str),
        "video_id": pa.Column(str),
        "url": pa.Column(str, pa.Check.str_startswith("https://")),  # pyright: ignore[reportUnknownMemberType]
        "event": pa.Column(str),
        "year": pa.Column("Int64", pa.Check.in_range(1990, 2100), nullable=True),  # pyright: ignore[reportUnknownMemberType]
        "meeting": pa.Column(int, pa.Check.ge(1)),  # pyright: ignore[reportUnknownMemberType]
        "fighter_a_id": pa.Column(str, pa.Check.str_matches(_FIGHTER_ID)),  # pyright: ignore[reportUnknownMemberType]
        "fighter_b_id": pa.Column(str, pa.Check.str_matches(_FIGHTER_ID)),  # pyright: ignore[reportUnknownMemberType]
        "fighter_a_name": pa.Column(str),
        "fighter_b_name": pa.Column(str),
        "weight_class": pa.Column(object, nullable=True),
        "result": pa.Column(object, nullable=True),
        "method": pa.Column(object, nullable=True),
        "round": pa.Column(object, nullable=True),
        "time": pa.Column(object, nullable=True),
        "height": pa.Column(int, pa.Check.ge(480)),  # pyright: ignore[reportUnknownMemberType]
        "duration_s": pa.Column(float, pa.Check.gt(0)),  # pyright: ignore[reportUnknownMemberType]
        "upload_date": pa.Column(str),
        "title": pa.Column(str),
    },
    checks=[pa.Check(_fighters_differ, error="a fighter can't fight themselves")],
    strict=True,
    ordered=True,
    coerce=True,
)

FIGHTERS_SCHEMA = pa.DataFrameSchema(
    {
        "fighter_id": pa.Column(str, pa.Check.str_matches(_FIGHTER_ID), unique=True),  # pyright: ignore[reportUnknownMemberType]
        "name": pa.Column(str),
        "name_key": pa.Column(str, unique=True),
        "variants": pa.Column(str),
        "n_fights": pa.Column(int, pa.Check.ge(1)),  # pyright: ignore[reportUnknownMemberType]
    },
    strict=True,
    ordered=True,
    coerce=True,
)
