"""Data contracts for perception outputs: tracks, identity, roles per box and poses."""

from pathlib import Path
from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from cornerman.perception.identity import FIGHTERS, OTHER, ROLES
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

ROLES_SCHEMA = pa.DataFrameSchema(
    {
        "frame": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "track_id": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "role": pa.Column(str, pa.Check.isin([*ROLES, OTHER])),  # pyright: ignore[reportUnknownMemberType]
    },
    unique=["frame", "track_id"],
    strict=True,
)
"""One role per tracked box. A tracklet can change role (tracker ID switches in tangles)."""


def role_keys(roles: pd.DataFrame) -> list[str]:
    """Join keys for a roles table: per box if it has a frame column, else per tracklet."""
    return ["frame", "track_id"] if "frame" in roles.columns else ["track_id"]


def load_roles(fight_dir: Path) -> pd.DataFrame:
    """Roles for a processed fight: per box (roles.parquet) if the fight has it, else per
    tracklet (identity.parquet, fights processed before identity round 3)."""
    per_box = fight_dir / "identity" / "roles.parquet"
    if per_box.exists():
        return pd.read_parquet(per_box)
    return pd.read_parquet(fight_dir / "identity" / "identity.parquet")


def identity_quality(derived: Path) -> pd.DataFrame:
    """One row per fight with identity round 3 output: fight_id, confidence, flagged, checks."""
    import json

    rows: list[dict[str, object]] = []
    for q in sorted(derived.glob("*/identity/quality.json")):
        data = json.loads(q.read_text())
        rows.append(
            {
                "fight_id": q.parent.parent.name,
                "confidence": float(data["confidence"]),
                "flagged": bool(data["flagged"]),
                **{k: float(v) for k, v in data["checks"].items()},
            }
        )
    return pd.DataFrame(rows, columns=["fight_id", "confidence", "flagged"] if not rows else None)


def fighter_boxes(
    tracks: pd.DataFrame,
    roles: pd.DataFrame,
    live: pd.DataFrame,
    fighting: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """At most one box per fighter per frame, from live shots (and fighting frames, if given).

    `roles` is per box (frame, track_id, role) or per tracklet (track_id, role).
    """
    t = tracks.merge(roles[[*role_keys(roles), "role"]], on=role_keys(roles))
    t = t[t["role"].isin(FIGHTERS) & t["shot_id"].isin(live.loc[live["live"], "shot_id"])]
    if fighting is not None:
        t = t[t["frame"].isin(fighting.loc[fighting["fighting"], "frame"])]
    t = t.sort_values(["frame", "role", "score"], ascending=[True, True, False])
    return t.drop_duplicates(["frame", "role"]).reset_index(drop=True)


def usable_fights(derived: Path) -> list[str]:
    """Fights whose identity isn't flagged. Later models should train and test on these only."""
    q = identity_quality(derived)
    return sorted(q.loc[~q["flagged"], "fight_id"].tolist()) if len(q) else []


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
