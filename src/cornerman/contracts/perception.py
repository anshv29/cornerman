"""Data contracts for perception outputs: tracks, identity, roles per box and poses."""

from pathlib import Path
from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from cornerman.contracts.hold import HOLD_CSV, held_fights
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


HULL_SCHEMA = pa.DataFrameSchema(
    {
        "frame": pa.Column(int, pa.Check.ge(0), unique=True),  # pyright: ignore[reportUnknownMemberType]
        "shot_id": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "x": pa.Column(object),
        "y": pa.Column(object),
    },
    strict=True,
)
"""The mat outline per sampled frame in a shot (arena stage): hull corners as fractions of
the frame width (x) and height (y). Empty lists where no mat was found."""

CLOCK_SCHEMA = pa.DataFrameSchema(
    {
        "frame": pa.Column(int, pa.Check.ge(0), unique=True),  # pyright: ignore[reportUnknownMemberType]
        "t": pa.Column(float, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "clock_s": pa.Column(float, pa.Check.in_range(0, 600), nullable=True),  # pyright: ignore[reportUnknownMemberType]
        "round": pa.Column("Int64", pa.Check.in_range(1, 9), nullable=True),  # pyright: ignore[reportUnknownMemberType]
    },
    strict=True,
)
"""The score bar's round clock and round (arena stage), None where not read."""


def _ordered_segment(df: pd.DataFrame) -> pd.Series:
    return df["end"] >= df["start"]


SEGMENTS_SCHEMA = pa.DataFrameSchema(
    {
        "shot_id": pa.Column(int, pa.Check.ge(0), unique=True),  # pyright: ignore[reportUnknownMemberType]
        "start": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "end": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "fight_s": pa.Column(float, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "n_red": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "n_blue": pa.Column(int, pa.Check.ge(0)),  # pyright: ignore[reportUnknownMemberType]
        "both": pa.Column(float, pa.Check.in_range(0, 1)),  # pyright: ignore[reportUnknownMemberType]
        "dup": pa.Column(float, pa.Check.in_range(0, 1)),  # pyright: ignore[reportUnknownMemberType]
        "rb_share": pa.Column(float, pa.Check.in_range(0, 1)),  # pyright: ignore[reportUnknownMemberType]
        "closest": pa.Column(float, nullable=True),
        "score": pa.Column(float, pa.Check.in_range(0, 1)),  # pyright: ignore[reportUnknownMemberType]
        "usable": pa.Column(bool),
    },
    checks=[pa.Check(_ordered_segment, error="segments end after they start")],
    strict=True,
)
"""Identity confidence per shot (identity_segments stage): one row per shot with fighting."""


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


def usable_fights(derived: Path, hold: Path = HOLD_CSV) -> list[str]:
    """Fights whose identity isn't flagged and that aren't on hold (contracts.hold): the
    only ones for anything per fighter (fighter models, scouting reports, the gold set)."""
    q = identity_quality(derived)
    if not len(q):
        return []
    q = q[~q["fight_id"].isin(held_fights(hold))]
    return sorted(q.loc[~q["flagged"], "fight_id"].tolist())


def fighting_table(fight_dir: Path) -> Path:
    """The fight's fighting-time table: v2 (clock, crowd) from the fighting stage once it
    has run, else identity's motion-only one."""
    v2 = fight_dir / "fighting" / "fighting.parquet"
    return v2 if v2.exists() else fight_dir / "identity" / "fighting.parquet"


def usable_frames(derived: Path, hold: Path = HOLD_CSV) -> dict[str, set[int]]:
    """Fighting frames whose identity can be used for anything per fighter, per fight.

    An unflagged fight gives all its fighting frames in live shots; a flagged fight gives only the
    frames of its usable segments (identity_segments), so its clean shots aren't lost.
    Fights on hold give nothing.
    """
    q = identity_quality(derived)
    held = held_fights(hold)
    out: dict[str, set[int]] = {}
    for fid, flagged in zip(
        q["fight_id"].tolist() if len(q) else [],
        q["flagged"].tolist() if len(q) else [],
        strict=True,
    ):
        if fid in held:
            continue
        fighting = pd.read_parquet(fighting_table(derived / fid))
        keep = fighting["fighting"]
        if "live" in fighting:  # v2 counts round time in shots without pose; never per fighter
            keep = keep & fighting["live"]
        frames = fighting.loc[keep, "frame"]
        if flagged:
            seg_path = derived / fid / "identity_segments" / "segments.parquet"
            if not seg_path.exists():
                continue
            seg = pd.read_parquet(seg_path)
            seg = seg[seg["usable"]]
            inside = pd.Series(False, index=frames.index)
            for start, end in zip(seg["start"].tolist(), seg["end"].tolist(), strict=True):
                inside |= frames.between(start, end)
            frames = frames[inside]
        if len(frames):
            out[str(fid)] = set(frames.astype(int).tolist())
    return out


def population_fights(derived: Path, hold: Path = HOLD_CSV) -> list[str]:
    """Every fight with identity output, flagged or not, except those on hold. Only for
    population-level numbers where who is red and who is blue doesn't matter (for
    example how often fights reach the ground); never for anything per fighter."""
    q = identity_quality(derived)
    held = held_fights(hold)
    return sorted(f for f in q["fight_id"].tolist() if f not in held) if len(q) else []


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
