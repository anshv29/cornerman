"""ViCoS jiu-jitsu combat positions, mapped onto our states (evaluation only).

Hudovernik, V. and Skočaj, D. (2022). Video-Based Detection of Combat Positions
and Automatic Scoring in Jiu-jitsu. Proceedings of the ACM Multimedia Workshop
MMSports '22, Lisboa. https://vicos.si/resources/jiujitsu/

Licence CC BY-NC-SA 4.0 (non-commercial, share-alike). Ansh's rule (DECISIONS.md,
2026-10-19): used only to measure our ground-state labelling, never to train a
model that ships, never committed or uploaded. The files live in
data/external/vicos_jiujitsu/, which git ignores.

What it gives us: 120,279 frames of two athletes sparring (6 sequences, 3 phone
cameras, 16 recordings), each with a hand-labelled position and COCO-17 keypoints
for one or both athletes. Only the annotation file is used (keypoints are enough
to test a pose-based labeller); the 10 GB of images aren't downloaded.

Which athlete a class suffix names was checked from the keypoints, not assumed:
in mount, side control, back and turtle the named athlete's hips are higher in
the image (on top) in 74 to 100% of frames; in open, closed and half guard they
are lower in 98 to 100% (the guard player, underneath). Takedowns are moments of
transition, not positions, and are left out. 50-50 guard has no one on top and
maps to ground-scrambling for both. ViCoS has no clinch, cage or strikes, so it
only tests standing versus ground and who is on top.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cornerman.taxonomy import State

ANNOTATIONS_URL = "http://data.vicos.si/datasets/JuiJuitsu/annotations.json"
ANNOTATIONS_SHA256 = "b7633ed161372bef7d0dcdd2cb3bc69400147b6a2532a4a5b8070048a1dd62bc"
FOLDER = Path("data/external/vicos_jiujitsu")

TOP_IS_NAMED = ("mount", "side_control", "back", "turtle")
BOTTOM_IS_NAMED = ("open_guard", "closed_guard", "half_guard")


def states(position: str) -> tuple[State, State] | None:
    """(athlete 1, athlete 2) states for a ViCoS position, or None if it isn't a state."""
    if position == "standing":
        return State.STANDING_NEUTRAL, State.STANDING_NEUTRAL
    if position == "5050_guard":
        return State.GROUND_SCRAMBLING, State.GROUND_SCRAMBLING
    name, who = position[:-1], position[-1]
    if who not in "12" or name == "takedown":
        return None
    named_top = name in TOP_IS_NAMED
    if not named_top and name not in BOTTOM_IS_NAMED:
        raise ValueError(f"unknown ViCoS position {position!r}")
    one_top = named_top == (who == "1")
    top, bottom = State.GROUND_TOP, State.GROUND_BOTTOM
    return (top, bottom) if one_top else (bottom, top)


def _pose(p: object) -> np.ndarray | None:
    return None if p is None else np.asarray(p, dtype=float).reshape(17, 3)


def load(path: Path = FOLDER / "annotations.json") -> pd.DataFrame:
    """One row per frame: recording, frame, position, state1, state2, pose1, pose2.

    Poses are (17, 3) arrays of x, y, confidence, or None where that athlete has no
    pose. Takedown frames are dropped.
    """
    import json

    rows: list[dict[str, object]] = []
    for a in json.loads(path.read_text()):
        s = states(str(a["position"]))
        if s is None:
            continue
        rows.append(
            {
                "recording": int(str(a["image"])[:2]),
                "frame": int(a["frame"]),
                "position": str(a["position"]),
                "state1": s[0].value,
                "state2": s[1].value,
                "pose1": _pose(a.get("pose1")),
                "pose2": _pose(a.get("pose2")),
            }
        )
    return pd.DataFrame(rows)
