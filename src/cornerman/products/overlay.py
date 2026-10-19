"""Draw tracks, roles and skeletons on top of a fight video.

Overlay videos contain fight footage, so they are written outside the repo's
tracked data, never committed and never published (hard rule 2). They're for
checking the pipeline by eye on a few fights.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.contracts.perception import load_roles, role_keys
from cornerman.perception.identity import OTHER
from cornerman.perception.video import frame_step, iter_frames

# BGR colours per role.
COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (40, 40, 220),
    "blue": (220, 120, 30),
    "referee": (200, 200, 200),
    OTHER: (90, 90, 90),
}
# COCO-17 skeleton edges.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]  # fmt: skip
MIN_KP_CONF = 0.3


def render(
    video: Path,
    derived: Path,
    out: Path,
    target_fps: float,
    seconds: float,
) -> Path:
    """Write an mp4 of the first `seconds` of live footage with everything drawn on.

    `derived` is the fight's folder of stage outputs (derived/{fight_id}).
    """
    vinfo = pd.read_json(derived / "ingest" / "video.json", typ="series")
    fps = float(vinfo["fps"])
    step = frame_step(fps, target_fps)
    tracks = pd.read_parquet(derived / "tracking" / "tracks.parquet")
    roles = load_roles(derived)
    keys = role_keys(roles)
    live = pd.read_parquet(derived / "tracking" / "live.parquet")
    poses = pd.read_parquet(derived / "pose" / "poses.parquet")
    poses = poses.merge(roles[[*keys, "role"]], on=keys, how="left")
    poses["role"] = poses["role"].fillna(OTHER)
    tracks = tracks.merge(roles[[*keys, "role"]], on=keys, how="left")
    tracks["role"] = tracks["role"].fillna(OTHER)
    live_ids = set(live.loc[live["live"], "shot_id"])
    live_frames = set(tracks.loc[tracks["shot_id"].isin(live_ids), "frame"])
    by_frame = dict(tuple(tracks.groupby("frame")))
    pose_by_frame = dict(tuple(poses.groupby("frame")))

    out.parent.mkdir(parents=True, exist_ok=True)
    size = (int(vinfo["width"]), int(vinfo["height"]))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps / step, size)
    max_frames = int(seconds * fps / step)
    written = 0
    try:
        for i, frame in iter_frames(video, step=step):
            if i not in live_frames:
                continue
            draw(frame, by_frame.get(i), pose_by_frame.get(i))
            cv2.putText(frame, f"t={i / fps:7.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)  # fmt: skip
            writer.write(frame)
            written += 1
            if written >= max_frames:
                break
    finally:
        writer.release()
    return out


def draw(frame: np.ndarray, boxes: pd.DataFrame | None, poses: pd.DataFrame | None) -> None:
    if boxes is not None:
        for row in boxes.itertuples():
            colour = COLOURS[str(row.role)]
            p1 = (int(row.x1), int(row.y1))  # pyright: ignore[reportArgumentType]
            p2 = (int(row.x2), int(row.y2))  # pyright: ignore[reportArgumentType]
            cv2.rectangle(frame, p1, p2, colour, 2)
            label = f"{row.role} #{row.track_id}"
            cv2.putText(frame, label, (p1[0], max(12, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, 1)  # fmt: skip
    if poses is not None:
        for _, g in poses.groupby("track_id"):
            colour = COLOURS[str(g["role"].iloc[0])]
            pts = g.sort_values("kp")[["x", "y", "conf"]].to_numpy()
            for a, b in SKELETON:
                if pts[a, 2] >= MIN_KP_CONF and pts[b, 2] >= MIN_KP_CONF:
                    pa = (int(pts[a, 0]), int(pts[a, 1]))
                    pb = (int(pts[b, 0]), int(pts[b, 1]))
                    cv2.line(frame, pa, pb, colour, 2)
            for x, y, c in pts:
                if c >= MIN_KP_CONF:
                    cv2.circle(frame, (int(x), int(y)), 3, colour, -1)
