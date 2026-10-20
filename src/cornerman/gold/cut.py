"""Cut the sampled silver-set clips out of a fight's video, with the corners drawn in.

Each clip gets a red box around the red-corner fighter and a blue box around the
blue-corner fighter, from the identity stage's role per box. The VLM is asked
about "the fighter in the red box", so it labels the same two people as the
rules labeller, even when both kits are black with coloured trim.

Frames come from one sequential pass at the tracking step, so every clip frame is
a tracked frame and its boxes are exact. Clips are H.264 at 10 fps, 480 px high:
enough for a VLM, which samples a few frames per second anyway.

Clips are footage: they stay in data/gold/clips (gitignored, local only) and go
to Kaggle only as a private dataset that is deleted after the labelling job.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cornerman.perception.decode import ffmpeg_exe
from cornerman.perception.video import iter_frames

COLOURS = {"red": (0, 0, 255), "blue": (255, 0, 0)}
"""BGR."""
HEIGHT = 480


def clip_id(fight_id: str, t0: float) -> str:
    return f"{fight_id}_{round(t0 * 10):06d}"


def clip_frames(frames: pd.DataFrame, clips: pd.DataFrame) -> dict[str, list[int]]:
    """For each clip (by clip_id), the tracked frame indices inside [t0, t1)."""
    out: dict[str, list[int]] = {}
    t = frames["t"].to_numpy(float)
    idx = frames["frame"].to_numpy(int)
    for cid, t0, t1 in zip(clips["clip_id"], clips["t0"], clips["t1"], strict=True):
        keep = (t >= float(t0)) & (t < float(t1))
        out[str(cid)] = idx[keep].tolist()
    return out


def draw(frame: np.ndarray, boxes: pd.DataFrame) -> np.ndarray:
    """The frame with each fighter's box in their corner colour."""
    img = frame.copy()
    thick = max(2, img.shape[0] // 240)
    for role, x1, y1, x2, y2 in zip(
        boxes["role"], boxes["x1"], boxes["y1"], boxes["x2"], boxes["y2"], strict=True
    ):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, COLOURS[str(role)], thick)
    return img


def encode(frames: list[np.ndarray], out: Path, fps: float, height: int = HEIGHT) -> None:
    """Write BGR frames as an H.264 mp4, scaled to `height` (even width)."""
    h, w = frames[0].shape[:2]
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    proc = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {out.name}")


def cut_fight(
    video: Path,
    step: int,
    frames: pd.DataFrame,
    boxes: pd.DataFrame,
    clips: pd.DataFrame,
    out_dir: Path,
    fps: float,
) -> list[Path]:
    """Write one mp4 per clip of this fight into out_dir; returns the paths written.

    `step` is the tracking frame step, `boxes` one box per fighter per frame (frame,
    role, x1..y2), `clips` this fight's rows of the sample (clip_id, t0, t1).
    """
    wanted = clip_frames(frames, clips)
    owners: dict[int, list[str]] = {}
    for cid, fs in wanted.items():
        for f in fs:
            owners.setdefault(f, []).append(cid)
    if not owners:
        return []
    by_frame = dict(tuple(boxes.groupby("frame")))
    buffers: dict[str, list[np.ndarray]] = {cid: [] for cid in wanted}
    written: list[Path] = []
    for idx, frame in iter_frames(video, step=step, until_frame=max(owners) + 1):
        if idx not in owners:
            continue
        b = by_frame.get(idx)
        img = draw(frame, b) if b is not None else frame
        for cid in owners[idx]:
            buffers[cid].append(img)
            if idx == wanted[cid][-1]:
                path = out_dir / f"{cid}.mp4"
                encode(buffers.pop(cid), path, fps)
                written.append(path)
    return written


def cut_all(
    clips: pd.DataFrame,
    urls: dict[str, str],
    derived: Path,
    out_dir: Path,
    work: Path,
    max_height: int,
    fps: float,
) -> pd.DataFrame:
    """Download each fight once, cut its clips, delete the video. One row per fight.

    The download must be the video the pipeline processed, or boxes and times may
    not line up: the same file (sha256 in ingest/video.json), or at least the same
    frame count, frame rate and size (a remux can change the bytes but not the
    frames). A fight that doesn't match is skipped and reported.
    """
    import hashlib
    import json

    from cornerman.contracts.perception import fighter_boxes, load_roles
    from cornerman.perception.video import frame_step, probe
    from cornerman.remote.videos import download

    rows: list[dict[str, object]] = []
    for fid, group in clips.groupby("fight_id"):
        fid = str(fid)
        todo = group[[not (out_dir / f"{c}.mp4").exists() for c in group["clip_id"]]]
        if todo.empty:
            rows.append({"fight_id": fid, "status": "done", "clips": len(group)})
            continue
        d = derived / fid
        info = json.loads((d / "ingest" / "video.json").read_text())
        video = work / f"{fid}.mp4"
        try:
            download(urls[fid], video, max_height)
            same = hashlib.sha256(video.read_bytes()).hexdigest() == info["sha256"]
            p = probe(video)
            frames_match = (p.n_frames, p.width, p.height) == (
                info["n_frames"],
                info["width"],
                info["height"],
            ) and abs(p.fps - float(info["fps"])) < 1e-3
            if not (same or frames_match):
                rows.append({"fight_id": fid, "status": "different video", "clips": 0})
                print(rows[-1], flush=True)
                continue
            boxes = fighter_boxes(
                pd.read_parquet(d / "tracking" / "tracks.parquet"),
                load_roles(d),
                pd.read_parquet(d / "tracking" / "live.parquet"),
            )
            written = cut_fight(
                video,
                frame_step(float(info["fps"]), fps),
                pd.read_parquet(d / "tracking" / "frames.parquet"),
                boxes,
                todo,
                out_dir,
                fps,
            )
            status = "cut" if same else "cut (same frames, different file)"
            rows.append({"fight_id": fid, "status": status, "clips": len(written)})
        except Exception as e:  # one bad download shouldn't stop the batch
            rows.append({"fight_id": fid, "status": f"error: {e}"[:300], "clips": 0})
        finally:
            video.unlink(missing_ok=True)
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)
