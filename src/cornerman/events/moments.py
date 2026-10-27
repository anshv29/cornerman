"""Moments: one row every 0.5 s of fighting time, with features and per-moment labellers.

This is the table the state classifier (events/states.py) and Dawid-Skene
(weak_labels/dawid_skene.py) work on. A moment is a frame of the masks stage's
grid (every `masks_every_s`) inside fighting v2 time, so mask features line up
exactly with box and pose features.

Two rules from the Step 1 report (Ansh, 2026-10-27) shape it:
- Skeleton features are only filled where the pair is not in contact. In
  clinch and ground frames no pose method gives two clean skeletons, and the
  rule meant to find the clean ones failed its blind check. Contact means the
  masks stage found two touching people, or, without masks, the fighters'
  boxes overlap like a clinch (IoU above the rules' clinch threshold). Where both
  fighters are boxed, a touching pair counts only if its masks match the two
  fighters' boxes: on the first 5 fights, half the touching pairs found with the
  fighters a body height apart were a fighter and the referee.
- Masks and tangle count only in fighting v2 time.

Red and blue exist only in live shots: identity runs there, and boxes in shots
that aren't live carry no real role. So in those moments every red/blue
feature is missing and the mask labeller can say "ground" but not who is on top.

Labellers per moment (each may abstain, None):
- A, rules on pose and boxes (gold/rules.py), only where not in contact:
  standing, clinch, ground-red-top, ground-blue-top, ground-even.
- M, mask rules (perception/tangle.py), wherever the masks stage ran: apart (no
  touching fighter pair), standing-contact, ground-red-top, ground-blue-top,
  ground-even, ground-top-unknown (on top is found, but not who, outside live
  shots). "apart" is a label, not an abstention: the mask labeller's silence
  depends on the state, and Dawid-Skene assumes abstentions don't.
- G, fence distance (perception/fence.py), standing apart at striking range:
  neutral, red-pressuring, blue-pressuring.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cornerman.config import PerceptionConfig
from cornerman.contracts.perception import fighter_boxes, fighting_table
from cornerman.gold.rules import CLINCH_IOU, body, frame_label, gap_iou, is_down
from cornerman.perception.fence import cage_call, feet, fence_distance, real_edges

A_LABELS = ("standing", "clinch", "ground-red-top", "ground-blue-top", "ground-even")
M_LABELS = (
    "apart",
    "standing-contact",
    "ground-red-top",
    "ground-blue-top",
    "ground-even",
    "ground-top-unknown",
)
G_LABELS = ("neutral", "red-pressuring", "blue-pressuring")

MATCH_IOU = 0.5
"""A mask instance is a fighter when its box overlaps the fighter's tracked box this much."""
SPEED_DT = 0.5
"""Speeds are measured over this many seconds either side of the moment."""


def _iou(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return float(iw * ih / union) if union > 0 else 0.0


def moment_frames(
    frames: pd.DataFrame, fighting: pd.DataFrame, fps: float, every_s: float
) -> pd.DataFrame:
    """The masks stage's grid (every `every_s` of the sampled frames) inside fighting time."""
    idx = frames["frame"].to_numpy(int)
    step = int(np.median(np.diff(idx))) if len(idx) > 1 else 1
    every = max(1, round(every_s * fps / step))
    grid = set(idx[::every].tolist())
    keep = fighting[fighting["fighting"] & fighting["frame"].isin(grid)]
    return keep[["frame", "t", "live"]].reset_index(drop=True)


def _keypoints(poses: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for (f, tid), g in poses.groupby(["frame", "track_id"]):
        kp = np.zeros((17, 3))
        kp[g["kp"].to_numpy(int)] = g[["x", "y", "conf"]].to_numpy(float)
        out[(int(f), int(tid))] = kp  # pyright: ignore[reportArgumentType]
    return out


def _speeds(boxes: pd.DataFrame, at: pd.Series) -> dict[str, np.ndarray]:
    """Each fighter's box-centre speed, in box heights per second, around each time in `at`."""
    out: dict[str, np.ndarray] = {}
    for role in ("red", "blue"):
        b = boxes[boxes["role"] == role].sort_values("t")
        if b.empty:
            out[role] = np.full(len(at), np.nan)
            continue
        c = pd.DataFrame(
            {
                "ms": (b["t"].to_numpy(float) * 1000).round().astype(np.int64),
                "cx": ((b["x1"] + b["x2"]) / 2).to_numpy(float),
                "cy": ((b["y1"] + b["y2"]) / 2).to_numpy(float),
                "h": (b["y2"] - b["y1"]).to_numpy(float),
            }
        )
        t_ms = (at.to_numpy(float) * 1000).round().astype(np.int64)
        ends: list[np.ndarray] = []
        for dt in (-SPEED_DT, SPEED_DT):
            q = pd.DataFrame({"ms": t_ms + round(dt * 1000), "order": np.arange(len(t_ms))})
            m = pd.merge_asof(q.sort_values("ms"), c, on="ms", direction="nearest", tolerance=150)
            ends.append(m.sort_values("order")[["cx", "cy", "h"]].to_numpy(float))
        dist = np.hypot(ends[1][:, 0] - ends[0][:, 0], ends[1][:, 1] - ends[0][:, 1])
        out[role] = dist / ((ends[0][:, 2] + ends[1][:, 2]) / 2) / (2 * SPEED_DT)
    return out


def fight_moments(fight_dir: Path, p: PerceptionConfig) -> pd.DataFrame:
    """Every moment of one fight with its features and labellers A, M and G."""
    video = json.loads((fight_dir / "ingest" / "video.json").read_text())
    fps, width, height = float(video["fps"]), float(video["width"]), float(video["height"])
    frames = pd.read_parquet(fight_dir / "tracking" / "frames.parquet")
    fighting = pd.read_parquet(fighting_table(fight_dir))
    if "live" not in fighting:
        fighting["live"] = True
    moments = moment_frames(frames, fighting, fps, p.masks_every_s)
    wanted = set(moments["frame"].tolist())

    tracks = pd.read_parquet(fight_dir / "tracking" / "tracks.parquet",
                             columns=["frame", "t", "track_id", "shot_id", "x1", "y1", "x2", "y2",
                                      "score"])  # fmt: skip
    roles = pd.read_parquet(fight_dir / "identity" / "roles.parquet")
    live = pd.read_parquet(fight_dir / "tracking" / "live.parquet")
    boxes = fighter_boxes(tracks, roles, live)
    speed = _speeds(boxes, moments["t"])
    n_people = tracks.groupby("frame")["track_id"].nunique()
    shot = frames.set_index("frame")["shot_id"]
    poses = pd.read_parquet(fight_dir / "pose" / "poses.parquet")
    poses = poses[poses["frame"].isin(wanted)]
    kps = _keypoints(poses)
    hull_df = pd.read_parquet(fight_dir / "arena" / "hull.parquet")
    hulls: dict[int, tuple[list[float], list[float]]] = {
        int(f): (list(x), list(y))
        for f, x, y in zip(hull_df["frame"], hull_df["x"], hull_df["y"], strict=True)
        if int(f) in wanted
    }
    by_frame: dict[tuple[int, str], list[float]] = {
        (int(f), str(r)): row
        for f, r, row in zip(
            boxes["frame"], boxes["role"],
            boxes[["track_id", "x1", "y1", "x2", "y2"]].to_numpy(float).tolist(), strict=True,
        )
        if int(f) in wanted
    }  # fmt: skip

    pairs: dict[int, tuple[int, int, int, str, int]] = {}
    inst_by: dict[tuple[int, int], list[float]] = {}
    n_masks: dict[int, int] = {}
    if (fight_dir / "tangle" / "pairs.parquet").exists():
        pdf = pd.read_parquet(fight_dir / "tangle" / "pairs.parquet")
        pairs = {
            int(f): (int(a), int(b), int(c), str(po), int(tp))
            for f, a, b, c, po, tp in zip(pdf["frame"], pdf["a"], pdf["b"], pdf["contact"],
                                          pdf["posture"], pdf["top"], strict=True)
        }  # fmt: skip
        cols = ["frame", "inst", "x1", "y1", "x2", "y2", "area"]
        inst = pd.read_parquet(fight_dir / "masks" / "instances.parquet", columns=cols)
        inst = inst[inst["frame"].isin(list(pairs))]
        rows_i = inst[["x1", "y1", "x2", "y2", "area"]].to_numpy(float).tolist()
        for f, k, row in zip(inst["frame"], inst["inst"], rows_i, strict=True):
            inst_by[(int(f), int(k))] = row
            n_masks[int(f)] = n_masks.get(int(f), 0) + 1

    rows: list[dict[str, object]] = []
    nan = float("nan")
    for k, (f, t, is_live) in enumerate(
        zip(moments["frame"].tolist(), moments["t"].tolist(), moments["live"].tolist(),
            strict=True)
    ):  # fmt: skip
        red, blue = by_frame.get((f, "red")), by_frame.get((f, "blue"))
        both = bool(is_live) and red is not None and blue is not None
        r: dict[str, object] = {
            "frame": f,
            "t": t,
            "shot_id": int(shot.get(f, -1)),
            "live": bool(is_live),
            "n_people": int(n_people.get(f, 0)),
            "n_fighter_boxes": int(red is not None) + int(blue is not None) if is_live else 0,
            "red_speed": speed["red"][k] if is_live else nan,
            "blue_speed": speed["blue"][k] if is_live else nan,
        }
        gap = iou = nan
        if both:
            assert red is not None and blue is not None
            gap, iou = gap_iou(red[1:], blue[1:])
            rh, bh = red[4] - red[2], blue[4] - blue[2]
            r.update(
                gap=gap,
                iou=iou,
                height_ratio=rh / max(bh, 1.0),
                red_aspect=rh / max(red[3] - red[1], 1.0),
                blue_aspect=bh / max(blue[3] - blue[1], 1.0),
                dx=((red[1] + red[3]) - (blue[1] + blue[3])) / 2 / ((rh + bh) / 2),
                dy=((red[2] + red[4]) - (blue[2] + blue[4])) / 2 / ((rh + bh) / 2),
            )
        elif is_live and (red is not None or blue is not None):
            one = red if red is not None else blue
            assert one is not None
            r["one_aspect"] = (one[4] - one[2]) / max(one[3] - one[1], 1.0)

        # Masks: the touching pair, its posture and who is on top. Where both fighters are
        # boxed, the pair must be them (often it is a fighter and the referee).
        is_pair = False
        m_label: str | None = None
        if f in pairs:
            ia, ib, n_contact, post, top = pairs[f]
            r["n_masks"] = n_masks.get(f, 0)
            touching = post != "none"
            a = inst_by.get((f, ia), [nan] * 5)
            b = inst_by.get((f, ib), [nan] * 5)
            side = _pair_sides(a, b, red, blue) if touching and both else None
            is_pair = touching and (side is not None or not both)
            r["mask_pair"] = is_pair
            m_label = None if is_pair else "apart"
            if is_pair:
                x1, y1 = min(a[0], b[0]), min(a[1], b[1])
                x2, y2 = max(a[2], b[2]), max(a[3], b[3])
                r["mask_contact"] = n_contact / max(min(a[4], b[4]), 1.0)
                r["mask_ratio"] = (y2 - y1) / max(x2 - x1, 1.0)
                if side is not None:  # +1: instance a is red
                    dy = ((a[1] + a[3]) - (b[1] + b[3])) / 2 / max(y2 - y1, 1.0)
                    r["mask_dy"] = dy * side  # positive: red's mask lower
                if post == "standing":
                    m_label = "standing-contact"
                elif top < 0:
                    m_label = "ground-even"
                elif side is None:
                    m_label = "ground-top-unknown"
                else:
                    red_top = (top == ia) == (side > 0)
                    m_label = "ground-red-top" if red_top else "ground-blue-top"
                r["mask_top"] = (
                    1.0 if m_label == "ground-red-top"
                    else -1.0 if m_label == "ground-blue-top"
                    else 0.0 if m_label == "ground-even" else nan
                )  # fmt: skip
        contact = is_pair or (both and iou > CLINCH_IOU)
        r["contact"] = bool(contact)
        r["M"] = m_label

        # Skeletons only where the pair is apart; rules A and fence G need them.
        a_label: str | None = None
        g_label: str | None = None
        if both and not contact:
            assert red is not None and blue is not None
            rk, bk = kps.get((f, int(red[0]))), kps.get((f, int(blue[0])))
            rb, bb = body(rk), body(bk)
            r.update(red_angle=rb["angle"], red_hip_ankle=rb["hip_over_ankle"],
                     blue_angle=bb["angle"], blue_hip_ankle=bb["hip_over_ankle"])  # fmt: skip
            torso = (rb["torso"] + bb["torso"]) / 2
            r["hip_diff"] = (bb["hip_y"] - rb["hip_y"]) / torso if torso > 0 else nan
            a_label = frame_label(rk, bk, gap, iou)
            hx, hy = hulls.get(f, ([], []))
            edges = real_edges(hx, hy, width, height)
            rf = feet(red[1:], rk, p.kp_min_conf, height, p.mat_cut)
            bf = feet(blue[1:], bk, p.kp_min_conf, height, p.mat_cut)
            body_h = ((red[4] - red[2]) + (blue[4] - blue[2])) / 2
            if rf is not None:
                r["red_fence"] = fence_distance(rf, edges)[0] / body_h
            if bf is not None:
                r["blue_fence"] = fence_distance(bf, edges)[0] / body_h
            down = (not np.isnan(rb["angle"]) and is_down(rb)) or (
                not np.isnan(bb["angle"]) and is_down(bb)
            )
            lo, hi = p.fence_gap
            if rf is not None and bf is not None and not down and lo <= gap <= hi:
                call = cage_call(rf, bf, body_h, edges, p.fence_near, p.fence_margin)
                g_label = {
                    "neutral": "neutral",
                    "red-pressured": "blue-pressuring",
                    "blue-pressured": "red-pressuring",
                }.get(call or "")
        r["A"] = a_label
        r["G"] = g_label
        rows.append(r)
    out = pd.DataFrame(rows)
    for col in FEATURES:
        if col not in out:
            out[col] = nan
    out[FEATURES] = out[FEATURES].astype(float)
    for col in ("A", "M", "G"):
        if col not in out:
            out[col] = None
    out.insert(0, "fight_id", fight_dir.name)
    return out


def _pair_sides(
    a: list[float], b: list[float], red: list[float] | None, blue: list[float] | None
) -> int | None:
    """+1 if mask instance a is red and b is blue, -1 if the other way, None if unclear."""
    if red is None or blue is None:
        return None
    ar, ab = _iou(a[:4], red[1:]), _iou(a[:4], blue[1:])
    br, bb = _iou(b[:4], red[1:]), _iou(b[:4], blue[1:])
    if ar >= MATCH_IOU and bb >= MATCH_IOU and ar + bb > ab + br:
        return 1
    if ab >= MATCH_IOU and br >= MATCH_IOU and ab + br > ar + bb:
        return -1
    return None


FEATURES = [
    "live",
    "n_people",
    "n_fighter_boxes",
    "red_speed",
    "blue_speed",
    "gap",
    "iou",
    "height_ratio",
    "red_aspect",
    "blue_aspect",
    "dx",
    "dy",
    "one_aspect",
    "n_masks",
    "mask_pair",
    "mask_contact",
    "mask_ratio",
    "mask_dy",
    "mask_top",
    "contact",
    "red_angle",
    "red_hip_ankle",
    "blue_angle",
    "blue_hip_ankle",
    "hip_diff",
    "red_fence",
    "blue_fence",
]
"""Model inputs. Red/blue features are missing outside live shots, skeleton features
in contact, mask features where the masks stage hasn't run."""


def all_moments(derived: Path, fights: list[str], p: PerceptionConfig) -> pd.DataFrame:
    """Moments of every listed fight that has fighting time and arena output."""
    parts = [
        fight_moments(derived / f, p)
        for f in fights
        if (derived / f / "arena" / "manifest.json").exists()
    ]
    parts = [x for x in parts if len(x)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
