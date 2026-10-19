"""What identity needs from each saved person crop, computed once per fight.

The tracking stage saves a small crop of every tracked person about once a
second (data/crops/{fight_id}/, local only). Per crop this module works out:

- a body mask from the crop's pose: torso polygon and limb lines, so background,
  canvas and the other fighter don't dominate what the crop "looks like";
- an appearance vector: DINOv2-small patch features averaged over that mask
  (models.embed_crops), unit length;
- the torso angle from vertical (on the ground or standing);
- counts of red and blue kit pixels inside the mask, and a hue histogram of
  strongly coloured body pixels (light canvas blue left out);
- whether the box is "merged": much wider than the fight's typical box, which
  in a clinch usually means one box around both fighters.

Same maths as experiments/identity_fighters.py, which produced the features the
identity method was developed and tested on (DECISIONS.md, identity round 3).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
TORSO = [5, 6, 12, 11]
"""COCO keypoints: shoulders and hips, in polygon order."""
KP_CONF = 0.3
INPUT = 224
GRID = 16
"""DINOv2-small with 14 px patches sees a 224 px input as a 16 x 16 grid."""
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
MIN_MASK_CELLS = 3.0
"""A crop whose body mask covers fewer grid cells than this has no usable appearance."""
HUE_BINS = 18
COLUMNS = ["track_id", "frame", "merged", "has_mask", "angle", "red", "blue", "npx"]


def crop_index(folder: Path, tracks: pd.DataFrame, merged_ratio: float) -> pd.DataFrame:
    """One row per saved crop that matches a tracked box: track_id, frame, path, box, merged.

    Crop files are named {track_id:06d}_{frame:07d}.jpg. Order is by file name.
    """
    files = sorted(folder.glob("*.jpg"))
    meta = pd.DataFrame(
        [(int(p.name[:6]), int(p.name[7:14]), str(p)) for p in files],
        columns=["track_id", "frame", "path"],
    )
    meta = meta.merge(
        tracks[["frame", "track_id", "x1", "y1", "x2", "y2"]], on=["frame", "track_id"]
    )
    wh = (meta["x2"] - meta["x1"]) / (meta["y2"] - meta["y1"])
    meta["merged"] = wh > merged_ratio * float(wh.median()) if len(meta) else False
    return meta.reset_index(drop=True)


def pose_lookup(poses: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    """(frame, track_id) -> (17, 3) array of x, y, confidence."""
    p = poses.sort_values(["frame", "track_id", "kp"])
    arr = p[["x", "y", "conf"]].to_numpy(dtype=float).reshape(-1, 17, 3)
    keys = p[["frame", "track_id"]].to_numpy(dtype=np.int64)[::17]
    return {(int(f), int(t)): a for (f, t), a in zip(keys, arr, strict=True)}


def body_mask(
    kp: np.ndarray | None, box: tuple[float, float, float, float], shape: tuple[int, int]
) -> np.ndarray:
    """1 on the torso polygon and along the limbs, in crop pixels; all 0 without a pose."""
    m = np.zeros(shape, np.uint8)
    if kp is None:
        return m
    x1, y1, _, y2 = box
    s = shape[0] / (y2 - y1)
    xy = ((kp[:, :2] - [x1, y1]) * s).astype(np.int32)
    c = kp[:, 2]
    if min(c[TORSO]) > KP_CONF:
        cv2.fillPoly(m, [xy[TORSO]], 1)  # pyright: ignore[reportCallIssue, reportArgumentType]
    width = max(2, int(0.08 * shape[0]))
    for a, b in LIMBS:
        if min(c[a], c[b]) > KP_CONF:
            cv2.line(m, (int(xy[a, 0]), int(xy[a, 1])), (int(xy[b, 0]), int(xy[b, 1])), 1, width)
    return m


def torso_angle(kp: np.ndarray | None) -> float:
    """Degrees between the torso (hip midpoint to shoulder midpoint) and vertical."""
    if kp is None or min(kp[TORSO, 2]) <= KP_CONF:
        return float("nan")
    v = (kp[5, :2] + kp[6, :2]) / 2 - (kp[11, :2] + kp[12, :2]) / 2
    return float(np.degrees(np.arctan2(abs(v[0]), -v[1])))


def colour_counts(img: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """Red and blue kit pixels in the mask, and its size (dark red counts; canvas blue not)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[mask > 0]
    if len(hsv) < 20:
        return 0, 0, 0
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1], hsv[:, 2]
    red = (s >= 70) & (v >= 35) & ((h <= 6) | (h >= 170))
    blue = (s >= 120) & (v >= 40) & (v <= 170) & (h >= 100) & (h <= 140)
    return int(red.sum()), int(blue.sum()), len(hsv)


def hue_hist(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """18-bin hue histogram of strongly coloured body pixels, as a share of all body pixels."""
    out = np.zeros(HUE_BINS, np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[mask > 0]
    if len(hsv) < 20:
        return out
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1], hsv[:, 2]
    strong = (s >= 70) & (v >= 35) & ~((v > 170) & (h >= 90) & (h <= 130))
    out += np.bincount(h[strong] * HUE_BINS // 180, minlength=HUE_BINS)[:HUE_BINS] / len(hsv)
    return out


def model_input(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The crop padded to a square, resized and normalised (3, 224, 224), and the mask
    as a weight per patch (256,)."""
    h, w = img.shape[:2]
    s = max(h, w)
    oy, ox = (s - h) // 2, (s - w) // 2
    sq = np.zeros((s, s, 3), np.uint8)
    sq[oy : oy + h, ox : ox + w] = img[:, :, ::-1]
    msq = np.zeros((s, s), np.float32)
    msq[oy : oy + h, ox : ox + w] = mask
    grid = cv2.resize(msq, (GRID, GRID), interpolation=cv2.INTER_AREA).reshape(-1)
    x = cv2.resize(sq, (INPUT, INPUT)).astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1).astype(np.float32), grid


def describe(
    meta: pd.DataFrame, poses: pd.DataFrame
) -> tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Per crop: the table (COLUMNS), model inputs, patch weights, and hue histograms."""
    pk = pose_lookup(poses)
    rows: list[tuple[int, int, bool, bool, float, int, int, int]] = []
    xs: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    hues: list[np.ndarray] = []
    for r in meta.itertuples(index=False):
        img = cv2.imread(str(r.path))
        if img is None:
            raise FileNotFoundError(r.path)
        kp = pk.get((int(r.frame), int(r.track_id)))  # pyright: ignore[reportArgumentType]
        box = (float(r.x1), float(r.y1), float(r.x2), float(r.y2))  # pyright: ignore[reportArgumentType]
        m = body_mask(kp, box, (img.shape[0], img.shape[1]))
        red, blue, n = colour_counts(img, m)
        x, grid = model_input(img, m)
        rows.append(
            (
                int(r.track_id),  # pyright: ignore[reportArgumentType]
                int(r.frame),  # pyright: ignore[reportArgumentType]
                bool(r.merged),
                bool(grid.sum() >= MIN_MASK_CELLS),
                torso_angle(kp),
                red,
                blue,
                n,
            )
        )
        xs.append(x)
        grids.append(grid)
        hues.append(hue_hist(img, m))
    table = pd.DataFrame(rows, columns=COLUMNS)
    hue = np.array(hues, dtype=np.float32).reshape(-1, HUE_BINS)
    return table, xs, grids, hue
