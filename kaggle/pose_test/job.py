"""Kaggle GPU job: three pose methods on frames of tangled clinch and ground shots.

Tracking marks a shot live only when most of its frames show two people, so long
clinch and ground shots (the two fighters merged into one box) never got pose.
Before rerunning pose on them, this compares, on the same frames:
  A. today's method: RF-DETR Large person boxes, ViTPose++ Large in each box;
  B. RF-DETR Seg Large instance masks: ViTPose++ in each instance's box, with the
     pixels of the other people greyed out, so it can't take the other body's limbs;
  C. RTMO-l (rtmlib, ONNX): bottom-up one-stage pose, no boxes first.
Frames come from a private dataset ({fight_id}__{frame}.jpg, flat), made by
experiments/pose_tangle_check.py. Output: one skeleton drawing per frame and method
(no boxes, same colours and keypoint threshold for all, so the methods look alike)
and results.json with counts per frame and method.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = ""  # BUILD:REPO
COMMIT = ""  # BUILD:COMMIT

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
DET_THRESHOLD = 0.4
"""Same as perception.det_threshold."""
KP_MIN = 0.3
"""Keypoints drawn and counted at or above this score, for every method."""
RTMO_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
    "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"
)
POSE_MODEL = "usyd-community/vitpose-plus-large"
POSE_REVISION = "e211df377e49"
"""Same as perception.pose_model and pose_revision."""
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]  # fmt: skip
COLOURS = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255),
           (255, 255, 255)]  # fmt: skip


def install() -> None:
    pkgs = [f"cornerman[gpu] @ git+{REPO}@{COMMIT}", "rtmlib==0.0.16", "onnxruntime-gpu"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def draw(img, kps, scores):
    import cv2

    out = img.copy()
    for i, (k, s) in enumerate(zip(kps, scores, strict=True)):
        c = COLOURS[i % len(COLOURS)]
        for a, b in EDGES:
            if s[a] >= KP_MIN and s[b] >= KP_MIN:
                pa = (int(k[a][0]), int(k[a][1]))
                pb = (int(k[b][0]), int(k[b][1]))
                cv2.line(out, pa, pb, c, 3, cv2.LINE_AA)
        for j in range(17):
            if s[j] >= KP_MIN:
                cv2.circle(out, (int(k[j][0]), int(k[j][1])), 4, c, -1, cv2.LINE_AA)
    return out


def method_a(img):
    from cornerman.perception.models import detect_people, estimate_poses

    boxes, _ = detect_people([img], "rfdetr-large", DET_THRESHOLD)[0]
    if not len(boxes):
        return [], []
    kps, sc = estimate_poses([img], [boxes], POSE_MODEL, POSE_REVISION, "cuda")[0]
    return list(kps), list(sc)


_SEG = None


def method_b(img):
    import numpy as np
    from rfdetr import RFDETRSegLarge

    from cornerman.perception.models import PERSON, estimate_poses

    global _SEG
    if _SEG is None:
        _SEG = RFDETRSegLarge()
    det = _SEG.predict(np.ascontiguousarray(img[:, :, ::-1]), threshold=DET_THRESHOLD)
    keep = np.asarray(det.class_id) == PERSON
    masks = np.asarray(det.mask)[keep] if det.mask is not None else None
    if masks is None or not len(masks):
        return [], []
    kps_all, sc_all = [], []
    grey = np.full_like(img, 127)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs) < 50:
            continue
        others = np.zeros_like(m)
        for j, o in enumerate(masks):
            if j != i:
                others |= o
        hide = others & ~m
        masked = np.where(hide[:, :, None], grey, img)
        box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=float)
        kps, sc = estimate_poses([masked], [box], POSE_MODEL, POSE_REVISION, "cuda")[0]
        kps_all.append(kps[0])
        sc_all.append(sc[0])
    return kps_all, sc_all


_RTMO = None


def method_c(img):
    from rtmlib import RTMO

    global _RTMO
    if _RTMO is None:
        _RTMO = RTMO(RTMO_URL, backend="onnxruntime", device="cuda")
    kps, sc = _RTMO(img, score_thr=DET_THRESHOLD)
    keep = [i for i in range(len(sc)) if sc[i].max() > 0]  # no detection: one zeroed row
    return [kps[i] for i in keep], [sc[i] for i in keep]


def main() -> None:
    import cv2

    start = time.time()
    install()
    frames = sorted(INPUT.rglob("*__*.jpg"))
    print(f"{len(frames)} frames, setup {time.time() - start:.0f}s", flush=True)
    results: list[dict[str, object]] = []
    methods = {"A": method_a, "B": method_b, "C": method_c}
    for name in methods:
        (WORK / "draw" / name).mkdir(parents=True, exist_ok=True)
    for path in frames:
        img = cv2.imread(str(path))
        row: dict[str, object] = {"frame": path.stem}
        for name, fn in methods.items():
            try:
                kps, sc = fn(img)
                good = [int((s >= KP_MIN).sum()) for s in sc]
                row[name] = {"n": len(sc), "n_10kp": sum(g >= 10 for g in good), "kp": good}
                cv2.imwrite(str(WORK / "draw" / name / path.name), draw(img, kps, sc))
            except Exception:
                row[name] = {"error": traceback.format_exc()[-1500:]}
        results.append(row)
        print(json.dumps(row)[:300], flush=True)
    (WORK / "results.json").write_text(json.dumps(results, indent=1))
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
