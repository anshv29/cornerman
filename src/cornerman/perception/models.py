"""Thin wrappers around the pretrained models. Everything that imports torch lives here.

Models load once per process (functools.cache), since a Kaggle job runs many
fights back to back. Choices and versions are logged in DECISIONS.md.
"""

from functools import cache
from typing import Any, cast

import numpy as np

from cornerman.perception.video import Frame

PERSON = 1
"""COCO class ID for person in RF-DETR's label map."""
COCO_DATASET_INDEX = 0
"""ViTPose++ has one head per training dataset; 0 is COCO's 17 keypoints."""


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@cache
def _transnet(device: str) -> Any:
    from transnetv2_pytorch import TransNetV2  # pyright: ignore[reportMissingTypeStubs]

    return TransNetV2(device=device).eval()  # pyright: ignore[reportUnknownMemberType]


def cut_probabilities(frames: np.ndarray, device: str) -> np.ndarray:
    """Per-frame probability of a cut or transition, for (n, 27, 48, 3) RGB frames."""
    import torch

    model = _transnet(device)
    x = torch.from_numpy(np.ascontiguousarray(frames)).to(model.device)  # pyright: ignore[reportUnknownMemberType]
    with torch.no_grad():
        single, _ = model.predict_frames(x, quiet=True)
    return single.cpu().numpy().astype(float)


@cache
def _detector(name: str, half: bool) -> Any:
    import torch
    from rfdetr import RFDETRLarge, RFDETRMedium

    models: dict[str, Any] = {"rfdetr-large": RFDETRLarge, "rfdetr-medium": RFDETRMedium}
    model = models[name]()
    if half and torch.cuda.is_available():
        # compile=False: batches vary in size (the last one of a fight is short).
        model.inference(compile=False, dtype=torch.float16, inplace=True)
    return model


def detect_people(
    frames: list[Frame], name: str, threshold: float, half: bool = False
) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each BGR frame: (boxes as x1 y1 x2 y2, confidences) for people only."""
    model = _detector(name, half)
    rgb = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
    preds = model.predict(rgb, threshold=threshold)
    if not isinstance(preds, list):
        preds = [preds]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for p in preds:  # pyright: ignore[reportUnknownVariableType]
        keep = np.asarray(p.class_id) == PERSON  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        boxes = np.asarray(p.xyxy, dtype=float)[keep]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        conf = np.asarray(p.confidence, dtype=float)[keep]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        out.append((boxes, conf))
    return out


class ShotTracker:
    """BoT-SORT, restarted at every camera cut so no track spans two shots."""

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.shot: int | None = None
        self.next_offset = 0
        self.offset = 0
        self.max_local = -1
        self._tracker: Any = None

    def update(self, shot_id: int, frame: Frame, boxes: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """Global track ID per box (-1 while a track is still unconfirmed)."""
        import supervision as sv
        from trackers import BoTSORTTracker

        if shot_id != self.shot:
            self.shot = shot_id
            self.offset = self.next_offset
            self._tracker = BoTSORTTracker(frame_rate=self.fps, lost_track_buffer=int(self.fps))
        det = sv.Detections(xyxy=boxes.reshape(-1, 4), confidence=conf)
        tracked = self._tracker.update(det, frame=frame)
        ids = np.asarray(tracked.tracker_id, dtype=np.int64)
        xyxy = np.asarray(tracked.xyxy, dtype=float).reshape(-1, 4)
        # Map tracker output back onto our boxes by exact box match.
        result = np.full(len(boxes), -1, dtype=np.int64)
        for box_id, box in zip(ids, xyxy, strict=True):
            if box_id < 0:
                continue
            match = np.flatnonzero(np.all(np.isclose(boxes, box, atol=1e-3), axis=1))
            if len(match):
                result[match[0]] = self.offset + int(box_id)
        if (result >= 0).any():
            self.next_offset = max(self.next_offset, int(result.max()) + 1)
        return result


@cache
def _pose(name: str, revision: str, device: str, half: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    processor = cast(Any, AutoProcessor.from_pretrained(name, revision=revision))  # pyright: ignore[reportUnknownMemberType]
    model: Any = VitPoseForPoseEstimation.from_pretrained(name, revision=revision)  # pyright: ignore[reportUnknownMemberType]
    dtype = torch.float16 if half and device.startswith("cuda") else torch.float32
    return processor, model.to(device=device, dtype=dtype).eval()


def estimate_poses(
    frames: list[Frame],
    boxes: list[np.ndarray],
    name: str,
    revision: str,
    device: str,
    half: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each frame and its boxes (x1 y1 x2 y2): keypoints (n, 17, 2) and scores (n, 17)."""
    import torch
    from PIL import Image

    processor, model = _pose(name, revision, device, half)
    images = [Image.fromarray(np.ascontiguousarray(f[:, :, ::-1])) for f in frames]
    coco = [np.column_stack([b[:, :2], b[:, 2:] - b[:, :2]]).tolist() for b in boxes]
    inputs = processor(images, boxes=coco, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    n_boxes = sum(len(b) for b in boxes)
    index = torch.full((n_boxes,), COCO_DATASET_INDEX, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(**inputs, dataset_index=index)
    # Heatmap peak finding in FP32, so FP16 only affects the network itself.
    outputs.heatmaps = outputs.heatmaps.float()
    results = processor.post_process_pose_estimation(outputs, boxes=coco)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for per_image in results:
        kps = np.array([p["keypoints"].cpu().numpy() for p in per_image]).reshape(-1, 17, 2)
        scores = np.array([p["scores"].cpu().numpy() for p in per_image]).reshape(-1, 17)
        out.append((kps, scores))
    return out
