"""Labeller A: coarse labels from pose and box geometry."""

import numpy as np

from cornerman.gold.rules import agrees, clip_label, frame_label
from cornerman.taxonomy import State


def _kp(hip_y: float, lying: bool = False, sitting: bool = False) -> np.ndarray:
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    if lying:
        kp[[5, 6], :2] = [[0, hip_y], [0, hip_y + 5]]
        kp[[11, 12], :2] = [[60, hip_y], [60, hip_y + 5]]
        kp[[15, 16], :2] = [[140, hip_y], [140, hip_y + 5]]
    else:
        kp[[5, 6], :2] = [[40, hip_y - 60], [60, hip_y - 60]]
        kp[[11, 12], :2] = [[42, hip_y], [58, hip_y]]
        ankle = hip_y + (20 if sitting else 110)
        kp[[15, 16], :2] = [[42, ankle], [58, ankle]]
    return kp


def test_standing_and_clinch() -> None:
    assert frame_label(_kp(100), _kp(100), gap=1.0, iou=0.0) == "standing"
    assert frame_label(_kp(100), _kp(100), gap=0.2, iou=0.3) == "clinch"


def test_ground_top_from_hip_height() -> None:
    # Blue lies on the mat (hips at y=300), red sits up on top (hips at y=200).
    assert frame_label(_kp(200, sitting=True), _kp(300, lying=True), 0.2, 0.5) == "ground-red-top"
    assert frame_label(_kp(300, lying=True), _kp(200, sitting=True), 0.2, 0.5) == "ground-blue-top"
    assert frame_label(_kp(300, lying=True), _kp(305, lying=True), 0.2, 0.5) == "ground-even"


def test_missing_pose_gives_no_label() -> None:
    assert frame_label(None, _kp(100), 1.0, 0.0) is None


def test_clip_label_needs_a_clear_majority() -> None:
    assert clip_label(["standing"] * 7 + ["clinch"] * 3 + [None] * 5) == "standing"
    assert clip_label(["standing"] * 5 + ["clinch"] * 5) is None
    assert clip_label([None, None]) is None


def test_agreement_with_fine_states() -> None:
    assert agrees(State.GROUND_TOP, State.GROUND_BOTTOM, "ground-red-top")
    assert not agrees(State.GROUND_TOP, State.GROUND_BOTTOM, "ground-blue-top")
    assert agrees(State.STANDING_CAGE_PRESSURED, State.STANDING_CAGE_PRESSURING, "clinch")
    assert agrees(State.GROUND_SCRAMBLING, State.GROUND_SCRAMBLING, "ground-even")
    assert not agrees(State.STANDING_NEUTRAL, State.STANDING_NEUTRAL, "clinch")
    assert not agrees(State.STANDING_NEUTRAL, State.STANDING_NEUTRAL, None)
