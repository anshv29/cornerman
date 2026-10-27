import numpy as np
import pytest

pytest.importorskip("cv2")

from cornerman.perception.tangle import (
    fighter_pair,
    grey_others,
    pack,
    posture,
    small_mask,
    top_of,
    unpack,
)

H, W = 90, 160


def _rect(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


def test_pack_round_trip_and_small_mask() -> None:
    masks = [_rect(10, 10, 40, 80), _rect(50, 5, 60, 6)]
    assert (unpack(pack(masks), (H, W)) == np.stack(masks)).all()
    s = small_mask(np.kron(masks[0], np.ones((4, 4), bool)), W)
    assert s.shape == (H, W) and (s == masks[0]).all()


def test_grey_others_hides_only_the_other_people() -> None:
    frame = np.full((H, W, 3), 200, np.uint8)
    masks = np.stack([_rect(10, 10, 40, 80), _rect(35, 10, 70, 80)])
    out = grey_others(frame, masks, 0)
    assert (out[20, 20] == 200).all()  # own pixels
    assert (out[20, 37] == 200).all()  # overlap belongs to person 0 too
    assert (out[20, 60] == 127).all()  # the other person
    assert (out[5, 5] == 200).all()  # background


def test_fighter_pair_is_the_two_in_contact() -> None:
    referee = _rect(120, 10, 140, 80)
    a, b = _rect(10, 40, 60, 70), _rect(58, 45, 100, 75)
    assert fighter_pair(np.stack([referee, a, b]), 2)[:2] == (1, 2)  # type: ignore[index]
    assert fighter_pair(np.stack([referee, _rect(10, 10, 30, 80)]), 2) is None
    assert fighter_pair(np.stack([a]), 2) is None


def test_posture_and_top() -> None:
    standing = (_rect(10, 5, 30, 85), _rect(28, 5, 50, 85))
    assert posture(*standing) == "standing"
    top, bottom = _rect(20, 30, 120, 55), _rect(10, 50, 130, 75)
    assert posture(top, bottom) == "ground"
    assert top_of(top, bottom) == 0 and top_of(bottom, top) == 1
    assert top_of(_rect(20, 40, 80, 60), _rect(70, 41, 130, 61)) is None  # side by side
