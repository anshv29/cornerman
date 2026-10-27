import numpy as np
import pytest

from cornerman.perception.fence import cage_call, feet, point_segment, real_edges

W, H = 1000.0, 500.0
# A mat that runs off the left and right of the picture and behind the score bar:
# only the top edge (the far fence) is real.
WIDE = ([0.0, 1.0, 1.0, 0.0], [0.3, 0.3, 0.9, 0.9])
# A whole cage in view: a diamond, every edge real.
CAGE = ([0.5, 0.9, 0.5, 0.1], [0.1, 0.5, 0.8, 0.5])


def test_real_edges_drop_frame_border_and_score_bar() -> None:
    edges = real_edges(*WIDE, W, H)
    assert edges == [(0.0, 150.0, 1000.0, 150.0)]
    assert len(real_edges(*CAGE, W, H)) == 4
    assert real_edges([0.1, 0.2], [0.1, 0.2], W, H) == []


def test_point_segment() -> None:
    seg = (0.0, 0.0, 10.0, 0.0)
    assert point_segment(5, 3, seg) == pytest.approx(3)
    assert point_segment(13, 4, seg) == pytest.approx(5)  # past the end
    assert point_segment(1, 1, (2.0, 2.0, 2.0, 2.0)) == pytest.approx(np.sqrt(2))


def test_feet_from_ankles_else_box_bottom() -> None:
    kp = np.zeros((17, 3))
    kp[15] = [100, 400, 0.9]
    kp[16] = [120, 410, 0.9]
    assert feet((0, 0, 200, 300), kp, 0.3, H) == (110, 405)
    assert feet((0, 0, 200, 300), None, 0.3, H) == (100, 300)
    # Box reaches into the bottom band and no ankle is seen: feet hidden.
    assert feet((0, 0, 200, 480), np.zeros((17, 3)), 0.3, H) is None


def test_cage_call() -> None:
    edges = real_edges(*WIDE, W, H)  # the far fence at y = 150
    h = 200.0  # body height in pixels
    # Red's feet 20 px from the fence, blue 150 px out in front: red is pinned.
    assert cage_call((500, 170), (520, 300), h, edges, 0.35, 0.3) == "red-pressured"
    assert cage_call((520, 300), (500, 170), h, edges, 0.35, 0.3) == "blue-pressured"
    # Both in open space.
    assert cage_call((300, 350), (500, 360), h, edges, 0.35, 0.3) == "neutral"
    # Side by side along the fence: nobody is in front, so abstain.
    assert cage_call((300, 170), (500, 180), h, edges, 0.35, 0.3) is None
    # Near the bottom of the picture is the frame edge, not a fence.
    assert cage_call((300, 440), (500, 300), h, edges, 0.35, 0.3) == "neutral"
    assert cage_call((300, 170), (500, 300), h, [], 0.35, 0.3) is None
