import numpy as np

from cornerman.config import PerceptionConfig
from cornerman.perception.strikes import _refractory, limb_strikes

P = PerceptionConfig()
H = 200.0
"""Body height in pixels."""


def _pose(dx_wrist: float = 0.0, ankle_lift: float = 0.0, ankle_dx: float = 0.0) -> np.ndarray:
    """A standing fighter facing right, shoulders at y 50, hips 100, ankles 200."""
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    kp[[5, 6], :2] = [[100, 50], [100, 50]]
    kp[[11, 12], :2] = [[100, 100], [100, 100]]
    kp[[9, 10], :2] = [[120 + dx_wrist, 60], [115, 60]]
    kp[[15, 16], :2] = [[100 + ankle_dx, 200 - ankle_lift], [100, 200]]
    return kp


OPP = _pose()
OPP[[5, 6], 0] = 300
"""The opponent, to the right."""


def test_a_jab_is_a_hand_strike() -> None:
    # 0.5 body heights toward the opponent in 0.1 s: 5 body heights per second
    got = limb_strikes(_pose(), _pose(dx_wrist=100), OPP, 0.1, H, P)
    assert [limb for limb, _ in got] == ["hand"]


def test_guard_movement_and_keypoint_jumps_are_not_strikes() -> None:
    assert limb_strikes(_pose(), _pose(dx_wrist=10), OPP, 0.1, H, P) == []
    assert limb_strikes(_pose(), _pose(dx_wrist=-100), OPP, 0.1, H, P) == []  # away
    assert limb_strikes(_pose(), _pose(dx_wrist=400), OPP, 0.1, H, P) == []  # 20 bh/s


def test_a_lifted_fast_foot_is_a_kick_and_a_step_is_not() -> None:
    got = limb_strikes(_pose(), _pose(ankle_lift=80, ankle_dx=40), OPP, 0.1, H, P)
    assert [limb for limb, _ in got] == ["foot"]
    assert limb_strikes(_pose(), _pose(ankle_dx=80), OPP, 0.1, H, P) == []  # not lifted


def test_refractory_keeps_one_per_fighter_per_window() -> None:
    rows = [{"t": t, "actor": a} for t, a in ((1.0, "red"), (1.1, "red"), (1.1, "blue"),
                                              (1.5, "red"))]  # fmt: skip
    kept = _refractory(rows, 0.3)  # pyright: ignore[reportArgumentType]
    assert [(r["t"], r["actor"]) for r in kept] == [(1.0, "red"), (1.1, "blue"), (1.5, "red")]
