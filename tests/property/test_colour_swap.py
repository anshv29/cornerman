"""Swapping red and blue in the inputs flips every output exactly (Ansh, 2026-11-14).

Red is the fighter listed first in the title; nothing about the fight should depend on
that. So each step is run twice, once as it is and once with the colours swapped
in its input, and the outputs must be mirror images:
- the moments' features and labels (`swap_colours` is its own inverse, and commutes
  with the +-1 s averages);
- the soft labels;
- the classifier and the smoothing: probabilities, states, how sure each direction is,
  whether it counts as known;
- the actions and who did them;
- the strikes and who threw them;
- the event table, row for row per fighter.
Discrete outputs must match exactly. Probabilities must match to 1e-12 (forward-backward
adds up states in matrix products, whose rounding can depend on the order).

Where a direction isn't known, the state is only a best guess at family level, and on
an exact tie Viterbi takes the first state (red). So states are compared in full where
the direction is known and at family level elsewhere, as the event contract says they
may be used.

Not covered: identity (which box is red). Red and blue kits are different colours, so
there is no exact swap to test there; see AUDIT.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cornerman.config import PerceptionConfig
from cornerman.events.actions import fight_actions
from cornerman.events.moments import FEATURES, PAIRED, SIGNED, swap_colours
from cornerman.events.states import WINDOW, cross_validate, with_window
from cornerman.events.table import fight_events
from cornerman.perception.strikes import fight_strikes
from cornerman.weak_labels.labellers import (
    FAMILIES,
    FAMILY,
    PAIR_STATES,
    SWAP,
    soft_labels,
    vlm_label,
)
from cornerman.weak_labels.labellers import swap_colours as swap_name

A = [None, "standing", "clinch", "ground-red-top", "ground-blue-top", "ground-even"]
M = [None, "apart", "standing-contact", "ground-red-top", "ground-blue-top", "ground-even"]
G = [None, "neutral", "red-pressuring", "blue-pressuring"]
B = [None, "standing", "clinch", "clinch-red", "clinch-blue", "ground-red-top",
     "ground-blue-top", "scramble"]  # fmt: skip


def _moments(seed: int, fights: int = 6, n: int = 40) -> pd.DataFrame:
    """Random moments in runs of one family, with features and labels that lean red
    (as the real ones lean one way or the other), so an unsymmetric model would show it."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for f in range(fights):
        state = "neutral"
        for i in range(n):
            if i % 8 == 0:
                state = str(rng.choice(PAIR_STATES, p=[0.3, 0.15, 0.05, 0.1, 0.05, 0.2, 0.1, 0.05]))
            fam = FAMILY[state]
            r: dict[str, object] = {"fight_id": f"f{f}", "t": i * 0.5, "frame": i * 5,
                                    "live": bool(rng.random() < 0.9)}  # fmt: skip
            for c in FEATURES[1:]:
                r[c] = float(rng.normal()) if rng.random() < 0.8 else np.nan
            r["dy"] = (3.0 if state == "red-top" else -3.0 if state == "blue-top" else 0.0) + float(
                rng.normal(0, 0.5)
            )
            r["gap"] = {"standing": 1.5, "clinch": 0.2, "ground": 0.1}[fam] + float(rng.random())
            for col, opts, share in (("A", A, 0.5), ("M", M, 0.5), ("G", G, 0.3), ("B", B, 0.3)):
                r[col] = opts[int(rng.integers(1, len(opts)))] if rng.random() < share else None
            if rng.random() < 0.7:  # labels mostly right, with red named more often
                top = state if state in ("red-top", "blue-top") else "red-top"
                r["A"] = {"standing": "standing", "clinch": "clinch"}.get(fam, f"ground-{top}")
            rows.append(r)
    return pd.DataFrame(rows)


def _mirror_pred(pred: pd.DataFrame) -> pd.DataFrame:
    """A predictions table as it reads with the colours swapped."""
    out = pred.copy()
    out["pred"] = [swap_name(p) for p in pred["pred"]]
    for pre in ("p_", "q_"):
        for p in PAIR_STATES:
            if pre + p in pred:
                out[pre + swap_name(p)] = pred[pre + p]
    for a, b in PAIRED:
        if a in pred and b in pred:
            out[a], out[b] = pred[b], pred[a]
    return out


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000))
def test_swap_is_its_own_inverse_and_commutes_with_windows(seed: int) -> None:
    m = _moments(seed, fights=2, n=12)
    pd.testing.assert_frame_equal(swap_colours(swap_colours(m)), m)
    pd.testing.assert_frame_equal(with_window(swap_colours(m)), swap_colours(with_window(m)))


def test_every_colour_feature_is_paired_or_signed() -> None:
    """A new feature that names a colour must be added to PAIRED or SIGNED."""
    named = {c for c in FEATURES if "red" in c or "blue" in c}
    assert named == {c for pair in PAIRED for c in pair}
    assert set(SIGNED) <= set(FEATURES)
    assert {f"{c}" for c in WINDOW} <= set(FEATURES)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000))
def test_soft_labels_mirror_exactly(seed: int) -> None:
    m = _moments(seed, fights=2, n=20)
    fam = np.random.default_rng(seed).dirichlet(np.ones(len(FAMILIES)), len(m))
    assert np.array_equal(soft_labels(swap_colours(m), fam), soft_labels(m, fam)[:, SWAP])


@given(st.lists(st.sampled_from(["standing-neutral", "clinch-controlling", "clinch-controlled",
                                 "ground-top", "ground-bottom", "ground-scrambling"]),
                min_size=6, max_size=6))  # fmt: skip
def test_vlm_label_mirrors(states: list[str]) -> None:
    keys = ["direct", "questions", "questions_swapped"]
    row: dict[str, object] = {f"live_{k}": "yes" for k in keys}
    for i, k in enumerate(keys):
        row[f"red_{k}"], row[f"blue_{k}"] = states[2 * i], states[2 * i + 1]
    flipped = {**row, **{f"red_{k}": row[f"blue_{k}"] for k in keys},
               **{f"blue_{k}": row[f"red_{k}"] for k in keys}}  # fmt: skip
    got, want = vlm_label(flipped), vlm_label(row)
    assert got == (swap_name(want) if want else None)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classifier_and_smoothing_mirror(seed: int) -> None:
    m = _moments(seed)
    a, _ = cross_validate(m, n_folds=3, seed=seed)
    b, _ = cross_validate(swap_colours(m), n_folds=3, seed=seed)
    a, b = _mirror_pred(a), b
    cols = [f"{pre}{p}" for pre in ("p_", "q_") for p in PAIR_STATES]
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), atol=1e-12, rtol=0)
    assert np.allclose(a["direction_sure"], b["direction_sure"], atol=1e-12, rtol=0)
    assert (a["pred_family"] == b["pred_family"]).all()
    assert (a["raw_family"] == b["raw_family"]).all()
    assert (a["direction_known"] == b["direction_known"]).all()
    known = a["direction_known"].to_numpy(bool)
    assert (a["pred"].to_numpy()[known] == b["pred"].to_numpy()[known]).all()
    # The test has teeth: the labels lean red, and directions do get decided.
    assert known.any()


def _pred(seed: int, n: int = 60) -> pd.DataFrame:
    """A random predictions table for one fight: runs of states, as smoothing makes them."""
    rng = np.random.default_rng(seed)
    states: list[str] = []
    while len(states) < n:
        states += [str(rng.choice(PAIR_STATES))] * int(rng.integers(1, 10))
    states = states[:n]
    df = pd.DataFrame({"fight_id": "f", "frame": np.arange(n) * 5, "t": np.arange(n) * 0.5,
                       "pred": states, "live": rng.random(n) < 0.9,
                       "direction_sure": rng.uniform(0.5, 1.0, n),
                       "red_speed": rng.uniform(0, 1, n), "blue_speed": rng.uniform(0, 1, n),
                       "shot_id": np.repeat(np.arange(n // 20 + 1), 20)[:n],
                       "pair_h": rng.uniform(0.1, 0.6, n)})  # fmt: skip
    df["red_speed"] = np.where(rng.random(n) < 0.1, 0.0, df["red_speed"])
    df["blue_speed"] = np.where(rng.random(n) < 0.1, 0.0, df["blue_speed"])
    df["pred_family"] = df["pred"].map(FAMILY)
    df["direction_known"] = (df["direction_sure"] >= 0.8) & (df["pred_family"] != "clinch")
    for p in PAIR_STATES:
        df[f"p_{p}"] = np.where(df["pred"] == p, 0.6, 0.4 / 7)
        df[f"q_{p}"] = df[f"p_{p}"]
    return df


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 100_000))
def test_actions_mirror(seed: int) -> None:
    pred = _pred(seed)
    a = fight_actions(pred)
    b = fight_actions(_mirror_pred(pred))
    a["actor"] = [swap_name(x) if isinstance(x, str) else None for x in a["actor"]]
    b["actor"] = [x if isinstance(x, str) else None for x in b["actor"]]
    pd.testing.assert_frame_equal(a, b)


@settings(max_examples=100, deadline=None)
@given(st.integers(0, 100_000))
def test_event_table_mirrors(seed: int) -> None:
    pred = _pred(seed)
    rng = np.random.default_rng(seed)
    acts = fight_actions(pred)
    strikes = pd.DataFrame({"t": np.sort(rng.uniform(0, 30, 10)),
                            "actor": rng.choice(["red", "blue"], 10)})  # fmt: skip
    usable = set(pred["frame"].tolist())
    corners = {"red": "F0001", "blue": "F0002"}
    a = fight_events(pred, acts, strikes, corners, usable)
    swap_actor = {"red": "blue", "blue": "red"}
    b = fight_events(
        _mirror_pred(pred),
        acts.assign(actor=acts["actor"].map(swap_actor)),
        strikes.assign(actor=strikes["actor"].map(swap_actor)),
        {"red": "F0002", "blue": "F0001"},
        usable,
    )
    key = ["t", "fighter"]
    a, b = (x.sort_values(key).reset_index(drop=True) for x in (a, b))
    known = a["direction_known"].to_numpy(bool)
    assert (known == b["direction_known"].to_numpy(bool)).all()
    fam = a["state"].str.split("-").str[0]
    assert (fam == b["state"].str.split("-").str[0]).all()
    assert (a["state"].to_numpy()[known] == b["state"].to_numpy()[known]).all()
    rest = [c for c in a.columns if c != "state"]
    pd.testing.assert_frame_equal(a[rest], b[rest])


def _strike_fight(tmp: Path, seed: int, swap: bool) -> pd.DataFrame:
    """Two random fighters facing each other, 10 fps, with fast random limbs."""
    rng = np.random.default_rng(seed)
    n = 40
    t = np.arange(n) * 0.1
    tracks: list[dict[str, float]] = []
    poses: list[dict[str, float]] = []
    for tid, x0 in ((1, 100.0), (2, 300.0)):
        for f in range(n):
            x = x0 + rng.normal(0, 5)
            tracks.append({"frame": f, "t": t[f], "track_id": tid, "shot_id": 0, "x1": x - 40,
                           "y1": 100.0, "x2": x + 40, "y2": 300.0, "score": 0.9})  # fmt: skip
            kp = np.column_stack([x + rng.normal(0, 30, 17), 100 + rng.uniform(0, 200, 17),
                                  rng.uniform(0.5, 1.0, 17)])  # fmt: skip
            for k in range(17):
                poses.append({"frame": f, "t": t[f], "track_id": tid, "kp": k,
                              "x": kp[k, 0], "y": kp[k, 1], "conf": kp[k, 2]})  # fmt: skip
    d = tmp / ("swapped" if swap else "plain") / str(seed)
    for sub in ("tracking", "identity", "pose"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tracks).to_parquet(d / "tracking" / "tracks.parquet")
    pd.DataFrame(poses).to_parquet(d / "pose" / "poses.parquet")
    pd.DataFrame({"shot_id": [0], "live": [True]}).to_parquet(d / "tracking" / "live.parquet")
    roles = ["blue", "red"] if swap else ["red", "blue"]
    role_table = pd.DataFrame({"frame": np.repeat(np.arange(n), 2), "track_id": np.tile([1, 2], n),
                               "role": np.tile(roles, n)})  # fmt: skip
    role_table.to_parquet(d / "identity" / "roles.parquet")
    moments = pd.DataFrame({"t": np.arange(0, 4, 0.5), "live": True, "contact": False})
    return fight_strikes(d, moments, PerceptionConfig())


@pytest.mark.parametrize("seed", range(5))
def test_strikes_mirror(tmp_path: Path, seed: int) -> None:
    a = _strike_fight(tmp_path, seed, swap=False)
    b = _strike_fight(tmp_path, seed, swap=True)
    assert len(a)  # random fast limbs do throw strikes
    a["actor"] = a["actor"].map({"red": "blue", "blue": "red"})
    key = ["t", "actor", "limb"]
    pd.testing.assert_frame_equal(
        a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    )
