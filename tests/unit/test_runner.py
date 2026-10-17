from pathlib import Path

import pytest

from cornerman.config import PathsConfig, Settings, StubConfig
from cornerman.pipeline.runner import StageResult, Status, run, stage_dir, topo_sort
from cornerman.pipeline.stage import QuarantineError, Stage, StageContext


def write_a(ctx: StageContext) -> None:
    (ctx.out_dir / "a.txt").write_text(f"fps={ctx.settings.stub.fps}")


def write_b(ctx: StageContext) -> None:
    a = (ctx.inputs["a"] / "a.txt").read_text()
    (ctx.out_dir / "b.txt").write_text(a + " then b")


def write_c(ctx: StageContext) -> None:
    b = (ctx.inputs["b"] / "b.txt").read_text()
    (ctx.out_dir / "c.txt").write_text(b + " then c")


def reject_bad_fight(ctx: StageContext) -> None:
    if ctx.fight_id == "bad":
        raise QuarantineError("too dark to see anything")
    (ctx.out_dir / "ok.txt").write_text("ok")


def crash(ctx: StageContext) -> None:
    (ctx.out_dir / "partial.txt").write_text("half")
    raise RuntimeError("session killed")


STAGES = [
    Stage("c", write_c, deps=("b",)),
    Stage("a", write_a, config_keys=("stub.fps",)),
    Stage("b", write_b, deps=("a",)),
]


def make_settings(tmp_path: Path, fps: float = 10.0, n_shots: int = 4) -> Settings:
    return Settings(
        seed=1,
        paths=PathsConfig(derived=tmp_path / "derived", quarantine=tmp_path / "quarantine"),
        stub=StubConfig(fps=fps, n_shots=n_shots),
    )


def statuses(results: list[StageResult]) -> dict[str, Status]:
    return {r.stage: r.status for r in results}


def test_topo_sort_orders_by_deps() -> None:
    assert [s.name for s in topo_sort(STAGES)] == ["a", "b", "c"]


def test_topo_sort_rejects_cycles_and_unknown_deps() -> None:
    with pytest.raises(ValueError, match="cycle"):
        topo_sort([Stage("x", write_a, deps=("y",)), Stage("y", write_a, deps=("x",))])
    with pytest.raises(ValueError, match="unknown"):
        topo_sort([Stage("x", write_a, deps=("nope",))])


def test_second_run_is_fully_cached(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    assert set(statuses(run(STAGES, s, ["f1"])).values()) == {Status.RAN}
    assert set(statuses(run(STAGES, s, ["f1"])).values()) == {Status.CACHED}
    assert (stage_dir(s, "f1", "c") / "c.txt").read_text() == "fps=10.0 then b then c"


def test_relevant_config_change_reruns_stage_and_downstream(tmp_path: Path) -> None:
    run(STAGES, make_settings(tmp_path), ["f1"])
    out = statuses(run(STAGES, make_settings(tmp_path, fps=25), ["f1"]))
    assert out == {"a": Status.RAN, "b": Status.RAN, "c": Status.RAN}


def test_unrelated_config_change_stays_cached(tmp_path: Path) -> None:
    run(STAGES, make_settings(tmp_path), ["f1"])
    out = statuses(run(STAGES, make_settings(tmp_path, n_shots=9), ["f1"]))
    assert set(out.values()) == {Status.CACHED}


def test_target_and_force_rerun_one_stage(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    run(STAGES, s, ["f1"])
    out = statuses(run(STAGES, s, ["f1"], target="b", force=True))
    assert out == {"a": Status.CACHED, "b": Status.RAN}


def test_tampered_output_is_rebuilt(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    run(STAGES, s, ["f1"])
    (stage_dir(s, "f1", "b") / "b.txt").write_text("edited by hand")
    out = statuses(run(STAGES, s, ["f1"]))
    assert out["b"] == Status.RAN
    # b's output is the same as before the edit, so c's key still matches.
    assert out["c"] == Status.CACHED


def test_bad_fight_is_quarantined_and_others_continue(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    stages = [Stage("check", reject_bad_fight), Stage("next", write_a, deps=("check",))]
    results = run(stages, s, ["bad", "good"])
    by_fight = {(r.fight_id, r.stage): r.status for r in results}
    assert by_fight[("bad", "check")] == Status.QUARANTINED
    assert by_fight[("bad", "next")] == Status.BLOCKED
    assert by_fight[("good", "next")] == Status.RAN
    record = (s.paths.quarantine / "bad.json").read_text()
    assert "too dark" in record


def test_crash_leaves_no_partial_output(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    with pytest.raises(RuntimeError):
        run([Stage("crash", crash)], s, ["f1"])
    fight_dir = s.paths.derived / "f1"
    assert list(fight_dir.iterdir()) == []
