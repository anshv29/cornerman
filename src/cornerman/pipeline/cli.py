"""Command line entry point.

`cornerman run --config configs/fixture.yaml` runs the pipeline.
`cornerman collect --config configs/collect.yaml` builds the local fight registry.
`cornerman kaggle push|status|pull|batch|import` drives Kaggle jobs, always
through the privacy checks in cornerman.remote.kaggle.
`cornerman metrics` prints label-free perception health numbers per fight.
`cornerman compare --a DIR --b DIR` checks two runs of one fight agree.
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from cornerman.collect.run import collect
from cornerman.config import load_settings
from cornerman.pipeline.runner import run
from cornerman.pipeline.stage import Stage
from cornerman.pipeline.stages import PIPELINE
from cornerman.remote import batch, kaggle

KAGGLE_BUILD = Path("kaggle/build")
PERCEPTION_KERNEL = Path("kaggle/perception")


def pipeline_for(name: str) -> list[Stage]:
    """The stub stages, or the real perception stages (which need the gpu extra)."""
    if name == "stub":
        return PIPELINE
    from cornerman.perception.stages import REAL_PIPELINE

    return REAL_PIPELINE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cornerman")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the pipeline")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--fight", action="append", help="only this fight (repeatable)")
    p_run.add_argument("--stage", help="run only this stage and what it depends on")
    p_run.add_argument("--force", action="store_true", help="ignore the cache")
    p_run.add_argument("-v", "--verbose", action="store_true")

    p_col = sub.add_parser("collect", help="fetch fight metadata and build the registry")
    p_col.add_argument("--config", type=Path, required=True)

    p_kg = sub.add_parser("kaggle", help="push, check and pull private Kaggle jobs")
    kg = p_kg.add_subparsers(dest="kaggle_command", required=True)
    p_push = kg.add_parser("push", help="build a kernel folder and push it")
    p_push.add_argument("folder", type=Path, help="e.g. kaggle/smoke")
    p_push.add_argument(
        "--values", type=Path, help="local JSON file with build-time values (not in git)"
    )
    p_status = kg.add_parser("status", help="show a kernel's run status")
    p_status.add_argument("folder", type=Path)
    p_pull = kg.add_parser("pull", help="download a kernel's output")
    p_pull.add_argument("folder", type=Path)
    p_pull.add_argument("--dest", type=Path, required=True)
    p_batch = kg.add_parser("batch", help="push the next unprocessed fights to the GPU job")
    p_batch.add_argument("--config", type=Path, default=Path("configs/kaggle_real.yaml"))
    p_batch.add_argument("--fights-csv", type=Path, default=Path("data/fights.csv"))
    p_batch.add_argument("--n", type=int, default=0, help="next n unprocessed fights")
    p_batch.add_argument(
        "--fight", action="append", default=[], help="these fights, even if already done"
    )
    p_batch.add_argument("--overlays", type=int, default=0)
    p_import = kg.add_parser("import", help="pull the GPU job's output into data/")
    p_import.add_argument("--data", type=Path, default=Path("data"))
    p_import.add_argument(
        "--derived", type=Path, help="put fights here instead of data/derived (for comparisons)"
    )

    p_met = sub.add_parser("metrics", help="perception health numbers for processed fights")
    p_met.add_argument("--derived", type=Path, default=Path("data/derived"))
    p_met.add_argument("--fps", type=float, default=10.0)
    p_cmp = sub.add_parser("compare", help="check two runs of the same fight agree")
    p_cmp.add_argument("--a", type=Path, required=True, help="derived/{fight_id} of run A")
    p_cmp.add_argument("--b", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "kaggle":
        return _kaggle(args)
    if args.command == "collect":
        return _collect(args.config)
    if args.command == "compare":
        from cornerman.eval.compare_runs import compare_fight

        for k, v in compare_fight(args.a, args.b).items():
            print(f"{k:<18} {v:.4g}")
        return 0
    if args.command == "metrics":
        return _metrics(args.derived, args.fps)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    settings = load_settings(args.config)
    stages = pipeline_for(settings.pipeline)
    fights: list[str] = args.fight or settings.fights
    stage: str | None = args.stage
    known = {s.name for s in stages}
    if stage is not None and stage not in known:
        parser.error(f"unknown stage {stage!r}; choose from {sorted(known)}")

    results = run(stages, settings, fights, target=stage, force=args.force)
    for r in results:
        print(f"{r.fight_id:<24} {r.stage:<12} {r.status}")
    counts = Counter(r.status.value for r in results)
    print(", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    return 0


def _collect(config: Path) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tables = collect(load_settings(config).collect)
    repeat = int((tables.fighters["n_fights"] > 1).sum())
    print(f"{len(tables.fights)} fights, {len(tables.fighters)} fighters ({repeat} with 2+ fights)")
    print("skipped:")
    print(tables.skipped["reason"].str.split(":").str[0].value_counts().to_string())
    return 0


def _push(folder: Path, values: dict[str, object]) -> None:
    built = kaggle.build_kernel(folder, KAGGLE_BUILD / folder.name, values)
    print(kaggle.push_kernel(built))
    print(f"pushed {kaggle.read_kernel_metadata(folder)['id']}, confirmed private on Kaggle")


def _kaggle(args: argparse.Namespace) -> int:
    if args.kaggle_command == "batch":
        fights = pd.read_csv(args.fights_csv)
        if args.fight:
            chosen = fights[fights["fight_id"].isin(args.fight)]
        else:
            chosen = batch.select_fights(fights, Path("data/derived"), args.n)
        if chosen.empty:
            print("no fights to run")
            return 1
        print(f"{len(chosen)} fights: {', '.join(chosen['fight_id'])}")
        _push(PERCEPTION_KERNEL, batch.build_values(chosen, args.config, args.overlays))
        return 0
    if args.kaggle_command == "import":
        data: Path = args.data
        ref = str(kaggle.read_kernel_metadata(PERCEPTION_KERNEL)["id"])
        src = kaggle.pull_output(ref, KAGGLE_BUILD / "perception_output")
        derived: Path = args.derived or data / "derived"
        done = batch.import_outputs(src, derived, data / "overlays", data / "logs")
        print(f"imported {len(done)} finished fights")
        return 0
    folder: Path = args.folder
    ref = str(kaggle.read_kernel_metadata(folder)["id"])
    if args.kaggle_command == "push":
        values_file: Path | None = args.values
        values: dict[str, object] = json.loads(values_file.read_text()) if values_file else {}
        _push(folder, values)
    elif args.kaggle_command == "status":
        print(kaggle.kernel_status(ref))
    else:
        print(kaggle.pull_output(ref, args.dest))
    return 0


def _metrics(derived: Path, fps: float) -> int:
    from cornerman.eval.perception_metrics import fight_metrics

    rows: list[dict[str, object]] = []
    for fid in sorted(batch.done_fights(derived)):
        d = derived / fid
        m = fight_metrics(
            frames=pd.read_parquet(d / "tracking" / "frames.parquet"),
            tracks=pd.read_parquet(d / "tracking" / "tracks.parquet"),
            roles=pd.read_parquet(d / "identity" / "identity.parquet"),
            live=pd.read_parquet(d / "identity" / "live.parquet"),
            poses=pd.read_parquet(d / "pose" / "poses.parquet"),
            step_seconds=1 / fps,
        )
        rows.append({"fight_id": fid, **m})
    if not rows:
        print("no processed fights")
        return 0
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
