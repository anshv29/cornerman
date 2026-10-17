"""Command line entry point.

`cornerman run --config configs/fixture.yaml` runs the pipeline.
`cornerman kaggle push|status|pull` drives Kaggle jobs, always through the
privacy checks in cornerman.remote.kaggle.
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from cornerman.config import load_settings
from cornerman.pipeline.runner import run
from cornerman.pipeline.stages import PIPELINE
from cornerman.remote import kaggle

KAGGLE_BUILD = Path("kaggle/build")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cornerman")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the pipeline")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--fight", action="append", help="only this fight (repeatable)")
    p_run.add_argument("--stage", help="run only this stage and what it depends on")
    p_run.add_argument("--force", action="store_true", help="ignore the cache")
    p_run.add_argument("-v", "--verbose", action="store_true")

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
    args = parser.parse_args(argv)

    if args.command == "kaggle":
        return _kaggle(args)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    settings = load_settings(args.config)
    fights: list[str] = args.fight or settings.fights
    stage: str | None = args.stage
    known = {s.name for s in PIPELINE}
    if stage is not None and stage not in known:
        parser.error(f"unknown stage {stage!r}; choose from {sorted(known)}")

    results = run(PIPELINE, settings, fights, target=stage, force=args.force)
    for r in results:
        print(f"{r.fight_id:<24} {r.stage:<12} {r.status}")
    counts = Counter(r.status.value for r in results)
    print(", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    return 0


def _kaggle(args: argparse.Namespace) -> int:
    folder: Path = args.folder
    ref = str(kaggle.read_kernel_metadata(folder)["id"])
    if args.kaggle_command == "push":
        values_file: Path | None = args.values
        values: dict[str, object] = json.loads(values_file.read_text()) if values_file else {}
        built = kaggle.build_kernel(folder, KAGGLE_BUILD / folder.name, values)
        print(kaggle.push_kernel(built))
        print(f"pushed {ref}, confirmed private on Kaggle")
    elif args.kaggle_command == "status":
        print(kaggle.kernel_status(ref))
    else:
        print(kaggle.pull_output(ref, args.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
