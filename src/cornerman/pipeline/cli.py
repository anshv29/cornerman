"""Command line entry point: `cornerman run --config configs/fixture.yaml`."""

import argparse
import logging
from collections import Counter
from pathlib import Path

from cornerman.config import load_settings
from cornerman.pipeline.runner import run
from cornerman.pipeline.stages import PIPELINE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cornerman")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the pipeline")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--fight", action="append", help="only this fight (repeatable)")
    p_run.add_argument("--stage", help="run only this stage and what it depends on")
    p_run.add_argument("--force", action="store_true", help="ignore the cache")
    p_run.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
