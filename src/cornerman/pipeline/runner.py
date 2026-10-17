"""A small DAG runner with content-hash caching.

For each fight, stages run in dependency order. Each stage's cache key is a
hash of everything that can change its output: its name, version, code, the
config fields it reads, the seed, its external source files, and the output
hashes of its upstream stages. If the key matches the one saved in the stage's
manifest and the saved files are intact, the stage is skipped.

Outputs are written to a temporary folder and renamed into place only when the
stage finishes, so a killed session never leaves a half-written stage that
looks complete.
"""

import json
import logging
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import reduce
from pathlib import Path
from typing import Any

from cornerman.config import Settings
from cornerman.pipeline.hashing import hash_code, hash_dir, hash_file, hash_obj
from cornerman.pipeline.stage import QuarantineError, Stage, StageContext

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"


class Status(StrEnum):
    RAN = "ran"
    CACHED = "cached"
    QUARANTINED = "quarantined"
    BLOCKED = "blocked"
    """An upstream stage of this fight was quarantined."""


@dataclass(frozen=True)
class StageResult:
    fight_id: str
    stage: str
    status: Status


def topo_sort(stages: Sequence[Stage]) -> list[Stage]:
    by_name = {s.name: s for s in stages}
    if len(by_name) != len(stages):
        raise ValueError("duplicate stage names")
    order: list[Stage] = []
    state: dict[str, int] = {}  # 1 = visiting, 2 = done

    def visit(name: str, path: tuple[str, ...]) -> None:
        if name not in by_name:
            raise ValueError(f"unknown stage {name!r} (needed by {path[-1]!r})")
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise ValueError(f"cycle: {' -> '.join((*path, name))}")
        state[name] = 1
        for dep in by_name[name].deps:
            visit(dep, (*path, name))
        state[name] = 2
        order.append(by_name[name])

    for s in stages:
        visit(s.name, ())
    return order


def ancestors(stages: Sequence[Stage], name: str) -> set[str]:
    by_name = {s.name: s for s in stages}
    seen: set[str] = set()
    todo = [name]
    while todo:
        for dep in by_name[todo.pop()].deps:
            if dep not in seen:
                seen.add(dep)
                todo.append(dep)
    return seen


def _config_value(settings: Settings, dotted: str) -> Any:
    return reduce(getattr, dotted.split("."), settings)


def stage_seed(global_seed: int, fight_id: str, stage: str) -> int:
    return int(hash_obj([global_seed, fight_id, stage])[:16], 16)


def stage_dir(settings: Settings, fight_id: str, stage: str) -> Path:
    return settings.paths.derived / fight_id / stage


def read_manifest(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / MANIFEST
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _cache_key(stage: Stage, settings: Settings, fight_id: str, dep_hashes: dict[str, str]) -> str:
    sources: Sequence[Path] = stage.sources(fight_id, settings) if stage.sources else ()
    return hash_obj(
        {
            "stage": stage.name,
            "version": stage.version,
            "code": hash_code(stage.fn),
            "config": {k: _config_value(settings, k) for k in stage.config_keys},
            "seed": settings.seed,
            "fight": fight_id,
            "deps": dep_hashes,
            "sources": [hash_file(p) for p in sources],
        }
    )


def _is_valid(out_dir: Path, key: str) -> str | None:
    """Return the saved output hash if the cache entry matches `key` and is intact."""
    manifest = read_manifest(out_dir)
    if manifest is None or manifest.get("key") != key:
        return None
    output_hash = str(manifest["output_hash"])
    if hash_dir(out_dir, exclude=frozenset({MANIFEST})) != output_hash:
        log.warning("%s: files changed since they were written, rerunning", out_dir)
        return None
    return output_hash


def _quarantine(settings: Settings, fight_id: str, stage: str, reason: str) -> None:
    settings.paths.quarantine.mkdir(parents=True, exist_ok=True)
    record = {
        "fight_id": fight_id,
        "stage": stage,
        "reason": reason,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (settings.paths.quarantine / f"{fight_id}.json").write_text(json.dumps(record, indent=2))


def _run_stage(stage: Stage, ctx: StageContext, key: str) -> str:
    tmp = ctx.out_dir.with_name(ctx.out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        stage.fn(StageContext(ctx.fight_id, ctx.settings, ctx.inputs, tmp, ctx.seed))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    output_hash = hash_dir(tmp)
    manifest = {"stage": stage.name, "key": key, "output_hash": output_hash}
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    shutil.rmtree(ctx.out_dir, ignore_errors=True)
    tmp.rename(ctx.out_dir)
    return output_hash


def run(
    stages: Sequence[Stage],
    settings: Settings,
    fight_ids: Iterable[str],
    target: str | None = None,
    force: bool = False,
) -> list[StageResult]:
    """Run the pipeline.

    With `target`, only that stage and what it depends on run, and `force`
    reruns just the target. Without it, every stage runs and `force` reruns all.
    """
    ordered = topo_sort(stages)
    if target is not None:
        wanted = ancestors(stages, target) | {target}
        ordered = [s for s in ordered if s.name in wanted]

    results: list[StageResult] = []
    for fight_id in fight_ids:
        output_hashes: dict[str, str] = {}
        blocked = False
        for stage in ordered:
            if blocked:
                results.append(StageResult(fight_id, stage.name, Status.BLOCKED))
                continue
            out_dir = stage_dir(settings, fight_id, stage.name)
            dep_hashes = {d: output_hashes[d] for d in stage.deps}
            key = _cache_key(stage, settings, fight_id, dep_hashes)
            forced = force and (target is None or stage.name == target)
            cached = None if forced else _is_valid(out_dir, key)
            if cached is not None:
                output_hashes[stage.name] = cached
                results.append(StageResult(fight_id, stage.name, Status.CACHED))
                log.info("%s/%s: cached", fight_id, stage.name)
                continue
            ctx = StageContext(
                fight_id=fight_id,
                settings=settings,
                inputs={d: stage_dir(settings, fight_id, d) for d in stage.deps},
                out_dir=out_dir,
                seed=stage_seed(settings.seed, fight_id, stage.name),
            )
            try:
                output_hashes[stage.name] = _run_stage(stage, ctx, key)
            except QuarantineError as e:
                _quarantine(settings, fight_id, stage.name, str(e))
                results.append(StageResult(fight_id, stage.name, Status.QUARANTINED))
                log.warning("%s/%s: quarantined: %s", fight_id, stage.name, e)
                blocked = True
                continue
            results.append(StageResult(fight_id, stage.name, Status.RAN))
            log.info("%s/%s: ran", fight_id, stage.name)
        if not blocked:
            # The fight now passes, so an old quarantine record is stale.
            (settings.paths.quarantine / f"{fight_id}.json").unlink(missing_ok=True)
    return results
