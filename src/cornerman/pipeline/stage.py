"""What a pipeline stage is, and what it gets handed when it runs."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cornerman.config import Settings


class QuarantineError(Exception):
    """Raised by a stage when a fight's data is bad.

    The runner records the reason in the quarantine folder and skips the rest
    of that fight's stages, instead of silently dropping it or crashing the run.
    """


@dataclass(frozen=True)
class StageContext:
    fight_id: str
    settings: Settings
    inputs: Mapping[str, Path]
    """Output folder of each upstream stage, by stage name."""
    out_dir: Path
    """Where the stage writes. It is empty when the stage starts."""
    seed: int
    """Derived from the global seed, the fight and the stage name."""

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


StageFn = Callable[[StageContext], None]
SourceFn = Callable[[str, Settings], Sequence[Path]]


@dataclass(frozen=True)
class Stage:
    name: str
    fn: StageFn
    deps: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    """Dotted config fields this stage reads, e.g. "stub.fps". Only these go
    into the cache key, so changing an unrelated field doesn't rerun it."""
    version: str = "1"
    """Bump to force a rerun when something the code hash can't see changes."""
    sources: SourceFn | None = field(default=None)
    """Files outside the pipeline this stage reads, e.g. raw fight metadata."""
