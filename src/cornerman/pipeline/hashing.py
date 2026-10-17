"""Stable content hashes for files, directories, Python objects and code.

Cache keys are built from these, so they must not depend on anything that
changes between machines or runs: dict ordering, absolute paths, timestamps.
"""

import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_CHUNK = 1 << 20


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(path: Path, exclude: frozenset[str] = frozenset()) -> str:
    """Hash every file under `path` by relative path and content.

    File names are part of the hash, so renaming a file changes it. Files whose
    name is in `exclude` (at any depth) are skipped.
    """
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.name not in exclude:
            h.update(p.relative_to(path).as_posix().encode())
            h.update(b"\0")
            h.update(hash_file(p).encode())
            h.update(b"\n")
    return h.hexdigest()


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"cannot hash object of type {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and no whitespace, so equal objects give equal text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_to_jsonable)


def hash_obj(obj: Any) -> str:
    return hash_bytes(canonical_json(obj).encode())


def hash_code(fn: Callable[..., Any]) -> str:
    """Hash a function's source text.

    This catches edits inside the function body. It does not see changes in
    helpers the function calls, which is what the stage `version` string is for.
    """
    return hash_bytes(inspect.getsource(fn).encode())
