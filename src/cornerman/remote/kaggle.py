"""Every call to the Kaggle CLI goes through this module.

The one rule it enforces: nothing we upload to Kaggle is ever public. Kernels
run on fight footage and datasets may hold raw video for a few hours, so a
public one would redistribute video (hard rule 2) and could expose real
fighter names. Checks happen in code before anything is sent:
- no command may carry `--public` / `-u`;
- a kernel folder must have `"is_private": true` in its metadata;
- after a kernel push we pull its metadata back from Kaggle and check again;
- after any kernel push or dataset upload we request it anonymously (no
  credentials) and fail unless Kaggle refuses, which is what private means.

The CLI itself runs through uvx at a pinned version, so it never needs to be
installed globally and the laptop and docs agree on the version.
"""

import json
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

KAGGLE_CLI = ("uvx", "--from", "kaggle==2.2.4", "kaggle")
KERNEL_METADATA = "kernel-metadata.json"
BUILD_MARK = "# BUILD:"
_PUBLIC_FLAGS = frozenset({"-u", "--public"})

Runner = Callable[[Sequence[str]], str]
Prober = Callable[[str], int]
"""Returns the HTTP status of an anonymous GET of a URL."""


class PrivacyError(Exception):
    """Something was about to be made public on Kaggle. Never caught on purpose."""


class KaggleError(Exception):
    """The Kaggle CLI returned an error."""


def _run_cli(args: Sequence[str]) -> str:
    result = subprocess.run([*KAGGLE_CLI, *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise KaggleError(f"kaggle {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result.stdout


def check_kernel_metadata(meta: Mapping[str, object]) -> None:
    """Raise unless the metadata says, explicitly, that the kernel is private."""
    if meta.get("is_private") is not True:
        raise PrivacyError(
            f'kernel {meta.get("id")!r} must set "is_private": true, got {meta.get("is_private")!r}'
        )


def read_kernel_metadata(folder: Path) -> dict[str, object]:
    meta: dict[str, object] = json.loads((folder / KERNEL_METADATA).read_text())
    return meta


def check_args(args: Sequence[str]) -> None:
    """Refuse public flags, and check kernel metadata before any kernel push."""
    public = _PUBLIC_FLAGS.intersection(args)
    if public:
        raise PrivacyError(f"refusing to pass {sorted(public)} to kaggle")
    if list(args[:2]) == ["kernels", "push"]:
        folder = Path(_flag_value(args, "-p", "--path") or ".")
        check_kernel_metadata(read_kernel_metadata(folder))


def _flag_value(args: Sequence[str], short: str, long: str) -> str | None:
    for i, a in enumerate(args):
        if a in (short, long) and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(long + "="):
            return a.split("=", 1)[1]
    return None


def kaggle(args: Sequence[str], runner: Runner = _run_cli) -> str:
    """Run one Kaggle CLI command after the privacy checks."""
    check_args(args)
    return runner(args)


def anonymous_status(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "cornerman-privacy-check"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as e:
        return int(e.code)


def check_not_public(url: str, probe: Prober = anonymous_status) -> None:
    """Fail if anyone without credentials can see this URL."""
    status = probe(url)
    if status == 200:
        raise PrivacyError(f"{url} is visible without logging in")


def build_kernel(
    src: Path,
    dest: Path,
    values: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Copy a kernel folder and fill in its build-time values.

    A line in the code file like `VIDEO_URLS: list[str] = []  # BUILD:VIDEO_URLS`
    becomes `VIDEO_URLS = [...]`, the value written as a Python literal. This keeps things
    such as video links out of git: they only exist in the private kernel.
    """
    meta = read_kernel_metadata(src)
    check_kernel_metadata(meta)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    if metadata:
        # e.g. dataset_sources for this batch. Privacy is checked again after merging.
        merged = {**meta, **metadata}
        check_kernel_metadata(merged)
        (dest / KERNEL_METADATA).write_text(json.dumps(merged, indent=2))
    code = dest / str(meta["code_file"])
    lines: list[str] = []
    missing = set(values)
    for line in code.read_text().splitlines():
        if BUILD_MARK in line:
            name = line.split(BUILD_MARK, 1)[1].strip()
            if name not in values:
                raise KeyError(f"no build value for {name!r} in {code}")
            indent = line[: len(line) - len(line.lstrip())]
            # repr, not JSON: JSON writes true/false/null, which aren't Python.
            line = f"{indent}{name} = {values[name]!r}"
            missing.discard(name)
        lines.append(line)
    if missing:
        raise KeyError(f"build values {sorted(missing)} have no {BUILD_MARK} line in {code}")
    code.write_text("\n".join(lines) + "\n")
    return dest


DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
"""Asked for explicitly: a push without it once sat queued for over an hour."""


def push_kernel(
    folder: Path,
    runner: Runner = _run_cli,
    accelerator: str = DEFAULT_ACCELERATOR,
    probe: Prober = anonymous_status,
) -> str:
    """Push a kernel, then confirm on Kaggle that it is private, two ways."""
    meta = read_kernel_metadata(folder)
    out = kaggle(["kernels", "push", "-p", str(folder), "--accelerator", accelerator], runner)
    verify_kernel_private(str(meta["id"]), runner)
    check_not_public(f"https://www.kaggle.com/code/{meta['id']}", probe)
    return out


def verify_kernel_private(ref: str, runner: Runner = _run_cli) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kaggle(["kernels", "pull", ref, "-p", tmp, "-m"], runner)
        check_kernel_metadata(read_kernel_metadata(Path(tmp)))


def kernel_status(ref: str, runner: Runner = _run_cli) -> str:
    return kaggle(["kernels", "status", ref], runner).strip()


def pull_output(ref: str, dest: Path, runner: Runner = _run_cli) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    kaggle(["kernels", "output", ref, "-p", str(dest)], runner)
    return dest


DATASET_METADATA = "dataset-metadata.json"


def dataset_exists(ref: str, runner: Runner = _run_cli) -> bool:
    try:
        kaggle(["datasets", "status", ref], runner)
    except KaggleError:
        return False
    return True


def upload_dataset(
    folder: Path,
    ref: str,
    runner: Runner = _run_cli,
    probe: Prober = anonymous_status,
    wait_s: float = 600,
) -> None:
    """Create (or add a version of) a private dataset, wait until it's ready, check privacy.

    Kaggle makes datasets private unless --public is passed, and kaggle() refuses
    that flag. The anonymous check afterwards confirms it.
    """
    if dataset_exists(ref, runner):
        kaggle(["datasets", "version", "-p", str(folder), "-m", "batch", "-q"], runner)
    else:
        kaggle(["datasets", "create", "-p", str(folder), "-q"], runner)
    deadline = time.monotonic() + wait_s
    while True:
        try:
            # Right after an upload Kaggle can answer 403 while it processes the files.
            if "ready" in kaggle(["datasets", "status", ref], runner).lower():
                break
        except KaggleError:
            pass
        if time.monotonic() > deadline:
            raise KaggleError(f"dataset {ref} not ready after {wait_s:.0f}s")
        time.sleep(10)
    check_not_public(f"https://www.kaggle.com/api/v1/datasets/view/{ref}", probe)


def delete_dataset(ref: str, runner: Runner = _run_cli) -> None:
    kaggle(["datasets", "delete", ref, "-y"], runner)
