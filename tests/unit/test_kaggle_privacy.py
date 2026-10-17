"""Nothing may be made public on Kaggle. These tests pin the checks."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from cornerman.remote.kaggle import (
    KaggleError,
    PrivacyError,
    build_kernel,
    check_args,
    check_kernel_metadata,
    kaggle,
    push_kernel,
)

ROOT = Path(__file__).resolve().parents[2]


def _kernel(folder: Path, is_private: object) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"id": "someone/k", "code_file": "job.py", "is_private": is_private}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta))
    (folder / "job.py").write_text("URLS: list[str] = []  # BUILD:URLS\nprint(URLS)\n")
    return folder


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_kernel_must_be_explicitly_private(value: object) -> None:
    with pytest.raises(PrivacyError):
        check_kernel_metadata({"id": "x", "is_private": value})


def test_missing_is_private_is_refused() -> None:
    with pytest.raises(PrivacyError):
        check_kernel_metadata({"id": "x"})


@pytest.mark.parametrize(
    "args",
    [
        ["datasets", "create", "-p", "x", "-u"],
        ["datasets", "create", "--public", "-p", "x"],
        ["datasets", "version", "-p", "x", "-m", "v2", "-u"],
    ],
)
def test_public_flags_are_refused(args: list[str]) -> None:
    with pytest.raises(PrivacyError):
        check_args(args)


def test_push_checks_metadata_before_running(tmp_path: Path) -> None:
    folder = _kernel(tmp_path / "k", is_private=False)
    calls: list[Sequence[str]] = []

    def runner(args: Sequence[str]) -> str:
        calls.append(args)
        return ""

    with pytest.raises(PrivacyError):
        kaggle(["kernels", "push", "-p", str(folder)], runner)
    assert calls == []


def test_push_verifies_privacy_on_kaggle_after_upload(tmp_path: Path) -> None:
    folder = _kernel(tmp_path / "k", is_private=True)

    def runner(args: Sequence[str]) -> str:
        if args[:2] == ["kernels", "pull"]:
            # Kaggle reports the kernel as public: the push must fail loudly.
            _kernel(Path(args[args.index("-p") + 1]), is_private=False)
        return ""

    with pytest.raises(PrivacyError):
        push_kernel(folder, runner)


def test_build_fills_values_and_keeps_privacy(tmp_path: Path) -> None:
    src = _kernel(tmp_path / "src", is_private=True)
    out = build_kernel(src, tmp_path / "out", {"URLS": ["a", "b"]})
    assert (out / "job.py").read_text().splitlines()[0] == "URLS = ['a', 'b']"
    assert "# BUILD:URLS" in (src / "job.py").read_text()


def test_build_refuses_public_kernel(tmp_path: Path) -> None:
    src = _kernel(tmp_path / "src", is_private=False)
    with pytest.raises(PrivacyError):
        build_kernel(src, tmp_path / "out", {"URLS": []})


def test_build_needs_every_value(tmp_path: Path) -> None:
    src = _kernel(tmp_path / "src", is_private=True)
    with pytest.raises(KeyError):
        build_kernel(src, tmp_path / "out", {})
    with pytest.raises(KeyError):
        build_kernel(src, tmp_path / "out", {"URLS": [], "EXTRA": 1})


def test_every_kernel_in_the_repo_is_private() -> None:
    found = list((ROOT / "kaggle").rglob("kernel-metadata.json"))
    assert found, "expected at least one kernel under kaggle/"
    for path in found:
        check_kernel_metadata(json.loads(path.read_text()))


def test_anonymous_200_means_public() -> None:
    from cornerman.remote.kaggle import check_not_public

    with pytest.raises(PrivacyError):
        check_not_public("https://x", probe=lambda url: 200)
    check_not_public("https://x", probe=lambda url: 403)
    check_not_public("https://x", probe=lambda url: 404)


def test_push_fails_if_kernel_is_visible_anonymously(tmp_path: Path) -> None:
    folder = _kernel(tmp_path / "k", is_private=True)

    def runner(args: Sequence[str]) -> str:
        if args[:2] == ["kernels", "pull"]:
            _kernel(Path(args[args.index("-p") + 1]), is_private=True)
        return ""

    with pytest.raises(PrivacyError):
        push_kernel(folder, runner, probe=lambda url: 200)
    push_kernel(folder, runner, probe=lambda url: 404)


def test_metadata_override_cannot_make_kernel_public(tmp_path: Path) -> None:
    src = _kernel(tmp_path / "src", is_private=True)
    with pytest.raises(PrivacyError):
        build_kernel(src, tmp_path / "out", {"URLS": []}, metadata={"is_private": False})
    out = build_kernel(
        src, tmp_path / "out", {"URLS": []}, metadata={"dataset_sources": ["me/videos"]}
    )
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["me/videos"]
    assert meta["is_private"] is True


def test_dataset_upload_is_private_and_checked(tmp_path: Path) -> None:
    from cornerman.remote.kaggle import upload_dataset

    calls: list[Sequence[str]] = []

    def runner(args: Sequence[str]) -> str:
        calls.append(args)
        if list(args[:2]) == ["datasets", "status"]:
            if len(calls) == 1:
                raise KaggleError("not found")  # first status: dataset doesn't exist yet
            return "ready"
        return ""

    upload_dataset(tmp_path, "me/videos", runner, probe=lambda url: 403)
    assert ["datasets", "create", "-p", str(tmp_path), "-q"] in [list(c) for c in calls]
    assert not any("-u" in c or "--public" in c for c in calls)

    calls.clear()
    with pytest.raises(PrivacyError):
        upload_dataset(tmp_path, "me/videos", runner, probe=lambda url: 200)


def test_built_values_are_valid_python(tmp_path: Path) -> None:
    import ast

    src = _kernel(tmp_path / "src", is_private=True)
    value = {"half": True, "none": None, "n": 1.5, "nested": [{"x": False}], "s": "it's"}
    out = build_kernel(src, tmp_path / "out", {"URLS": value})
    line = (out / "job.py").read_text().splitlines()[0]
    assert ast.literal_eval(line.split(" = ", 1)[1]) == value
