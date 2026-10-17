"""The data/ allow-list must keep anything with real names out of git."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, check=False
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.mark.parametrize(
    "path",
    [
        "data/fights.csv",
        "data/fighters.csv",
        "data/raw/fight.json",
        "data/derived/x/events.parquet",
        "data/quarantine/x.json",
        "data/anything_new.csv",
        "clip.mp4",
    ],
)
def test_private_data_is_ignored(path: str) -> None:
    assert ignored(path)


@pytest.mark.parametrize(
    "path",
    ["data/.gitignore", "data/derived.dvc", "data/fixtures/fixture_fight/meta.json"],
)
def test_allowed_data_is_tracked(path: str) -> None:
    assert not ignored(path)
