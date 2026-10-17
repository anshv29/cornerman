from pathlib import Path

import pytest

from cornerman.config import Settings, load_settings

ROOT = Path(__file__).resolve().parents[2]


def test_fixture_config_loads() -> None:
    s = load_settings(ROOT / "configs" / "fixture.yaml")
    assert s.seed == 7
    assert s.fights == ["fixture_fight"]
    assert s.stub.fps == 10


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORNERMAN_STUB__FPS", "30")
    assert Settings().stub.fps == 30


def test_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        Settings(stub={"fps": -1})  # pyright: ignore[reportArgumentType]
