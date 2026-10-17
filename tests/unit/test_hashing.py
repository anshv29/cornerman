from pathlib import Path

from cornerman.pipeline.hashing import hash_dir, hash_obj


def test_hash_obj_ignores_key_order() -> None:
    assert hash_obj({"a": 1, "b": [1, 2]}) == hash_obj({"b": [1, 2], "a": 1})
    assert hash_obj({"a": 1}) != hash_obj({"a": 2})


def test_hash_dir_sees_names_and_content(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("hello")
    h1 = hash_dir(tmp_path)
    (tmp_path / "x.txt").rename(tmp_path / "y.txt")
    h2 = hash_dir(tmp_path)
    (tmp_path / "y.txt").write_text("hello!")
    h3 = hash_dir(tmp_path)
    assert len({h1, h2, h3}) == 3


def test_hash_dir_exclude(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("d")
    before = hash_dir(tmp_path, exclude=frozenset({"manifest.json"}))
    (tmp_path / "manifest.json").write_text("{}")
    assert hash_dir(tmp_path, exclude=frozenset({"manifest.json"})) == before
