from pathlib import Path

import pytest

from src.core.sources_loader import _FILENAME, load


def _write_sources(task_dir: Path, content: str) -> None:
    (task_dir / _FILENAME).write_text(content)


# ----- load: file present -----


def test_load_returns_parsed_list_when_sources_yaml_exists(tmp_path):
    _write_sources(tmp_path, "- a\n- b\n")
    assert load(tmp_path) == ["a", "b"]


# ----- load: missing file / empty file fall back to default -----


@pytest.mark.parametrize(
    "default",
    [None, ["fallback"]],
)
def test_load_missing_file_returns_default(tmp_path, default):
    assert not (tmp_path / _FILENAME).exists()
    assert load(tmp_path, default=default) is default


def test_load_missing_file_default_default_is_none(tmp_path):
    assert load(tmp_path) is None


def test_load_empty_yaml_returns_default_not_none(tmp_path):
    _write_sources(tmp_path, "")
    default = ["fallback"]
    assert load(tmp_path, default=default) is default
