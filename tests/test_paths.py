"""Project-root resolution must not depend on how the package was installed."""

from __future__ import annotations

import pytest

from california_housing_pulse import paths
from california_housing_pulse.paths import PROJECT_ROOT, find_project_root


def test_project_root_contains_the_marker():
    """The resolved root is a real repository checkout, not a venv directory."""
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_project_root_is_not_inside_a_virtualenv():
    """A fixed-depth walk under a non-editable install lands in .venv; guard it."""
    assert ".venv" not in PROJECT_ROOT.parts
    assert "site-packages" not in PROJECT_ROOT.parts


def test_data_directories_live_under_the_project_root():
    for directory in (paths.RAW_DIR, paths.INTERIM_DIR, paths.PROCESSED_DIR):
        assert directory.is_relative_to(PROJECT_ROOT)


def test_env_override_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("CHP_PROJECT_ROOT", str(tmp_path))
    assert find_project_root() == tmp_path.resolve()


def test_env_override_must_be_a_directory(tmp_path, monkeypatch):
    missing = tmp_path / "nope"
    monkeypatch.setenv("CHP_PROJECT_ROOT", str(missing))
    with pytest.raises(RuntimeError, match="not a directory"):
        find_project_root()


def test_missing_marker_raises_instead_of_guessing(tmp_path, monkeypatch):
    monkeypatch.delenv("CHP_PROJECT_ROOT", raising=False)
    # tmp_path is outside the repository, so no pyproject.toml exists above it.
    with pytest.raises(RuntimeError, match="Could not locate the project root"):
        find_project_root(start=tmp_path)
