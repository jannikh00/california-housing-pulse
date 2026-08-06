"""Canonical project paths.

The project root is located by searching upward for the ``pyproject.toml``
marker rather than by walking a fixed number of parent directories. That matters
because the package may be imported either from ``src/`` (editable install) or
from ``.venv/lib/pythonX.Y/site-packages/`` (non-editable install); a fixed-depth
walk resolves to a different, wrong directory in the second case, which would
place the entire ``data/`` tree inside the virtual environment.

An operator may override the root with the ``CHP_PROJECT_ROOT`` environment
variable, which is useful when raw snapshots live on an external volume. If no
marker is found and no override is set, this module raises rather than guessing.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_MARKER = "pyproject.toml"


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor of ``start`` that contains ``pyproject.toml``.

    ``CHP_PROJECT_ROOT`` takes precedence when set. Raises ``RuntimeError`` if no
    marker is found, so a misconfigured environment fails loudly instead of
    silently writing data to the wrong place.
    """
    override = os.environ.get("CHP_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(
                f"CHP_PROJECT_ROOT is set to '{override}', which is not a directory."
            )
        return root

    origin = Path(start).resolve() if start else Path(__file__).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / PROJECT_MARKER).is_file():
            return candidate

    raise RuntimeError(
        f"Could not locate the project root: no '{PROJECT_MARKER}' found in "
        f"'{origin}' or any parent directory. Set CHP_PROJECT_ROOT to the "
        "repository root to override."
    )


PROJECT_ROOT = find_project_root()

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_MANIFEST = RAW_DIR / "manifest.json"

_WRITABLE_DIRS = (
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    SNAPSHOT_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
)


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for directory in _WRITABLE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    """Render a path relative to the project root for logs and metadata."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
