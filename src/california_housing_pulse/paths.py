"""Canonical project paths.

Every path is derived from the installed package location, so the pipeline never
contains a machine-specific absolute path (Milestone 4 definition of done).
An operator may override the project root with the CHP_PROJECT_ROOT environment
variable, which is useful when raw snapshots live on an external volume.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/california_housing_pulse/paths.py -> repository root is three levels up.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_PROJECT_ROOT = _PACKAGE_ROOT.parent.parent

PROJECT_ROOT = Path(os.environ.get("CHP_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)).resolve()

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
