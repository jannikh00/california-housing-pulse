"""The column registry (``configs/columns.yaml``).

The registry declares what each panel column *should* be — source, meaning,
unit, and the two bound families. Nothing observed is stored here; measured
statistics come from the built panel in :mod:`.dictionary`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from ..paths import CONFIG_DIR

COLUMNS_CONFIG = CONFIG_DIR / "columns.yaml"


@dataclass(frozen=True)
class ColumnSpec:
    """Declared intent for one panel column."""

    name: str
    source: str
    meaning: str
    unit: str
    hard: tuple[float, float] | None = None
    plausible: tuple[float, float] | None = None

    @property
    def is_bounded(self) -> bool:
        return self.hard is not None


@lru_cache(maxsize=1)
def load_columns(config_path: Path | None = None) -> dict[str, ColumnSpec]:
    """Parse ``configs/columns.yaml`` into ordered :class:`ColumnSpec` entries."""
    path = Path(config_path) if config_path else COLUMNS_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Column registry not found at {path}")

    raw = yaml.safe_load(path.read_text())
    specs: dict[str, ColumnSpec] = {}
    for name, entry in raw["columns"].items():
        hard = entry.get("hard")
        plausible = entry.get("plausible")
        specs[name] = ColumnSpec(
            name=name,
            source=entry["source"],
            meaning=" ".join(entry["meaning"].split()),
            unit=entry["unit"],
            hard=tuple(hard) if hard else None,
            plausible=tuple(plausible) if plausible else None,
        )
    return specs


def bounded_columns() -> dict[str, ColumnSpec]:
    """Only the columns that declare numeric bounds."""
    return {name: spec for name, spec in load_columns().items() if spec.is_bounded}
