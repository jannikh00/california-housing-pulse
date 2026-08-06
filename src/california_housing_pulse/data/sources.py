"""Loading and representation of the source registry (``configs/sources.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ..paths import CONFIG_DIR, RAW_DIR

SOURCES_CONFIG = CONFIG_DIR / "sources.yaml"


@dataclass(frozen=True)
class Source:
    """One declared upstream input."""

    source_id: str
    role: str
    url: str
    filename: str
    format: str
    license: str
    citation: str
    notes: str
    manual_procedure: str
    required: bool = True
    approx_bytes: int | None = None

    @property
    def raw_path(self) -> Path:
        """Where ``chp fetch`` stores this source's raw snapshot."""
        return RAW_DIR / self.filename


@dataclass(frozen=True)
class SourceRegistry:
    """All declared sources plus registry-level settings."""

    user_agent: str
    sources: dict[str, Source] = field(default_factory=dict)

    def __getitem__(self, source_id: str) -> Source:
        try:
            return self.sources[source_id]
        except KeyError:
            known = ", ".join(sorted(self.sources))
            raise KeyError(f"Unknown source '{source_id}'. Known sources: {known}") from None

    def __iter__(self):
        return iter(self.sources.values())

    @property
    def ids(self) -> list[str]:
        return list(self.sources)


@lru_cache(maxsize=1)
def load_registry(config_path: Path | None = None) -> SourceRegistry:
    """Parse ``configs/sources.yaml`` into a :class:`SourceRegistry`."""
    path = Path(config_path) if config_path else SOURCES_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Source registry not found at {path}")

    raw = yaml.safe_load(path.read_text())
    sources = {
        source_id: Source(
            source_id=source_id,
            role=entry["role"],
            url=entry["url"],
            filename=entry["filename"],
            format=entry["format"],
            license=entry["license"],
            citation=entry["citation"],
            notes=entry.get("notes", ""),
            manual_procedure=entry.get("manual_procedure", ""),
            required=entry.get("required", True),
            approx_bytes=entry.get("approx_bytes"),
        )
        for source_id, entry in raw["sources"].items()
    }
    return SourceRegistry(user_agent=raw["user_agent"], sources=sources)
