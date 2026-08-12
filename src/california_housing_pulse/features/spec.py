"""Parsing of ``configs/features.yaml`` into typed feature specifications.

The split of responsibility mirrors :mod:`features.target`: the YAML is the
reviewable declaration of *what* exists, this module turns it into objects, and
``features.build`` executes it. Nothing here computes a feature value.

The one piece of real logic is :attr:`FeatureSpec.effective_lag`, which adds a
source's publication lag to whatever offset a feature declares. Because the
builder can only reach a column through a :class:`FeatureSpec`, there is no path
by which a feature bypasses its source's lag — the leakage guarantee is
structural rather than a rule someone has to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ..paths import CONFIG_DIR
from .transforms import TRANSFORMS, history_depth

FEATURES_CONFIG = CONFIG_DIR / "features.yaml"


@dataclass(frozen=True)
class SourceTiming:
    """When a source's values become knowable, relative to the reference month."""

    source_id: str
    release_lag_months: int
    rationale: str


@dataclass(frozen=True)
class MissingPolicy:
    """What to do about gaps in one source's columns."""

    source_id: str
    method: str
    max_gap_months: int
    indicator: str | None
    rationale: str


@dataclass(frozen=True)
class FeatureSpec:
    """One generated feature column."""

    name: str
    column: str
    source: str
    family: str
    transform: str
    param: int
    offset: int
    release_lag_months: int

    @property
    def effective_lag(self) -> int:
        """Months the feature is shifted before its own window is applied."""
        return self.release_lag_months + self.offset

    @property
    def reach_months(self) -> int:
        """Oldest month the feature reads, counted back from the reference month.

        This is what makes the availability table honest: a 12-month rolling
        standard deviation of a two-month-lagged series reads 13 months of
        history, and a reader should not have to work that out themselves.
        """
        return self.effective_lag + history_depth(self.transform, self.param)

    def describe(self) -> str:
        """One-line human summary, used in the generated availability table."""
        lag = self.effective_lag
        window = {
            "lag": f"level at t-{lag + self.param}",
            "diff": f"change from t-{lag + self.param} to t-{lag}",
            "log_diff": f"log growth from t-{lag + self.param} to t-{lag}",
            "rollmean": f"{self.param}-month mean ending t-{lag}",
            "rollstd": f"{self.param}-month SD ending t-{lag}",
            "reltrend": f"t-{lag} vs its {self.param}-month mean",
        }[self.transform]
        return f"{self.column}: {window}"


@dataclass(frozen=True)
class FeatureContract:
    """The whole parsed specification."""

    sources: dict[str, SourceTiming]
    missing: dict[str, MissingPolicy]
    excluded_columns: dict[str, str]
    features: tuple[FeatureSpec, ...] = field(default_factory=tuple)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.features)

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Panel columns the specification reads, in first-appearance order."""
        seen: dict[str, None] = {}
        for spec in self.features:
            seen.setdefault(spec.column, None)
        return tuple(seen)

    @property
    def max_reach_months(self) -> int:
        """History the deepest feature needs — the feature-side lead-in."""
        return max((spec.reach_months for spec in self.features), default=0)

    def by_family(self) -> dict[str, list[FeatureSpec]]:
        grouped: dict[str, list[FeatureSpec]] = {}
        for spec in self.features:
            grouped.setdefault(spec.family, []).append(spec)
        return grouped


def _build_name(column: str, transform: str, param: int, offset: int) -> str:
    suffix = f"_o{offset}" if offset else ""
    return f"{column}__{transform}{param}{suffix}"


@lru_cache(maxsize=1)
def load_feature_contract(config_path: Path | None = None) -> FeatureContract:
    """Parse ``configs/features.yaml``.

    Every reference is validated at parse time — unknown transform names, unknown
    sources, a feature built on an explicitly excluded column, or a duplicate
    output name all raise here rather than producing a silently wrong matrix.
    """
    path = Path(config_path) if config_path else FEATURES_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Feature specification not found at {path}")

    raw = yaml.safe_load(path.read_text())

    sources = {
        source_id: SourceTiming(
            source_id=source_id,
            release_lag_months=int(entry["release_lag_months"]),
            rationale=str(entry.get("rationale", "")).strip(),
        )
        for source_id, entry in raw["sources"].items()
    }

    missing_raw = dict(raw.get("missing") or {})
    excluded = {
        str(entry["column"]): str(entry.get("rationale", "")).strip()
        for entry in missing_raw.pop("excluded_columns", []) or []
    }
    missing = {
        source_id: MissingPolicy(
            source_id=source_id,
            method=str(entry["method"]),
            max_gap_months=int(entry.get("max_gap_months", 0)),
            indicator=entry.get("indicator"),
            rationale=str(entry.get("rationale", "")).strip(),
        )
        for source_id, entry in missing_raw.items()
    }

    features: list[FeatureSpec] = []
    seen: set[str] = set()
    for entry in raw["features"]:
        column = str(entry["column"])
        source_id = str(entry["source"])
        if column in excluded:
            raise ValueError(
                f"Feature specification builds on '{column}', which the same file "
                "lists under missing.excluded_columns."
            )
        if source_id not in sources:
            raise ValueError(
                f"Feature on '{column}' names source '{source_id}', which has no "
                f"entry under `sources`. Known sources: {', '.join(sorted(sources))}."
            )
        offset = int(entry.get("offset", 0))
        for block in entry["transforms"]:
            transform = str(block["transform"])
            if transform not in TRANSFORMS:
                raise ValueError(
                    f"Feature on '{column}' names transform '{transform}'; known "
                    f"transforms are {', '.join(sorted(TRANSFORMS))}."
                )
            for param in block["params"]:
                param = int(param)
                if param < 0:
                    raise ValueError(
                        f"Feature '{column}' {transform} has a negative parameter "
                        f"{param}; transforms are backward-looking only."
                    )
                name = _build_name(column, transform, param, offset)
                if name in seen:
                    raise ValueError(f"Duplicate feature name '{name}' in {path.name}.")
                seen.add(name)
                features.append(
                    FeatureSpec(
                        name=name,
                        column=column,
                        source=source_id,
                        family=str(entry.get("family", "other")),
                        transform=transform,
                        param=param,
                        offset=offset,
                        release_lag_months=sources[source_id].release_lag_months,
                    )
                )

    return FeatureContract(
        sources=sources,
        missing=missing,
        excluded_columns=excluded,
        features=tuple(features),
    )
