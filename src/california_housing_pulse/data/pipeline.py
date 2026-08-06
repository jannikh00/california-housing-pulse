"""The single documented rebuild: raw snapshots -> staged tables -> panel -> checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..io import write_parquet
from ..paths import SNAPSHOT_DIR, ensure_dirs, relative
from .panel import build_panel
from .sources import SourceRegistry, load_registry
from .staging import build_staged_tables
from .validate import ValidationReport, validate_panel, write_report

# The committed California slices. These are small enough to version, so the
# repository rebuilds without re-downloading the 241 MB national Redfin file.
SNAPSHOT_TABLES = ("counties", "redfin", "mortgage_rate", "unemployment")


@dataclass
class BuildResult:
    panel: pd.DataFrame
    validation: ValidationReport
    report_path: str


def write_snapshots(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Persist the committed California-only slices."""
    written = []
    for name in SNAPSHOT_TABLES:
        path = write_parquet(tables[name], SNAPSHOT_DIR / f"{name}.parquet", quiet=True)
        written.append(relative(path))
    return written


def build(
    registry: SourceRegistry | None = None,
    *,
    write: bool = True,
) -> BuildResult:
    """Rebuild every processed artifact from the retained raw snapshots."""
    ensure_dirs()
    registry = registry or load_registry()

    print("[1/4] Staging sources …")
    tables, staging_reports = build_staged_tables(registry, write=write)
    staging_summaries = [report.summary() for report in staging_reports]
    for summary in staging_summaries:
        print(summary)

    print("\n[2/4] Joining onto a complete county-month spine …")
    panel, join_report = build_panel(tables, write=write)
    print(join_report.summary())

    print("\n[3/4] Validating …")
    validation = validate_panel(panel)
    print(validation.summary())

    print("\n[4/4] Writing snapshots and report …")
    report_path = ""
    if write:
        for path in write_snapshots(tables):
            print(f"  snapshot {path}")
        report_path = write_report(
            panel,
            validation,
            staging_summaries=staging_summaries,
            join_summary=join_report.summary(),
        )
        print(f"  report {report_path}")

    return BuildResult(panel=panel, validation=validation, report_path=report_path)
