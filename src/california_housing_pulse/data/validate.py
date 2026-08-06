"""Automated data-quality checks over the joined panel.

Every check returns a :class:`CheckResult` rather than raising, so one run
reports every problem instead of stopping at the first. Results are written to
``reports/data_quality.md``. Checks marked ``ERROR`` fail the pipeline; checks
marked ``WARN`` are measurements that must be visible but do not by themselves
invalidate the panel — coverage gaps, for instance, are expected and are the
thing the complete spine exists to expose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from ..paths import REPORTS_DIR, relative
from .normalize import CALIFORNIA_STATE_FIPS, month_range
from .panel import PANEL_KEY

ERROR = "ERROR"
WARN = "WARN"

# Plausibility bounds. These are deliberately wide: the goal is to catch parsing
# and unit errors, not to second-guess the housing market.
VALUE_BOUNDS: dict[str, tuple[float, float]] = {
    "median_sale_price": (1_000, 50_000_000),
    "median_list_price": (1_000, 50_000_000),
    "median_ppsf": (1, 20_000),
    "homes_sold": (0, 100_000),
    "pending_sales": (0, 100_000),
    "new_listings": (0, 100_000),
    "inventory": (0, 500_000),
    "months_of_supply": (0, 120),
    "median_dom": (0, 1_000),
    "avg_sale_to_list": (0.2, 3.0),
    "sold_above_list": (0.0, 1.0),
    "price_drops": (0.0, 1.0),
    "off_market_in_two_weeks": (0.0, 1.0),
    "unemployment_rate": (0.0, 100.0),
    "unemployed": (0, 5_000_000),
    "employed": (0, 10_000_000),
    "labor_force": (0, 15_000_000),
    "mortgage_rate_30y": (0.5, 25.0),
}


@dataclass
class CheckResult:
    """Outcome of one data-quality check."""

    name: str
    severity: str
    passed: bool
    detail: str
    count: int = 0

    @property
    def status(self) -> str:
        if self.passed:
            return "pass"
        return "FAIL" if self.severity == ERROR else "warn"


@dataclass
class ValidationReport:
    results: list[CheckResult] = field(default_factory=list)
    context: dict[str, str] = field(default_factory=dict)

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == ERROR]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = []
        for result in self.results:
            lines.append(f"  [{result.status:>4}] {result.name}: {result.detail}")
        lines.append(
            f"  => {len(self.results)} checks, {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings"
        )
        return "\n".join(lines)


def _check(
    report: ValidationReport,
    name: str,
    severity: str,
    passed: bool,
    detail: str,
    count: int = 0,
) -> None:
    report.add(CheckResult(name=name, severity=severity, passed=passed, detail=detail, count=count))


def check_unique_key(panel: pd.DataFrame, report: ValidationReport) -> None:
    duplicates = panel.duplicated(subset=PANEL_KEY).sum()
    _check(
        report,
        "unique_county_month_key",
        ERROR,
        duplicates == 0,
        (
            f"{len(panel):,} rows, one per (county_fips, reference_month)"
            if duplicates == 0
            else f"{duplicates:,} duplicate county-month keys"
        ),
        int(duplicates),
    )


def check_county_fips_format(panel: pd.DataFrame, report: ValidationReport) -> None:
    fips = panel["county_fips"].astype("string")
    bad = ~fips.str.fullmatch(rf"{CALIFORNIA_STATE_FIPS}\d{{3}}")
    _check(
        report,
        "county_fips_format",
        ERROR,
        not bad.any(),
        (
            f"all {fips.nunique()} county codes are 5-character California FIPS strings"
            if not bad.any()
            else f"{int(bad.sum()):,} rows with a malformed county FIPS"
        ),
        int(bad.sum()),
    )


def check_month_alignment(panel: pd.DataFrame, report: ValidationReport) -> None:
    months = panel["reference_month"]
    misaligned = months.dt.day != 1
    _check(
        report,
        "reference_month_is_month_start",
        ERROR,
        not misaligned.any(),
        (
            "every reference_month is pinned to the first of the month"
            if not misaligned.any()
            else f"{int(misaligned.sum()):,} rows not aligned to a month start"
        ),
        int(misaligned.sum()),
    )


def check_spine_completeness(panel: pd.DataFrame, report: ValidationReport) -> None:
    counties = panel["county_fips"].nunique()
    months = month_range(panel["reference_month"].min(), panel["reference_month"].max())
    expected = counties * len(months)
    _check(
        report,
        "complete_county_month_spine",
        ERROR,
        len(panel) == expected,
        (
            f"{counties} counties x {len(months)} months = {expected:,} rows, as built"
            if len(panel) == expected
            else f"expected {expected:,} rows for a complete spine, found {len(panel):,}"
        ),
        abs(expected - len(panel)),
    )


def check_date_coverage(panel: pd.DataFrame, report: ValidationReport) -> None:
    observed = pd.DatetimeIndex(sorted(panel["reference_month"].unique()))
    expected = month_range(observed.min(), observed.max())
    missing = expected.difference(observed)
    _check(
        report,
        "contiguous_month_coverage",
        ERROR,
        len(missing) == 0,
        (
            f"{len(observed)} consecutive months, {observed.min():%Y-%m} to {observed.max():%Y-%m}"
            if len(missing) == 0
            else f"{len(missing)} month(s) missing from the coverage window"
        ),
        len(missing),
    )


def check_prediction_cutoff(panel: pd.DataFrame, report: ValidationReport) -> None:
    invalid = panel["prediction_as_of"] <= panel["reference_month"]
    _check(
        report,
        "prediction_cutoff_after_reference_month",
        ERROR,
        not invalid.any(),
        (
            "every prediction_as_of falls after its reference month"
            if not invalid.any()
            else f"{int(invalid.sum()):,} rows with a cutoff at or before the reference month"
        ),
        int(invalid.sum()),
    )


def check_value_bounds(panel: pd.DataFrame, report: ValidationReport) -> None:
    violations: dict[str, int] = {}
    for column, (low, high) in VALUE_BOUNDS.items():
        if column not in panel.columns:
            continue
        values = pd.to_numeric(panel[column], errors="coerce")
        outside = values.notna() & ((values < low) | (values > high))
        if outside.any():
            violations[column] = int(outside.sum())

    total = sum(violations.values())
    detail = (
        f"{len(VALUE_BOUNDS)} bounded columns within plausible ranges"
        if not violations
        else "; ".join(f"{col}: {n:,} rows outside bounds" for col, n in violations.items())
    )
    _check(report, "values_within_plausible_bounds", ERROR, total == 0, detail, total)


def check_missingness(panel: pd.DataFrame, report: ValidationReport) -> None:
    """Measure, rather than remove, missing values."""
    measured = [c for c in VALUE_BOUNDS if c in panel.columns]
    missing = panel[measured].isna().mean().sort_values(ascending=False)
    worst = missing.head(3)
    detail = ", ".join(f"{col} {share:.1%}" for col, share in worst.items())
    _check(
        report,
        "column_missingness_measured",
        WARN,
        bool((missing == 0).all()),
        f"highest missingness — {detail}",
        int(panel[measured].isna().sum().sum()),
    )


def check_source_coverage(panel: pd.DataFrame, report: ValidationReport) -> None:
    for flag, label in (("has_redfin", "Redfin housing"), ("has_unemployment", "BLS unemployment")):
        if flag not in panel.columns:
            continue
        gaps = int((~panel[flag]).sum())
        share = gaps / len(panel) if len(panel) else 0
        _check(
            report,
            f"coverage_{flag}",
            WARN,
            gaps == 0,
            (
                f"{label}: complete across all {len(panel):,} county-months"
                if gaps == 0
                else f"{label}: {gaps:,} county-months without data ({share:.1%})"
            ),
            gaps,
        )


def check_target_source_present(panel: pd.DataFrame, report: ValidationReport) -> None:
    """The panel is useless without the price series that defines the target."""
    present = int(panel["median_sale_price"].notna().sum())
    _check(
        report,
        "target_price_series_present",
        ERROR,
        present > 0,
        f"{present:,} county-months carry a median sale price",
        present,
    )


def validate_panel(panel: pd.DataFrame, *, context: dict[str, str] | None = None) -> ValidationReport:
    """Run every check over the joined panel."""
    report = ValidationReport(context=context or {})
    check_unique_key(panel, report)
    check_county_fips_format(panel, report)
    check_month_alignment(panel, report)
    check_spine_completeness(panel, report)
    check_date_coverage(panel, report)
    check_prediction_cutoff(panel, report)
    check_target_source_present(panel, report)
    check_value_bounds(panel, report)
    check_missingness(panel, report)
    check_source_coverage(panel, report)
    return report


def coverage_by_county(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-county Redfin coverage, used for the Milestone 2 inclusion rule."""
    grouped = panel.groupby(["county_fips", "county_name"], as_index=False).agg(
        months=("reference_month", "size"),
        months_with_price=("median_sale_price", "count"),
    )
    grouped["coverage"] = grouped["months_with_price"] / grouped["months"]
    return grouped.sort_values("coverage", ignore_index=True)


def write_report(
    panel: pd.DataFrame,
    report: ValidationReport,
    *,
    staging_summaries: list[str] | None = None,
    join_summary: str | None = None,
) -> str:
    """Render the data-quality report to ``reports/data_quality.md``."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "data_quality.md"

    coverage = coverage_by_county(panel)
    lowest = coverage.head(10)

    lines = [
        "# Data quality report",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} by `chp build`.",
        "This file is regenerated by the pipeline; do not edit it by hand.",
        "",
        "## Panel",
        "",
        f"- rows: **{len(panel):,}**",
        f"- grain: one row per `(county_fips, reference_month)`",
        f"- counties: {panel['county_fips'].nunique()}",
        (
            f"- coverage: {panel['reference_month'].min():%Y-%m} to "
            f"{panel['reference_month'].max():%Y-%m}"
        ),
        f"- columns: {panel.shape[1]}",
        "",
    ]

    if staging_summaries:
        lines += ["## Staging row accounting", "", "```text"]
        lines += staging_summaries
        lines += ["```", ""]

    if join_summary:
        lines += ["## Join accounting", "", "```text", join_summary, "```", ""]

    lines += [
        "## Checks",
        "",
        "| Check | Severity | Status | Detail |",
        "|---|---|---|---|",
    ]
    for result in report.results:
        lines.append(
            f"| `{result.name}` | {result.severity} | {result.status} | {result.detail} |"
        )

    lines += [
        "",
        f"**{len(report.results)} checks — {len(report.errors)} errors, "
        f"{len(report.warnings)} warnings.**",
        "",
        "## Lowest-coverage counties",
        "",
        "Milestone 2 sets the inclusion rule; these counts are the evidence for it.",
        "",
        "| County | FIPS | Months | Months with price | Coverage |",
        "|---|---|---|---|---|",
    ]
    for row in lowest.itertuples():
        lines.append(
            f"| {row.county_name} | {row.county_fips} | {row.months} | "
            f"{row.months_with_price} | {row.coverage:.1%} |"
        )
    lines.append("")

    path.write_text("\n".join(lines))
    return relative(path)
