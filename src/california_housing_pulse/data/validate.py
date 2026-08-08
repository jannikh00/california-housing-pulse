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

from ..features.target import load_contract, modeling_rows
from ..paths import REPORTS_DIR, relative
from .columns import bounded_columns, load_columns
from .normalize import CALIFORNIA_STATE_FIPS, month_range
from .panel import PANEL_KEY

ERROR = "ERROR"
WARN = "WARN"


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


def check_documented_columns(panel: pd.DataFrame, report: ValidationReport) -> None:
    """The panel and the column registry must describe exactly the same columns.

    Enforced in both directions: an undocumented column fails the build, and so
    does a documented column that the panel no longer produces.
    """
    declared = set(load_columns())
    actual = set(panel.columns)
    undocumented = sorted(actual - declared)
    missing = sorted(declared - actual)

    problems = []
    if undocumented:
        problems.append(
            f"{len(undocumented)} column(s) in the panel but not documented: "
            + ", ".join(undocumented)
        )
    if missing:
        problems.append(
            f"{len(missing)} documented column(s) absent from the panel: " + ", ".join(missing)
        )

    _check(
        report,
        "documented_columns_match_panel",
        ERROR,
        not problems,
        (
            f"all {len(actual)} panel columns are documented in configs/columns.yaml"
            if not problems
            else "; ".join(problems)
        ),
        len(undocumented) + len(missing),
    )


def _bound_violations(panel: pd.DataFrame, family: str) -> dict[str, int]:
    """Count values falling outside the named bound family, per column."""
    violations: dict[str, int] = {}
    for name, spec in bounded_columns().items():
        bounds = getattr(spec, family)
        if bounds is None or name not in panel.columns:
            continue
        low, high = bounds
        values = pd.to_numeric(panel[name], errors="coerce")
        outside = values.notna() & ((values < low) | (values > high))
        if outside.any():
            violations[name] = int(outside.sum())
    return violations


def check_hard_bounds(panel: pd.DataFrame, report: ValidationReport) -> None:
    """Physically impossible values indicate a parsing or unit error."""
    violations = _bound_violations(panel, "hard")
    total = sum(violations.values())
    detail = (
        f"{len(bounded_columns())} bounded columns contain no impossible values"
        if not violations
        else "; ".join(f"{col}: {n:,} impossible values" for col, n in violations.items())
    )
    _check(report, "values_within_hard_bounds", ERROR, total == 0, detail, total)


def check_extreme_values(panel: pd.DataFrame, report: ValidationReport) -> None:
    """Extreme but possible values, typically thin low-volume county-months.

    These are measured and reported rather than removed: they are real market
    observations and are exactly the evidence Milestone 2 needs for its county
    inclusion rule.
    """
    violations = _bound_violations(panel, "plausible")
    total = sum(violations.values())
    detail = (
        "no values outside the plausible ranges"
        if not violations
        else "; ".join(f"{col}: {n:,} extreme values" for col, n in sorted(violations.items()))
    )
    _check(report, "extreme_but_possible_values", WARN, total == 0, detail, total)


def check_missingness(panel: pd.DataFrame, report: ValidationReport) -> None:
    """Measure, rather than remove, missing values."""
    measured = [c for c in bounded_columns() if c in panel.columns]
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


def check_target_horizon_within_coverage(panel: pd.DataFrame, report: ValidationReport) -> None:
    """No row may claim a label whose outcome month lies outside the panel.

    ``target_dg`` reads three months forward. If a row at the very end of the
    coverage window carried a label, the forward value would have come from
    somewhere other than an observed future month — a shift misalignment or a
    silently filled gap. This is the check that would catch it.
    """
    if "has_target" not in panel.columns:
        return
    contract = load_contract()
    last_month = panel["reference_month"].max()
    horizon = pd.DateOffset(months=contract.horizon_months)
    too_late = panel["has_target"] & (panel["reference_month"] + horizon > last_month)
    _check(
        report,
        "target_horizon_within_coverage",
        ERROR,
        not too_late.any(),
        (
            f"every labelled row resolves at or before {last_month:%Y-%m}, "
            f"{contract.horizon_months} months ahead"
            if not too_late.any()
            else f"{int(too_late.sum()):,} labelled rows resolve past the coverage window"
        ),
        int(too_late.sum()),
    )


def check_target_leadin_is_unlabelled(panel: pd.DataFrame, report: ValidationReport) -> None:
    """The smoothing window and year-over-year lag must consume real months.

    ``growth_yoy`` cannot exist before ``window - 1 + lag`` months have elapsed.
    A value appearing earlier would mean a rolling or shift operation ran over
    the county boundary, mixing one county's history into another's.
    """
    if "growth_yoy" not in panel.columns:
        return
    contract = load_contract()
    lead_in = contract.smoothing_window - 1 + contract.growth_lag_months
    first_valid = panel["reference_month"].min() + pd.DateOffset(months=lead_in)
    premature = panel["growth_yoy"].notna() & (panel["reference_month"] < first_valid)
    _check(
        report,
        "target_leadin_months_unlabelled",
        ERROR,
        not premature.any(),
        (
            f"no growth value before {first_valid:%Y-%m}, as the {lead_in}-month "
            "lead-in requires"
            if not premature.any()
            else f"{int(premature.sum()):,} rows carry growth inside the lead-in window"
        ),
        int(premature.sum()),
    )


def check_target_class_prevalence(panel: pd.DataFrame, report: ValidationReport) -> None:
    """No directional class may be unusably sparse — the Milestone 2 gate."""
    if "target_label" not in panel.columns:
        return
    contract = load_contract()
    model = modeling_rows(panel)
    if not len(model):
        _check(report, "target_class_prevalence", WARN, False, "no modelling rows produced", 0)
        return

    shares = model["target_label"].value_counts(normalize=True)
    detail = ", ".join(f"{name} {shares.get(name, 0.0):.1%}" for name in contract.label_names)
    smallest = min(float(shares.get(name, 0.0)) for name in contract.label_names)
    _check(
        report,
        "target_class_prevalence",
        WARN,
        smallest >= 0.05,
        (
            f"{len(model):,} modelling rows at tau=+/-{contract.tau:g} — {detail}"
            if smallest >= 0.05
            else f"smallest class is only {smallest:.1%} of rows — {detail}"
        ),
        len(model),
    )


def check_target_inclusion_rule(panel: pd.DataFrame, report: ValidationReport) -> None:
    """Report which counties the volume floor removes, and how much data goes."""
    if "is_included" not in panel.columns:
        return
    contract = load_contract()
    excluded = (
        panel.loc[~panel["is_included"], ["county_fips", "county_name"]]
        .drop_duplicates()
        .sort_values("county_fips")
    )
    names = ", ".join(excluded["county_name"].astype(str))
    dropped_rows = int((~panel["is_included"]).sum())
    _check(
        report,
        "target_inclusion_rule",
        WARN,
        len(excluded) == 0,
        (
            "no county falls below the volume floor"
            if len(excluded) == 0
            else (
                f"{len(excluded)} counties below median homes_sold "
                f"{contract.min_homes_sold:g} excluded ({dropped_rows:,} rows): {names}"
            )
        ),
        dropped_rows,
    )


def validate_panel(
    panel: pd.DataFrame, *, context: dict[str, str] | None = None
) -> ValidationReport:
    """Run every check over the joined panel."""
    report = ValidationReport(context=context or {})
    check_unique_key(panel, report)
    check_county_fips_format(panel, report)
    check_month_alignment(panel, report)
    check_spine_completeness(panel, report)
    check_date_coverage(panel, report)
    check_prediction_cutoff(panel, report)
    check_documented_columns(panel, report)
    check_target_source_present(panel, report)
    check_hard_bounds(panel, report)
    check_extreme_values(panel, report)
    check_missingness(panel, report)
    check_source_coverage(panel, report)
    check_target_horizon_within_coverage(panel, report)
    check_target_leadin_is_unlabelled(panel, report)
    check_target_inclusion_rule(panel, report)
    check_target_class_prevalence(panel, report)
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
    target_summary: str | None = None,
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
        "- grain: one row per `(county_fips, reference_month)`",
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

    if target_summary:
        lines += [
            "## Target construction",
            "",
            f"Frozen contract: {load_contract().describe()}.",
            "See `configs/target.yaml` for the full definition and rationale.",
            "",
            "```text",
            target_summary,
            "```",
            "",
        ]

    lines += [
        "## Checks",
        "",
        "| Check | Severity | Status | Detail |",
        "|---|---|---|---|",
    ]
    for result in report.results:
        lines.append(f"| `{result.name}` | {result.severity} | {result.status} | {result.detail} |")

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
