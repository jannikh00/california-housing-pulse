"""Data-quality checks must actually fail when the data is wrong."""

from __future__ import annotations

import pandas as pd

from california_housing_pulse.data.validate import (
    ERROR,
    check_documented_columns,
    check_extreme_values,
    check_hard_bounds,
    check_unique_key,
    coverage_by_county,
    validate_panel,
)


def _result(report, name):
    return next(r for r in report.results if r.name == name)


def test_clean_panel_passes_every_error_check(panel: pd.DataFrame):
    report = validate_panel(panel)
    assert report.ok, [r.detail for r in report.errors]


def test_duplicate_key_is_an_error(panel: pd.DataFrame):
    from california_housing_pulse.data.validate import ValidationReport

    duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    report = ValidationReport()
    check_unique_key(duplicated, report)
    result = _result(report, "unique_county_month_key")
    assert not result.passed
    assert result.severity == ERROR
    assert result.count == 1


def test_impossible_value_fails_the_build(panel: pd.DataFrame):
    """A unit error — prices recorded in cents — must not pass silently."""
    from california_housing_pulse.data.validate import ValidationReport

    broken = panel.copy()
    broken.loc[0, "median_sale_price"] = 500_000_00_000.0
    report = ValidationReport()
    check_hard_bounds(broken, report)
    result = _result(report, "values_within_hard_bounds")
    assert not result.passed
    assert result.severity == ERROR


def test_extreme_but_possible_value_warns_without_failing(panel: pd.DataFrame):
    """466 months of supply in a 2-sale county is real data, not corruption."""
    from california_housing_pulse.data.validate import ValidationReport

    thin = panel.copy()
    thin.loc[0, "months_of_supply"] = 466.5
    report = ValidationReport()
    check_hard_bounds(thin, report)
    check_extreme_values(thin, report)

    assert _result(report, "values_within_hard_bounds").passed
    extreme = _result(report, "extreme_but_possible_values")
    assert not extreme.passed
    assert extreme.severity == "WARN"
    assert report.ok


def test_undocumented_column_fails_the_build(panel: pd.DataFrame):
    from california_housing_pulse.data.validate import ValidationReport

    extended = panel.assign(mystery_column=1)
    report = ValidationReport()
    check_documented_columns(extended, report)
    result = _result(report, "documented_columns_match_panel")
    assert not result.passed
    assert "mystery_column" in result.detail


def test_documented_column_missing_from_panel_fails_the_build(panel: pd.DataFrame):
    from california_housing_pulse.data.validate import ValidationReport

    reduced = panel.drop(columns=["unemployment_rate"])
    report = ValidationReport()
    check_documented_columns(reduced, report)
    result = _result(report, "documented_columns_match_panel")
    assert not result.passed
    assert "unemployment_rate" in result.detail


def test_coverage_by_county_measures_the_gap(panel: pd.DataFrame):
    coverage = coverage_by_county(panel)
    alpine = coverage[coverage["county_fips"] == "06003"].iloc[0]
    assert alpine["months"] == 4
    assert alpine["months_with_price"] == 3
    assert alpine["coverage"] == 0.75
