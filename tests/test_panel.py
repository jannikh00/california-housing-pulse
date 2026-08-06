"""Panel construction: the spine, the joins, and the prediction cutoff."""

from __future__ import annotations

import pandas as pd

from california_housing_pulse.data.panel import PANEL_KEY, build_panel, prediction_as_of


def test_panel_has_one_row_per_county_month(panel: pd.DataFrame):
    assert not panel.duplicated(subset=PANEL_KEY).any()


def test_spine_is_complete_regardless_of_source_coverage(panel: pd.DataFrame, months):
    """2 counties x 4 months = 8 rows even though Redfin supplies only 7."""
    assert len(panel) == 2 * len(months)
    assert panel["county_fips"].nunique() == 2
    assert panel["reference_month"].nunique() == len(months)


def test_coverage_gap_becomes_a_visible_row_not_a_missing_one(panel: pd.DataFrame, months):
    """The whole point of the complete spine: gaps are countable."""
    gap = panel[(panel["county_fips"] == "06003") & (panel["reference_month"] == months[-1])]
    assert len(gap) == 1
    assert bool(gap["has_redfin"].iloc[0]) is False
    assert pd.isna(gap["median_sale_price"].iloc[0])
    assert int((~panel["has_redfin"]).sum()) == 1


def test_national_mortgage_rate_repeats_across_counties(panel: pd.DataFrame, months):
    month = months[0]
    rates = panel.loc[panel["reference_month"] == month, "mortgage_rate_30y"]
    assert rates.nunique() == 1
    assert rates.notna().all()


def test_join_report_counts_matched_and_unmatched_rows(staged_tables):
    _, report = build_panel(staged_tables, write=False)
    assert report.spine_rows == 8
    assert report.joins["redfin"]["spine_rows_matched"] == 7
    assert report.joins["redfin"]["spine_rows_unmatched"] == 1
    assert report.joins["mortgage_rate"]["spine_rows_matched"] == 8


def test_prediction_cutoff_is_the_fifteenth_of_the_following_month():
    months = pd.Series(pd.to_datetime(["2024-01-01", "2024-11-01", "2024-12-01"]))
    result = prediction_as_of(months)
    assert list(result) == list(pd.to_datetime(["2024-02-15", "2024-12-15", "2025-01-15"]))


def test_prediction_cutoff_always_follows_its_reference_month(panel: pd.DataFrame):
    assert (panel["prediction_as_of"] > panel["reference_month"]).all()


def test_county_fips_stays_a_string_through_the_join(panel: pd.DataFrame):
    assert pd.api.types.is_string_dtype(panel["county_fips"])
    assert panel["county_fips"].str.len().eq(5).all()
