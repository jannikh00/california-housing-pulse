"""Identifier, name and date normalization rules."""

from __future__ import annotations

import pandas as pd

from california_housing_pulse.data.normalize import (
    month_range,
    normalize_county_name,
    snake_case,
    to_county_fips,
    to_reference_month,
)


def test_county_fips_keeps_leading_zero():
    """The whole join depends on 06037 never becoming 6037."""
    result = to_county_fips(pd.Series(["6", "06"]), pd.Series(["37", "037"]))
    assert list(result) == ["06037", "06037"]


def test_county_name_normalization_matches_redfin_to_census():
    """Redfin's 'Los Angeles County, CA' must reduce to the Census name key."""
    redfin = normalize_county_name(pd.Series(["Los Angeles County, CA"]))
    census = normalize_county_name(pd.Series(["Los Angeles County"]))
    assert redfin.iloc[0] == census.iloc[0] == "los angeles"


def test_county_name_normalization_handles_multiword_and_punctuation():
    names = pd.Series(["San Luis Obispo County, CA", "DEL NORTE COUNTY", "  Yuba County  "])
    assert list(normalize_county_name(names)) == ["san luis obispo", "del norte", "yuba"]


def test_reference_month_pins_any_date_to_month_start():
    values = pd.Series(["2024-03-01", "2024-03-17", "2024-03-31"])
    result = to_reference_month(values)
    assert (result == pd.Timestamp("2024-03-01")).all()


def test_reference_month_is_nat_for_unparseable_input():
    assert pd.isna(to_reference_month(pd.Series(["not a date"])).iloc[0])


def test_month_range_is_inclusive_of_both_ends():
    months = month_range(pd.Timestamp("2024-01-15"), pd.Timestamp("2024-04-02"))
    assert list(months) == list(pd.date_range("2024-01-01", "2024-04-01", freq="MS"))


def test_snake_case_normalizes_source_headers():
    assert snake_case(["PERIOD_BEGIN", "  Median Sale Price ", "AVG_SALE_TO_LIST"]) == [
        "period_begin",
        "median_sale_price",
        "avg_sale_to_list",
    ]
