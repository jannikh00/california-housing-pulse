"""Shared normalization rules for identifiers, dates, and units.

Two conventions are enforced everywhere downstream:

``county_fips``
    A five-character zero-padded string. California is state FIPS ``06``, so
    every county code in this project begins with ``06`` and *must* be carried as
    text; storing it as an integer silently produces ``6037`` and breaks joins.

``reference_month``
    A ``datetime64`` timestamp pinned to the first day of the month. Sources
    variously report a month as its first day, its last day, or a year/period
    pair, and comparing those directly produces silent join loss.
"""

from __future__ import annotations

import re

import pandas as pd

CALIFORNIA_STATE_FIPS = "06"

# "Los Angeles County, CA" -> "los angeles"; also handles the parish/borough/
# census-area variants used by the Census county list.
_COUNTY_SUFFIXES = (
    "county",
    "parish",
    "borough",
    "census area",
    "city and borough",
    "municipality",
)
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def to_county_fips(state_fips: pd.Series, county_fips: pd.Series) -> pd.Series:
    """Combine 2-digit state and 3-digit county codes into a 5-character FIPS."""
    state = state_fips.astype("string").str.strip().str.zfill(2)
    county = county_fips.astype("string").str.strip().str.zfill(3)
    return (state + county).astype("string")


def normalize_county_name(names: pd.Series) -> pd.Series:
    """Reduce a county name to a comparable join key.

    ``"Los Angeles County, CA"``, ``"Los Angeles County"`` and ``"LOS ANGELES"``
    all become ``"los angeles"``. Only the trailing state qualifier is stripped,
    so ``"District of Columbia"`` and similar names survive intact.
    """
    cleaned = names.astype("string").str.strip().str.lower()
    # Drop a trailing ", ca" / ", california" state qualifier.
    cleaned = cleaned.str.replace(r",\s*[a-z .]+$", "", regex=True)
    # Drop a trailing geography-type word.
    suffix_pattern = r"\s+(" + "|".join(_COUNTY_SUFFIXES) + r")$"
    cleaned = cleaned.str.replace(suffix_pattern, "", regex=True)
    cleaned = cleaned.str.replace(_NON_ALPHANUMERIC.pattern, " ", regex=True)
    return cleaned.str.strip().str.replace(r"\s+", " ", regex=True).astype("string")


def to_reference_month(values: pd.Series) -> pd.Series:
    """Coerce any date-like column to a month-start timestamp."""
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp()


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Every month-start timestamp from ``start`` to ``end`` inclusive."""
    return pd.date_range(
        pd.Timestamp(start).to_period("M").to_timestamp(),
        pd.Timestamp(end).to_period("M").to_timestamp(),
        freq="MS",
    )


def snake_case(columns: list[str]) -> list[str]:
    """Normalize source column names to lower snake_case."""
    return [_NON_ALPHANUMERIC.sub("_", column.strip().lower()).strip("_") for column in columns]
