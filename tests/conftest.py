"""Synthetic fixtures.

The test suite must run in a clean checkout without the 241 MB Redfin download,
so these fixtures build a miniature panel with the same schema and conventions
as the real one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from california_housing_pulse.data.columns import load_columns
from california_housing_pulse.data.panel import build_panel
from california_housing_pulse.features.target import add_target


@pytest.fixture
def counties() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "county_fips": pd.array(["06001", "06003"], dtype="string"),
            "county_name": pd.array(["Alameda County", "Alpine County"], dtype="string"),
            "county_name_key": pd.array(["alameda", "alpine"], dtype="string"),
        }
    )


@pytest.fixture
def months() -> list[pd.Timestamp]:
    return list(pd.date_range("2024-01-01", "2024-04-01", freq="MS"))


@pytest.fixture
def staged_tables(counties: pd.DataFrame, months: list[pd.Timestamp]) -> dict[str, pd.DataFrame]:
    """Staged tables with one deliberate coverage gap in the housing series."""
    rows = []
    for fips, name in zip(counties["county_fips"], counties["county_name"], strict=True):
        for offset, month in enumerate(months):
            # Alpine County is missing its final month, creating a real gap.
            if fips == "06003" and month == months[-1]:
                continue
            rows.append(
                {
                    "county_fips": fips,
                    "county_name": name,
                    "reference_month": month,
                    "median_sale_price": 500_000 + 1_000 * offset,
                    "median_list_price": 510_000 + 1_000 * offset,
                    "median_ppsf": 400.0,
                    "homes_sold": 100 + offset,
                    "pending_sales": 90,
                    "new_listings": 120,
                    "inventory": 300,
                    "months_of_supply": 3.0,
                    "median_dom": 25.0,
                    "avg_sale_to_list": 1.01,
                    "sold_above_list": 0.4,
                    "price_drops": 0.1,
                    "off_market_in_two_weeks": 0.3,
                    "redfin_last_updated": pd.Timestamp("2026-06-01"),
                }
            )
    redfin = pd.DataFrame(rows)
    redfin["county_fips"] = redfin["county_fips"].astype("string")

    mortgage = pd.DataFrame(
        {
            "reference_month": months,
            "mortgage_rate_30y": [6.5, 6.6, 6.7, 6.8],
            "mortgage_rate_30y_last": [6.55, 6.65, 6.75, 6.85],
            "mortgage_rate_weeks_observed": [4, 5, 4, 4],
            "mortgage_rate_last_release": [m + pd.Timedelta(days=25) for m in months],
        }
    )

    unemployment = pd.DataFrame(
        [
            {
                "county_fips": fips,
                "reference_month": month,
                "employed": 700_000,
                "labor_force": 750_000,
                "unemployed": 50_000,
                "unemployment_rate": 6.7,
            }
            for fips in counties["county_fips"]
            for month in months
        ]
    )
    unemployment["county_fips"] = unemployment["county_fips"].astype("string")

    return {
        "counties": counties,
        "redfin": redfin,
        "mortgage_rate": mortgage,
        "unemployment": unemployment,
    }


@pytest.fixture
def panel(staged_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The panel as the pipeline publishes it: joined, then target-enriched.

    Four months is far too short to produce a label, so every target column here
    is null by construction. That is deliberate — it exercises the path where the
    lead-in swallows the whole panel, which is also what the earliest real months
    look like.
    """
    built, _ = build_panel(staged_tables, write=False)
    return add_target(built)


@pytest.fixture
def column_specs():
    return load_columns()
