"""The frozen target: hand-worked arithmetic, timing, and the freeze itself.

The arithmetic tests deliberately recompute the expected values with plain
``math.log`` over hand-written prices rather than reusing pandas' rolling and
shift machinery. If the implementation had an off-by-one shift, grouped by the
wrong key, or leaked one county's history into another, a test that reused the
same machinery would agree with the bug; these do not.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from california_housing_pulse.features.target import (
    add_target,
    assign_tier,
    classify,
    load_contract,
    modeling_rows,
    summarize,
)

# ---------------------------------------------------------------------------
# The freeze
# ---------------------------------------------------------------------------


def test_contract_matches_the_frozen_values():
    """Milestone 2 froze these. A change here must be a deliberate, argued one.

    This is the guard that makes ``configs/target.yaml`` a freeze rather than a
    convenient knob: editing the config to improve a score also breaks this test.
    """
    contract = load_contract()
    assert contract.price_column == "median_sale_price"
    assert contract.smoothing_window == 3
    assert contract.smoothing_statistic == "mean"
    assert contract.smoothing_min_periods == 3
    assert contract.growth_method == "log"
    assert contract.growth_lag_months == 12
    assert contract.horizon_months == 3
    assert contract.tau == 2.0
    assert contract.min_homes_sold == 10
    assert contract.frozen_status == "frozen"


def test_volume_tiers_are_exhaustive_and_ordered():
    tiers = sorted(load_contract().volume_tiers, key=lambda tier: tier.min)
    assert [tier.name for tier in tiers] == ["excluded", "thin", "small", "mid", "large"]
    assert tiers[0].min == 0
    assert tiers[-1].max is None
    for lower, upper in zip(tiers, tiers[1:], strict=False):
        # Each band must start exactly where the previous one ends, or a county
        # median could fall into a gap and be tiered as NA.
        assert lower.max is not None and upper.min == lower.max + 1


# ---------------------------------------------------------------------------
# Hand-worked example
# ---------------------------------------------------------------------------

# One county, 24 months, with two deliberate step changes in price. Chosen so
# every smoothed value, growth rate and target is computable by hand.
STEP_PRICES = [100_000.0] * 12 + [110_000.0] * 6 + [121_000.0] * 6


def _county_frame(prices: list[float], fips: str = "06001", homes_sold: float = 100.0):
    months = pd.date_range("2020-01-01", periods=len(prices), freq="MS")
    return pd.DataFrame(
        {
            "county_fips": pd.array([fips] * len(prices), dtype="string"),
            "county_name": pd.array(["Test County"] * len(prices), dtype="string"),
            "reference_month": months,
            "median_sale_price": prices,
            "homes_sold": [homes_sold] * len(prices),
        }
    )


@pytest.fixture
def stepped() -> pd.DataFrame:
    return add_target(_county_frame(STEP_PRICES))


def test_smoothing_is_a_trailing_three_month_mean(stepped: pd.DataFrame):
    smoothed = stepped["price_smoothed"]
    # The first two months cannot fill a three-month window.
    assert smoothed.iloc[0:2].isna().all()
    assert smoothed.iloc[2] == pytest.approx(100_000.0)
    # Month 12 is the first to mix the old and new price level.
    assert smoothed.iloc[12] == pytest.approx((100_000 + 100_000 + 110_000) / 3)
    assert smoothed.iloc[13] == pytest.approx((100_000 + 110_000 + 110_000) / 3)
    assert smoothed.iloc[14] == pytest.approx(110_000.0)
    assert smoothed.iloc[18] == pytest.approx((110_000 + 110_000 + 121_000) / 3)
    assert smoothed.iloc[20] == pytest.approx(121_000.0)


def test_growth_is_log_change_against_the_same_month_last_year(stepped: pd.DataFrame):
    growth = stepped["growth_yoy"]
    # Months 12 and 13 need a smoothed value from months 0 and 1, which do not exist.
    assert growth.iloc[12:14].isna().all()
    # Month 14: 110,000 against 100,000 a year earlier.
    assert growth.iloc[14] == pytest.approx(100 * math.log(110_000 / 100_000))
    assert growth.iloc[14] == pytest.approx(9.531018, abs=1e-5)
    # Month 18 mixes levels: 113,333.33 against 100,000.
    assert growth.iloc[18] == pytest.approx(100 * math.log(((110_000 * 2 + 121_000) / 3) / 100_000))
    assert growth.iloc[20] == pytest.approx(100 * math.log(121_000 / 100_000))


def test_target_is_the_three_month_forward_change_in_growth(stepped: pd.DataFrame):
    growth, target = stepped["growth_yoy"], stepped["target_dg"]
    for month in (14, 15, 16, 17, 18, 19, 20):
        assert target.iloc[month] == pytest.approx(growth.iloc[month + 3] - growth.iloc[month])

    # Hand-computed: growth is flat either side of the step, so the change is zero.
    assert target.iloc[14] == pytest.approx(0.0, abs=1e-9)
    # Between the steps the growth rate genuinely accelerates.
    assert target.iloc[17] == pytest.approx(
        100 * math.log(121_000 / 100_000) - 100 * math.log(110_000 / 100_000)
    )
    assert target.iloc[17] == pytest.approx(9.531018, abs=1e-5)


def test_the_final_months_have_no_observable_target(stepped: pd.DataFrame):
    """A label three months ahead cannot exist for the last three months."""
    assert stepped["target_dg"].iloc[-3:].isna().all()
    assert not stepped["has_target"].iloc[-3:].any()


def test_lead_in_months_carry_no_growth(stepped: pd.DataFrame):
    """Two smoothing months plus a twelve-month lag is a fourteen-month lead-in."""
    assert stepped["growth_yoy"].iloc[:14].isna().all()


# ---------------------------------------------------------------------------
# County isolation
# ---------------------------------------------------------------------------


def test_rolling_and_shifts_never_cross_a_county_boundary():
    """The classic panel bug: one county's last month feeding the next county's first."""
    first = _county_frame(STEP_PRICES, fips="06001")
    # A second county whose prices are wildly different, so any bleed is obvious.
    second = _county_frame([500_000.0] * 24, fips="06003")
    combined = add_target(pd.concat([first, second], ignore_index=True))

    alone = add_target(first)
    got = combined.loc[combined["county_fips"] == "06001"].reset_index(drop=True)
    pd.testing.assert_series_equal(
        got["target_dg"], alone["target_dg"], check_names=False, check_index=False
    )

    # The flat county has constant prices, so its growth and target are exactly zero.
    flat = combined.loc[combined["county_fips"] == "06003"].reset_index(drop=True)
    assert flat["growth_yoy"].iloc[14] == pytest.approx(0.0, abs=1e-9)
    assert flat["target_dg"].iloc[14] == pytest.approx(0.0, abs=1e-9)


def test_a_missing_month_blocks_the_window_rather_than_being_skipped():
    """A gap must produce NA, not a mean quietly taken over two months."""
    prices = STEP_PRICES.copy()
    prices[5] = float("nan")
    gapped = add_target(_county_frame(prices))
    # The gap sits inside the windows ending at months 5, 6 and 7.
    assert gapped["price_smoothed"].iloc[5:8].isna().all()
    assert gapped["price_smoothed"].iloc[8] == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_thresholds_are_inclusive_at_exactly_tau():
    contract = load_contract()
    values = pd.Series([-2.5, -2.0, -1.999, 0.0, 1.999, 2.0, 2.5, float("nan")])
    labels = classify(values, contract)
    assert list(labels[:7]) == [
        "cooling",
        "cooling",
        "stable",
        "stable",
        "stable",
        "heating",
        "heating",
    ]


def test_an_unobservable_target_is_not_labelled_stable():
    """Absence of an outcome is not a forecast of no change."""
    labels = classify(pd.Series([float("nan"), 0.0]), load_contract())
    assert pd.isna(labels.iloc[0])
    assert labels.iloc[1] == "stable"


# ---------------------------------------------------------------------------
# Inclusion rule
# ---------------------------------------------------------------------------


def test_a_thin_county_is_flagged_but_its_rows_are_kept():
    thin = _county_frame(STEP_PRICES, fips="06049", homes_sold=4.0)
    busy = _county_frame(STEP_PRICES, fips="06037", homes_sold=6000.0)
    panel = add_target(pd.concat([thin, busy], ignore_index=True))

    excluded = panel.loc[panel["county_fips"] == "06049"]
    assert len(excluded) == 24, "excluded counties stay visible as rows"
    assert not excluded["is_included"].any()
    assert (excluded["volume_tier"] == "excluded").all()
    # The target is still computed for them; only modelling use is withheld.
    assert excluded["has_target"].any()

    assert panel.loc[panel["county_fips"] == "06037", "is_included"].all()
    assert (panel.loc[panel["county_fips"] == "06037", "volume_tier"] == "large").all()


def test_modeling_rows_keeps_only_included_counties_with_a_label():
    thin = _county_frame(STEP_PRICES, fips="06049", homes_sold=4.0)
    busy = _county_frame(STEP_PRICES, fips="06037", homes_sold=6000.0)
    panel = add_target(pd.concat([thin, busy], ignore_index=True))

    model = modeling_rows(panel)
    assert set(model["county_fips"]) == {"06037"}
    assert model["target_dg"].notna().all()
    assert model["is_included"].all()


def test_a_fractional_county_median_still_lands_in_a_tier():
    """Lassen County's median is 24.5; integer tier ranges must not drop it."""
    volume = pd.Series(
        [1.0, 10.0, 24.5, 25.5, 100.0, 101.0, 500.5, 6000.0],
        index=[f"0600{i}" for i in range(8)],
    )
    tiers = assign_tier(volume, load_contract())
    assert list(tiers) == [
        "excluded",
        "thin",
        "thin",
        "thin",
        "small",
        "mid",
        "mid",
        "large",
    ]
    assert tiers.notna().all()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_summary_reports_exclusions_and_prevalence():
    thin = _county_frame(STEP_PRICES, fips="06049", homes_sold=4.0)
    busy = _county_frame(STEP_PRICES, fips="06037", homes_sold=6000.0)
    panel = add_target(pd.concat([thin, busy], ignore_index=True))

    report = summarize(panel)
    assert report.counties == 2
    assert len(report.excluded_counties) == 1
    assert report.modeling_rows == len(modeling_rows(panel))
    assert report.prevalence is not None
    assert set(report.prevalence) == {"cooling", "stable", "heating"}
    assert sum(report.prevalence.values()) == pytest.approx(1.0)
