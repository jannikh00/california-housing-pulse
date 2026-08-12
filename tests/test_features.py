"""Tests for the Milestone 3 feature layer.

The suite is organised around the one claim that matters: **no feature uses
information that was unavailable when the forecast was made.** Three independent
checks defend it, because they fail on different bugs.

*The publication audit* compares each feature's newest input against the row's
forecast cutoff using the publisher's real release delay. It catches a lag that
is too short for a source — the failure a purely structural check cannot see,
because such a feature is backward-looking in panel terms and still impossible in
real terms.

*The truncation replay* rebuilds every feature on a panel cut off at month *t*
and demands identical values. It catches a mis-signed shift or a window that
straddles the cutoff — the failure the audit cannot see, because the audit reads
the specification rather than the arithmetic.

*The forward-shift refusal* makes the whole class unreachable at the lowest level.

A test that only restates the configuration would prove nothing, so
:func:`test_audit_rejects_a_release_lag_that_is_too_short` deliberately breaks the
contract and asserts the audit notices.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from california_housing_pulse.data.panel import build_panel
from california_housing_pulse.features.build import (
    all_specs,
    apply_missing_policy,
    availability_table,
    build_features,
    feature_names,
)
from california_housing_pulse.features.spec import load_feature_contract
from california_housing_pulse.features.target import add_target
from california_housing_pulse.features.transforms import (
    assert_complete_spine,
    log_diff,
    reltrend,
    rollmean,
)

# The deepest feature reads 14 months, and a rolling window needs room to move
# after that, so a fixture shorter than roughly three years cannot exercise the
# transforms at all — every column would be NA and the replay test would pass
# vacuously.
FIXTURE_START = "2018-01-01"
FIXTURE_END = "2021-12-01"

# One month of unemployment is dropped from the fixture, mirroring the real
# 2025-10 BLS gap the missing-data policy exists for.
UNEMPLOYMENT_GAP = pd.Timestamp("2020-06-01")


@pytest.fixture
def long_months() -> list[pd.Timestamp]:
    return list(pd.date_range(FIXTURE_START, FIXTURE_END, freq="MS"))


@pytest.fixture
def long_panel(long_months: list[pd.Timestamp]) -> pd.DataFrame:
    """A four-year, three-county panel with the real schema.

    Values move deterministically but non-linearly: a constant series would make
    ``log_diff`` and ``reltrend`` identically zero, and a test that compares zero
    to zero cannot distinguish a correct shift from an off-by-one.
    """
    counties = pd.DataFrame(
        {
            "county_fips": pd.array(["06001", "06037", "06063"], dtype="string"),
            "county_name": pd.array(
                ["Alameda County", "Los Angeles County", "Plumas County"], dtype="string"
            ),
            "county_name_key": pd.array(["alameda", "los angeles", "plumas"], dtype="string"),
        }
    )
    # Deliberately different scales, so scale-free transforms can be tested for
    # actually being scale-free across counties rather than within one.
    scale = {"06001": 1.0, "06037": 25.0, "06063": 0.05}

    rows = []
    for fips, name in zip(counties["county_fips"], counties["county_name"], strict=True):
        for i, month in enumerate(long_months):
            wave = np.sin(i / 5.0)
            rows.append(
                {
                    "county_fips": fips,
                    "county_name": name,
                    "reference_month": month,
                    "median_sale_price": 500_000 * (1 + 0.01 * i + 0.05 * wave),
                    "median_list_price": 510_000 * (1 + 0.01 * i + 0.04 * wave),
                    "median_ppsf": 400.0 * (1 + 0.008 * i),
                    "homes_sold": 200 * scale[fips] * (1 + 0.3 * wave),
                    "pending_sales": 190 * scale[fips] * (1 + 0.3 * wave),
                    "new_listings": 240 * scale[fips] * (1 + 0.2 * wave),
                    "inventory": 600 * scale[fips] * (1 - 0.2 * wave),
                    "months_of_supply": 3.0 + wave,
                    "median_dom": 25.0 + 8 * wave,
                    "avg_sale_to_list": 1.0 + 0.03 * wave,
                    "sold_above_list": 0.4 + 0.1 * wave,
                    "price_drops": 0.1,
                    "off_market_in_two_weeks": 0.3 + 0.05 * wave,
                    "redfin_last_updated": pd.Timestamp("2026-06-01"),
                }
            )
    redfin = pd.DataFrame(rows)
    redfin["county_fips"] = redfin["county_fips"].astype("string")

    mortgage = pd.DataFrame(
        {
            "reference_month": long_months,
            "mortgage_rate_30y": [4.0 + 0.02 * i for i in range(len(long_months))],
            "mortgage_rate_30y_last": [4.01 + 0.02 * i for i in range(len(long_months))],
            "mortgage_rate_weeks_observed": [4] * len(long_months),
            "mortgage_rate_last_release": [m + pd.Timedelta(days=25) for m in long_months],
        }
    )

    unemployment = pd.DataFrame(
        [
            {
                "county_fips": fips,
                "reference_month": month,
                "employed": 700_000.0,
                "labor_force": 750_000.0,
                "unemployed": 50_000.0,
                "unemployment_rate": 5.0 + 0.5 * np.sin(i / 4.0),
            }
            for fips in counties["county_fips"]
            for i, month in enumerate(long_months)
            if month != UNEMPLOYMENT_GAP
        ]
    )
    unemployment["county_fips"] = unemployment["county_fips"].astype("string")

    built, _ = build_panel(
        {
            "counties": counties,
            "redfin": redfin,
            "mortgage_rate": mortgage,
            "unemployment": unemployment,
        },
        write=False,
    )
    return add_target(built)


@pytest.fixture
def contract():
    return load_feature_contract()


# ---------------------------------------------------------------------------
# The specification parses, and refuses to parse anything ambiguous
# ---------------------------------------------------------------------------


def test_every_feature_names_a_known_source_and_transform(contract):
    for spec in contract.features:
        assert spec.source in contract.sources
        assert spec.release_lag_months == contract.sources[spec.source].release_lag_months
        assert spec.effective_lag >= spec.release_lag_months


def test_feature_names_are_unique(contract):
    names = feature_names(contract)
    assert len(names) == len(set(names))


def test_no_feature_is_built_on_a_target_column(contract):
    """The label may never appear on the feature side, in any form."""
    forbidden = {"target_dg", "target_label", "has_target"}
    assert not forbidden & {spec.column for spec in contract.features}


def test_no_feature_is_built_on_a_whole_panel_statistic(contract):
    """``homes_sold_median`` and ``volume_tier`` summarise the test window too.

    Both are legitimate as reporting dimensions and both would leak as inputs, so
    the boundary is asserted rather than left to reviewer discipline.
    """
    forbidden = {"homes_sold_median", "volume_tier", "is_included"}
    assert not forbidden & {spec.column for spec in contract.features}


def test_excluded_columns_cannot_be_used(contract):
    assert "price_drops" in contract.excluded_columns
    assert "price_drops" not in {spec.column for spec in contract.features}


def test_reach_accounts_for_both_lag_and_window(contract):
    """A 12-month window on a two-month-lagged source reads 14 months, not 12."""
    unemployment = {s.name: s for s in contract.features if s.source == "bls_lau_california"}
    assert unemployment["unemployment_rate__diff12"].reach_months == 14
    momentum = {s.name: s for s in contract.features if s.column == "growth_yoy"}
    assert momentum["growth_yoy__rollstd12"].reach_months == 11
    assert momentum["growth_yoy__lag12"].reach_months == 12


# ---------------------------------------------------------------------------
# Leakage check 1 — the publication audit
# ---------------------------------------------------------------------------


def test_cutoff_rule_reproduces_the_panels_prediction_as_of(long_panel, contract):
    """The audit's notion of the cutoff must be the panel's, not a parallel one."""
    rebuilt = contract.cutoff.for_month(long_panel["reference_month"])
    pd.testing.assert_series_equal(
        rebuilt, long_panel["prediction_as_of"], check_names=False, check_dtype=False
    )


def test_no_feature_reads_past_the_forecast_cutoff(long_panel, contract):
    audit = contract.audit_publication(long_panel["reference_month"])
    leaking = audit.loc[audit["leaks"]]
    assert leaking.empty, f"features reading past the cutoff:\n{leaking}"
    assert (audit["min_margin_days"] >= 0).all()


def test_audit_rejects_a_release_lag_that_is_too_short(long_panel, contract):
    """The guard against the audit being a tautology.

    County unemployment is published roughly 50 days after the month ends, while
    the cutoff falls on the 15th of the following month. A one-month lag is
    therefore about a week short, and the audit has to say so — otherwise it is
    only reading back the number it was given.
    """
    shortened = dataclasses.replace(
        contract,
        sources={
            source_id: (
                dataclasses.replace(timing, release_lag_months=1)
                if source_id == "bls_lau_california"
                else timing
            )
            for source_id, timing in contract.sources.items()
        },
        features=tuple(
            dataclasses.replace(spec, release_lag_months=1)
            if spec.source == "bls_lau_california"
            else spec
            for spec in contract.features
        ),
    )
    audit = shortened.audit_publication(long_panel["reference_month"])
    leaking = audit.loc[audit["leaks"]]
    assert not leaking.empty
    assert set(leaking["source"]) == {"bls_lau_california"}
    assert leaking["min_margin_days"].min() < 0


# ---------------------------------------------------------------------------
# Leakage check 2 — the truncation replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", ["2020-03-01", "2021-01-01", "2021-07-01"])
def test_truncation_replay_reproduces_every_feature(long_panel, cut):
    """Rebuilding on history alone must reproduce the value exactly.

    This is the strongest statement the feature layer can make: if any transform
    reached forward, hiding the future would change the answer. The comparison is
    exact rather than approximate — a shift either happened or it did not.
    """
    cut = pd.Timestamp(cut)
    full, _ = build_features(long_panel)
    truncated, _ = build_features(long_panel.loc[long_panel["reference_month"] <= cut].copy())

    names = feature_names()
    left = full.loc[full["reference_month"] == cut, names].reset_index(drop=True)
    right = truncated.loc[truncated["reference_month"] == cut, names].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)
    # Guard against the replay passing because everything is NA at the cut month.
    assert right.notna().to_numpy().sum() > 0


def test_transforms_refuse_a_forward_shift(long_panel):
    """The failure mode is impossible to express, not merely absent."""
    values = long_panel["median_sale_price"].astype("float64")
    with pytest.raises(ValueError, match="backward-looking"):
        rollmean(long_panel, values, 3, lag_months=-1)


def test_no_feature_is_available_before_the_history_it_needs(long_panel, contract):
    """No feature may invent history the panel does not have.

    ``reach_months`` is a lower bound on when a feature can appear, not an
    equality: a feature may also be delayed by the lead-in of the column beneath
    it. ``growth_yoy__lag0`` reaches back zero months of its own yet cannot exist
    before 2019-03 here, because ``growth_yoy`` needs fourteen months of price
    history first. What must never happen is a feature appearing *earlier* than
    its own reach allows.
    """
    features, _ = build_features(long_panel)
    start = features["reference_month"].min()

    for spec in all_specs(contract):
        present = features.loc[features[spec.name].notna(), "reference_month"]
        if present.empty:
            continue
        earliest = start + pd.DateOffset(months=spec.reach_months)
        assert present.min() >= earliest, (
            f"{spec.name} reads {spec.reach_months} months of history but first "
            f"appears at {present.min():%Y-%m}, before {earliest:%Y-%m}"
        )

    # Kept concrete so the loop above cannot pass vacuously: a level drawn
    # straight from a raw source with no lag really is available at once, and a
    # twelve-month window on the same source really is not.
    immediate = features.loc[features["mortgage_rate_30y__lag0"].notna(), "reference_month"]
    assert immediate.min() == start
    delayed = features.loc[features["mortgage_rate_30y__diff12"].notna(), "reference_month"]
    assert delayed.min() == start + pd.DateOffset(months=12)


# ---------------------------------------------------------------------------
# The transforms compute what they claim
# ---------------------------------------------------------------------------


def test_lag_and_diff_match_hand_computed_values(long_panel):
    features, _ = build_features(long_panel)
    panel = long_panel.sort_values(["county_fips", "reference_month"], ignore_index=True)

    county = panel.loc[panel["county_fips"] == "06001"].reset_index(drop=True)
    row = 30
    month = county.loc[row, "reference_month"]
    got = features.loc[
        (features["county_fips"] == "06001") & (features["reference_month"] == month)
    ].iloc[0]

    # mortgage_rate_30y has a zero release lag, so diff3 at t is x(t) - x(t-3).
    assert got["mortgage_rate_30y__diff3"] == pytest.approx(
        county.loc[row, "mortgage_rate_30y"] - county.loc[row - 3, "mortgage_rate_30y"]
    )
    # unemployment_rate carries a two-month lag, so lag0 at t is x(t-2).
    assert got["unemployment_rate__lag0"] == pytest.approx(county.loc[row - 2, "unemployment_rate"])
    # ...and diff3 spans t-5 to t-2, not t-3 to t.
    assert got["unemployment_rate__diff3"] == pytest.approx(
        county.loc[row - 2, "unemployment_rate"] - county.loc[row - 5, "unemployment_rate"]
    )


def test_log_diff_is_symmetric(long_panel):
    """A rise and the fall that exactly undoes it are equal and opposite."""
    panel = long_panel.sort_values(["county_fips", "reference_month"], ignore_index=True).copy()
    up = pd.Series([100.0, 150.0] * (len(panel) // 2 + 1))[: len(panel)]
    down = pd.Series([150.0, 100.0] * (len(panel) // 2 + 1))[: len(panel)]

    rise = log_diff(panel, up, 1, lag_months=0).dropna()
    fall = log_diff(panel, down, 1, lag_months=0).dropna()
    magnitudes = pd.concat([rise, fall]).abs()
    assert magnitudes.max() == pytest.approx(magnitudes.min())
    assert magnitudes.min() == pytest.approx(100.0 * np.log(1.5))
    # Simple percentage change would give +50 and -33.3 here; log growth is the
    # reason a symmetric threshold can classify both moves consistently.
    assert rise.max() == pytest.approx(-fall.min())


def test_reltrend_is_invariant_to_county_scale(long_panel):
    """Los Angeles and Plumas differ 500-fold in volume; the feature must not."""
    features, _ = build_features(long_panel)
    wide = features.pivot_table(
        index="reference_month", columns="county_fips", values="homes_sold__reltrend12"
    ).dropna()
    assert not wide.empty
    # The fixture gives all three counties the same shape at different scales.
    assert wide["06001"].sub(wide["06037"]).abs().max() < 1e-9
    assert wide["06001"].sub(wide["06063"]).abs().max() < 1e-9


def test_reltrend_and_rollmean_agree_by_construction(long_panel):
    values = long_panel["median_dom"].astype("float64")
    mean = rollmean(long_panel, values, 12, lag_months=0)
    rel = reltrend(long_panel, values, 12, lag_months=0)
    expected = values / mean - 1.0
    pd.testing.assert_series_equal(rel.dropna(), expected.dropna(), check_names=False)


def test_log_diff_yields_na_rather_than_infinity_at_zero(long_panel):
    """A thin county with no sales is a real observation with no growth rate."""
    zeros = pd.Series(0.0, index=long_panel.index)
    result = log_diff(long_panel, zeros, 3, lag_months=0)
    assert result.isna().all()
    assert not np.isinf(result.to_numpy(dtype="float64", na_value=0.0)).any()


# ---------------------------------------------------------------------------
# Missing-data policy
# ---------------------------------------------------------------------------


def test_unemployment_gap_is_filled_and_flagged(long_panel, contract):
    filled, counts = apply_missing_policy(
        long_panel.sort_values(["county_fips", "reference_month"], ignore_index=True), contract
    )
    gap = filled["reference_month"] == UNEMPLOYMENT_GAP

    assert filled.loc[gap, "unemployment_rate"].notna().all()
    assert filled.loc[gap, "unemployment_imputed"].eq(1.0).all()
    assert counts["unemployment_imputed"] == int(gap.sum())
    # Everything outside the gap is untouched.
    assert filled.loc[~gap, "unemployment_imputed"].eq(0.0).all()


def test_the_fill_carries_the_previous_month_not_the_next(long_panel, contract):
    """Backfilling would be a leak; the direction is the whole point."""
    ordered = long_panel.sort_values(["county_fips", "reference_month"], ignore_index=True)
    filled, _ = apply_missing_policy(ordered, contract)

    previous = ordered.loc[
        (ordered["county_fips"] == "06001")
        & (ordered["reference_month"] == UNEMPLOYMENT_GAP - pd.DateOffset(months=1)),
        "unemployment_rate",
    ].iloc[0]
    got = filled.loc[
        (filled["county_fips"] == "06001") & (filled["reference_month"] == UNEMPLOYMENT_GAP),
        "unemployment_rate",
    ].iloc[0]
    assert got == pytest.approx(float(previous))


def test_the_missingness_indicator_carries_the_release_lag(contract):
    """At month t the model sees unemployment from t-2, so it must be told
    whether *that* month was imputed — not whether t was."""
    indicator = next(spec for spec in all_specs(contract) if spec.name == "unemployment_imputed")
    assert indicator.effective_lag == 2


def test_price_drops_never_reaches_the_feature_matrix(long_panel):
    features, _ = build_features(long_panel)
    assert not [name for name in features.columns if name.startswith("price_drops")]


# ---------------------------------------------------------------------------
# Spine and structure
# ---------------------------------------------------------------------------


def test_incomplete_spine_is_rejected(long_panel):
    """Positional shifts only equal calendar shifts on a gap-free spine."""
    holed = long_panel.loc[long_panel["reference_month"] != pd.Timestamp("2019-05-01")]
    with pytest.raises(ValueError, match="incomplete"):
        assert_complete_spine(holed)
    with pytest.raises(ValueError, match="incomplete"):
        build_features(holed)


def test_build_preserves_the_panel_grain(long_panel):
    features, report = build_features(long_panel)
    assert len(features) == len(long_panel)
    assert not features.duplicated(["county_fips", "reference_month"]).any()
    assert report.rows == len(long_panel)
    assert report.feature_count == len(feature_names())


def test_context_columns_survive_but_are_not_features(long_panel):
    features, _ = build_features(long_panel)
    names = set(feature_names())
    for column in ("target_dg", "target_label", "volume_tier", "is_included"):
        assert column in features.columns
        assert column not in names


def test_availability_table_describes_every_feature(long_panel, contract):
    features, _ = build_features(long_panel)
    table = availability_table(features, contract)

    assert list(table["feature"]) == feature_names(contract)
    assert table["definition"].str.len().gt(0).all()
    assert (table["reads_back_months"] >= table["release_lag_months"]).all()

    # The observed first-available month must not precede what the spec predicts.
    observed = table.dropna(subset=["earliest_month_observed"])
    assert (observed["earliest_month_observed"] >= observed["earliest_month_expected"]).all()
