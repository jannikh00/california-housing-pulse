"""Tests for the frozen chronological split.

Two jobs. The first is a guard, in the same spirit as
``test_target.py::test_contract_matches_the_frozen_values``: the boundaries were
written down before any model was fitted, and this suite fails if they move. The
second is to check the properties the boundaries are *for* — that the split is
chronological, that no row is used twice, and that the embargo genuinely stops
training labels from resolving inside the validation window.

Row-count assertions run against the real feature matrix when it has been built,
and skip otherwise, so a clean checkout without the 241 MB Redfin download still
passes the rest.
"""

from __future__ import annotations

import pandas as pd
import pytest

from california_housing_pulse.features.target import load_contract
from california_housing_pulse.io import read_parquet
from california_housing_pulse.modeling.split import (
    EMBARGO,
    SPLIT_ORDER,
    TEST,
    TRAIN,
    VALIDATION,
    assign_split,
    check_no_label_crosses_a_boundary,
    eligible,
    load_split_contract,
    summarize_split,
)
from california_housing_pulse.paths import PROCESSED_DIR

FEATURES_PATH = PROCESSED_DIR / "features.parquet"

requires_features = pytest.mark.skipif(
    not FEATURES_PATH.exists(),
    reason="features.parquet not built; run `chp build && chp features`",
)


@pytest.fixture
def split_contract():
    return load_split_contract()


@pytest.fixture
def features():
    return read_parquet(FEATURES_PATH)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_contract_matches_the_frozen_values(split_contract):
    """Frozen 11 August 2026, before any model was fitted.

    If this fails, the split moved. That is only legitimate as a deliberate,
    documented re-freeze made *before* the test set is read again — never as a
    response to a result.
    """
    assert split_contract.frozen_status == "frozen"
    assert split_contract.frozen_date == "2026-08-11"
    assert split_contract.embargo_months == 3
    assert split_contract.lead_in_first_month == pd.Timestamp("2014-03-01")
    assert split_contract.require_complete_features is False

    assert split_contract.window(TRAIN).start is None
    assert split_contract.window(TRAIN).end == pd.Timestamp("2023-11-01")
    assert split_contract.window(VALIDATION).start == pd.Timestamp("2024-03-01")
    assert split_contract.window(VALIDATION).end == pd.Timestamp("2024-11-01")
    assert split_contract.window(TEST).start == pd.Timestamp("2025-03-01")
    assert split_contract.window(TEST).end == pd.Timestamp("2026-02-01")

    assert split_contract.expected_rows == {
        "train": 6271,
        "validation": 486,
        "test": 648,
        "embargo": 324,
        "eligible": 7729,
    }


def test_the_embargo_is_exactly_one_forecast_horizon(split_contract):
    """The embargo exists to cover the horizon; a shorter one would not."""
    assert split_contract.embargo_months == load_contract().horizon_months


def test_the_gap_between_windows_is_the_embargo(split_contract):
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:], strict=False):
        end = split_contract.window(earlier).end
        start = split_contract.window(later).start
        gap = (start.year - end.year) * 12 + (start.month - end.month)
        assert gap == split_contract.embargo_months + 1, (
            f"{earlier} ends {end:%Y-%m} and {later} starts {start:%Y-%m}, "
            f"leaving {gap - 1} embargoed months"
        )


def test_windows_are_chronological_and_do_not_overlap(split_contract):
    windows = [split_contract.window(name) for name in SPLIT_ORDER]
    for earlier, later in zip(windows, windows[1:], strict=False):
        assert earlier.end < later.start


def test_the_lead_in_covers_the_deepest_feature(split_contract):
    """The first eligible month must clear the deepest feature's history.

    Two chains reach back, and the lead-in has to clear both. Features drawn
    straight from a raw source start from the panel's first month, 2012-01, and
    the deepest of those reads 14 months. Features built on ``growth_yoy`` start
    from *its* first month, 2013-03, because the target contract's own 14-month
    lead-in has already been spent — and the deepest of those reads a further 12.
    The second chain binds, which is why the answer is 2014-03 and not 2013-03.
    """
    from california_housing_pulse.features.build import all_specs
    from california_housing_pulse.features.spec import load_feature_contract

    contract = load_feature_contract()
    specs = all_specs(contract)

    panel_start = pd.Timestamp("2012-01-01")
    growth_start = pd.Timestamp("2013-03-01")

    from_raw = max(spec.reach_months for spec in specs if spec.column != "growth_yoy")
    from_growth = max(spec.reach_months for spec in specs if spec.column == "growth_yoy")
    assert from_raw == 14
    assert from_growth == 12

    required = max(
        panel_start + pd.DateOffset(months=from_raw),
        growth_start + pd.DateOffset(months=from_growth),
    )
    assert required == pd.Timestamp("2014-03-01")
    assert split_contract.lead_in_first_month == required


# ---------------------------------------------------------------------------
# Properties of the assignment
# ---------------------------------------------------------------------------


@requires_features
def test_row_counts_match_the_frozen_contract(features, split_contract):
    split = assign_split(features)
    counts = split.value_counts()
    for name, expected in split_contract.expected_rows.items():
        if name == "eligible":
            assert int(eligible(features).sum()) == expected
        else:
            assert int(counts.get(name, 0)) == expected, f"{name} row count moved"


@requires_features
def test_every_row_lands_in_exactly_one_split(features):
    split = assign_split(features)
    assert len(split) == len(features)
    assert split.notna().all()
    assert set(split.unique()) <= {TRAIN, VALIDATION, TEST, EMBARGO, "ineligible"}


@requires_features
def test_the_splits_are_chronological_in_the_data(features):
    """Not merely configured chronologically — actually ordered in the rows."""
    frame = features.assign(split=assign_split(features))
    months = {name: frame.loc[frame["split"] == name, "reference_month"] for name in SPLIT_ORDER}
    assert months[TRAIN].max() < months[VALIDATION].min()
    assert months[VALIDATION].max() < months[TEST].min()


@requires_features
def test_no_training_label_resolves_inside_a_later_split(features):
    """The property the embargo exists to create, recomputed from the data."""
    split = assign_split(features)
    problems = check_no_label_crosses_a_boundary(features, split, load_contract().horizon_months)
    assert problems == [], problems


@requires_features
def test_the_embargo_falls_between_the_windows(features, split_contract):
    frame = features.assign(split=assign_split(features))
    embargoed = frame.loc[frame["split"] == EMBARGO, "reference_month"]
    assert len(embargoed) == split_contract.expected_rows["embargo"]
    assert embargoed.nunique() == 2 * split_contract.embargo_months

    train_end = split_contract.window(TRAIN).end
    test_start = split_contract.window(TEST).start
    assert embargoed.min() > train_end
    assert embargoed.max() < test_start


@requires_features
def test_every_county_appears_in_every_split(features):
    """The reason eligibility is a date rather than a completeness test.

    Under strict completeness Del Norte vanished from the training set and
    Colusa kept 22 of 117 months. Milestone 2 committed to reporting error by
    volume tier, which is not meaningful if a tier is missing from training.
    """
    frame = features.assign(split=assign_split(features))
    per_split = {
        name: set(frame.loc[frame["split"] == name, "county_fips"]) for name in SPLIT_ORDER
    }
    assert len(per_split[TRAIN]) == 54
    assert per_split[TRAIN] == per_split[VALIDATION] == per_split[TEST]


@requires_features
def test_every_volume_tier_is_represented_in_training(features):
    frame = features.assign(split=assign_split(features))
    train = frame.loc[frame["split"] == TRAIN]
    assert set(train["volume_tier"].unique()) == {"thin", "small", "mid", "large"}
    assert (train["volume_tier"] == "thin").sum() > 500


@requires_features
def test_no_ineligible_row_carries_a_split(features):
    frame = features.assign(split=assign_split(features))
    used = frame.loc[frame["split"].isin(SPLIT_ORDER)]
    assert used["is_included"].all()
    assert used["has_target"].all()
    assert used["target_dg"].notna().all()
    assert (used["reference_month"] >= load_split_contract().lead_in_first_month).all()


@requires_features
def test_summary_reports_each_period_class_mix(features):
    report = summarize_split(features, assign_split(features))
    for name in SPLIT_ORDER:
        shares = report.prevalence[name]
        assert set(shares) == {"cooling", "stable", "heating"}
        assert sum(shares.values()) == pytest.approx(1.0)
    assert "train" in report.summary()
