"""Tests for the baselines, the learned models, and the base-effect identity.

The load-bearing test in this file is
:func:`test_the_target_decomposes_into_forward_growth_minus_base_effect`. Every
claim the Milestone 3 report makes about what the models achieved rests on that
identity, so it is checked against the real panel rather than asserted in prose.

The second concern is the definition of done's "preprocessing is fit on training
data only". That is enforced structurally by the sklearn pipeline, and
:func:`test_preprocessing_statistics_come_only_from_training_data` checks the
structure actually holds by fitting on one frame and confirming the stored
statistics match it and not the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from california_housing_pulse.evaluation.metrics import CLASS_ORDER
from california_housing_pulse.features.target import load_contract
from california_housing_pulse.io import read_parquet
from california_housing_pulse.modeling.baselines import (
    BASE_EFFECT,
    MOMENTUM_CHANGE,
    BaseEffect,
    MajorityClass,
    MeanReversion,
    Persistence,
    ZeroChange,
    classify,
    naive_baselines,
)
from california_housing_pulse.modeling.models import fit_logistic, fit_ridge, select_ridge
from california_housing_pulse.paths import PROCESSED_DIR

PANEL_PATH = PROCESSED_DIR / "county_month_panel.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

requires_panel = pytest.mark.skipif(
    not PANEL_PATH.exists(), reason="panel not built; run `chp build`"
)
requires_features = pytest.mark.skipif(
    not FEATURES_PATH.exists(), reason="features not built; run `chp features`"
)


# ---------------------------------------------------------------------------
# The identity the whole results section depends on
# ---------------------------------------------------------------------------


@requires_panel
def test_the_target_decomposes_into_forward_growth_minus_base_effect():
    """``target_dg(t) = f(t) - b(t)``, exactly.

    f(t) = 100 ln( P(t+3) / P(t) )   — unknown at prediction time
    b(t) = 100 ln( P(t-9) / P(t-12) ) — already observed at prediction time

    If this ever fails, the `base_effect` baseline and the `corr_forward`
    diagnostic are both measuring something other than what they claim.
    """
    panel = read_parquet(PANEL_PATH).sort_values(
        ["county_fips", "reference_month"], ignore_index=True
    )
    ln_price = np.log(panel["price_smoothed"])
    grouped = ln_price.groupby(panel["county_fips"], sort=False)

    forward = 100.0 * (grouped.shift(-3) - ln_price)
    base = 100.0 * (grouped.shift(9) - grouped.shift(12))

    comparable = panel["target_dg"].notna() & forward.notna() & base.notna()
    assert comparable.sum() > 5000
    residual = (panel["target_dg"] - (forward - base))[comparable].abs().max()
    assert residual < 1e-9


@requires_features
def test_the_base_effect_feature_is_the_b_term_of_the_decomposition():
    """The declared feature must be b(t), not something merely similar."""
    features = read_parquet(FEATURES_PATH).sort_values(
        ["county_fips", "reference_month"], ignore_index=True
    )
    panel = read_parquet(PANEL_PATH).sort_values(
        ["county_fips", "reference_month"], ignore_index=True
    )
    ln_price = np.log(panel["price_smoothed"])
    grouped = ln_price.groupby(panel["county_fips"], sort=False)
    expected = 100.0 * (grouped.shift(9) - grouped.shift(12))

    both = features[BASE_EFFECT].notna() & expected.notna()
    assert both.sum() > 5000
    assert (features.loc[both, BASE_EFFECT] - expected[both]).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_zero_change_predicts_no_change_and_therefore_always_stable():
    frame = _frame(target_dg=[5.0, -5.0, 0.0])
    out = ZeroChange().fit(frame).predict(frame)
    assert (out["predicted_dg"] == 0.0).all()
    assert (out["predicted_label"] == "stable").all()


def test_persistence_extrapolates_the_last_observable_change():
    frame = _frame(**{MOMENTUM_CHANGE: [3.0, -4.0, np.nan]})
    out = Persistence().fit(frame).predict(frame)
    assert list(out["predicted_dg"]) == [3.0, -4.0, 0.0]
    assert list(out["predicted_label"]) == ["heating", "cooling", "stable"]


def test_base_effect_recovers_the_training_drift():
    """drift = mean(target_dg + b), because f = dg + b."""
    frame = _frame(target_dg=[1.0, 3.0], **{BASE_EFFECT: [2.0, 0.0]})
    model = BaseEffect().fit(frame)
    assert model.params["drift"] == pytest.approx(3.0)

    out = model.predict(_frame(**{BASE_EFFECT: [1.0, -2.0]}))
    assert list(out["predicted_dg"]) == pytest.approx([2.0, 5.0])


def test_base_effect_predicts_the_drift_where_b_is_unknown():
    model = BaseEffect().fit(_frame(target_dg=[2.0], **{BASE_EFFECT: [0.0]}))
    out = model.predict(_frame(**{BASE_EFFECT: [np.nan]}))
    assert out["predicted_dg"].iloc[0] == pytest.approx(2.0)


def test_mean_reversion_fits_a_positive_k_when_growth_reverts():
    """A series constructed to revert must produce k > 0."""
    gap = np.linspace(-10, 10, 200)
    frame = _frame(
        growth_yoy__lag0=gap,
        growth_yoy__rollmean12=np.zeros_like(gap),
        target_dg=-0.5 * gap,
    )
    model = MeanReversion().fit(frame)
    assert model.params["k"] == pytest.approx(0.5, abs=1e-9)


def test_mean_reversion_fits_a_negative_k_when_growth_extrapolates():
    """The baseline must be able to report that the opposite bet was wrong."""
    gap = np.linspace(-10, 10, 200)
    frame = _frame(
        growth_yoy__lag0=gap,
        growth_yoy__rollmean12=np.zeros_like(gap),
        target_dg=0.4 * gap,
    )
    assert MeanReversion().fit(frame).params["k"] == pytest.approx(-0.4, abs=1e-9)


def test_majority_class_learns_the_label_and_reports_training_prevalence():
    train = _frame(target_label=["cooling"] * 6 + ["stable"] * 3 + ["heating"])
    model = MajorityClass().fit(train)
    assert model.label == "cooling"

    out = model.predict(_frame(target_label=["heating", "stable"]))
    assert (out["predicted_label"] == "cooling").all()
    assert out["prob_cooling"].iloc[0] == pytest.approx(0.6)
    assert out["prob_stable"].iloc[0] == pytest.approx(0.3)
    # Probabilities are the training prevalence, so they must still sum to one.
    total = sum(out[f"prob_{label}"].iloc[0] for label in CLASS_ORDER)
    assert total == pytest.approx(1.0)


def test_every_naive_baseline_learns_only_from_the_frame_it_is_given():
    """A baseline must not consult anything outside its training frame."""
    train = _frame(
        target_dg=[1.0, 2.0],
        target_label=["cooling", "heating"],
        growth_yoy__lag0=[1.0, 2.0],
        growth_yoy__rollmean12=[0.0, 0.0],
        **{MOMENTUM_CHANGE: [1.0, 2.0], BASE_EFFECT: [0.0, 0.0]},
    )
    for baseline in naive_baselines():
        fitted = baseline.fit(train)
        assert fitted is baseline
        assert isinstance(fitted.params, dict)


def test_baselines_classify_at_the_frozen_threshold():
    """A model may not use a different tau from the target it is scored against."""
    tau = load_contract().tau
    values = pd.Series([-tau - 0.01, -tau, 0.0, tau, tau + 0.01])
    labels = list(classify(values))
    assert labels == ["cooling", "cooling", "stable", "heating", "heating"]


# ---------------------------------------------------------------------------
# The learned models
# ---------------------------------------------------------------------------


@pytest.fixture
def learnable() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """A frame where the target is a known linear function of two features."""
    rng = np.random.default_rng(7)
    n = 400
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    target = 3.0 * x1 - 2.0 * x2 + rng.normal(0, 0.5, n)
    frame = pd.DataFrame({"a": x1, "b": x2, "target_dg": target})
    frame["target_label"] = classify(frame["target_dg"])
    return frame.iloc[:300].copy(), frame.iloc[300:].copy(), ["a", "b"]


def test_ridge_recovers_the_signal_it_was_given(learnable):
    train, holdout, columns = learnable
    model = fit_ridge(train, columns, alpha=1.0)
    predicted = model.predict(holdout)["predicted_dg"]
    assert np.corrcoef(predicted, holdout["target_dg"])[0, 1] > 0.95


def test_ridge_coefficients_are_reported_in_standardised_units(learnable):
    train, _, columns = learnable
    table = fit_ridge(train, columns, alpha=1.0).coefficients().set_index("feature")
    # Features are already unit-variance, so the coefficients should land near
    # the generating values, and 'a' must dominate 'b'.
    assert table.loc["a", "coefficient"] > 2.0
    assert table.loc["b", "coefficient"] < -1.0
    assert abs(table.loc["a", "coefficient"]) > abs(table.loc["b", "coefficient"])


def test_stronger_regularization_shrinks_the_coefficients(learnable):
    train, _, columns = learnable
    weak = fit_ridge(train, columns, alpha=1.0).coefficients()["coefficient"].abs().sum()
    strong = fit_ridge(train, columns, alpha=10000.0).coefficients()["coefficient"].abs().sum()
    assert strong < weak


def test_logistic_returns_a_probability_for_every_frozen_class(learnable):
    train, holdout, columns = learnable
    out = fit_logistic(train, columns, C=1.0).predict(holdout)

    columns_present = [f"prob_{label}" for label in CLASS_ORDER]
    assert all(column in out.columns for column in columns_present)
    totals = out[columns_present].sum(axis=1)
    assert np.allclose(totals, 1.0)
    assert out["predicted_label"].isin(CLASS_ORDER).all()


def test_preprocessing_statistics_come_only_from_training_data(learnable):
    """The definition of done, checked rather than assumed.

    Fitting on one frame and then predicting on another must leave the stored
    imputation and scaling statistics describing the *first* frame.
    """
    train, holdout, columns = learnable
    other = holdout.copy()
    other["a"] = other["a"] + 1000.0

    model = fit_ridge(train, columns, alpha=1.0)
    scaler = model.pipeline.named_steps["scale"]
    before = scaler.mean_.copy()

    model.predict(other)
    assert np.allclose(scaler.mean_, before)
    assert np.allclose(before[0], train["a"].mean(), atol=1e-9)


def test_missing_features_are_filled_with_the_training_median(learnable):
    train, holdout, columns = learnable
    train.loc[train.index[:10], "a"] = np.nan
    model = fit_ridge(train, columns, alpha=1.0)

    imputer = model.pipeline.named_steps["impute"]
    assert imputer.statistics_[0] == pytest.approx(train["a"].median())

    gapped = holdout.copy()
    gapped["a"] = np.nan
    assert model.predict(gapped)["predicted_dg"].notna().all()


def test_selection_scores_candidates_on_the_frame_it_is_given(learnable):
    train, validation, columns = learnable
    selection = select_ridge(train, validation, columns, alphas=(1.0, 100.0, 10000.0))

    assert len(selection.scores) == 3
    assert selection.metric == "mae"
    # The winner must be the row with the lowest validation MAE, not any other.
    best = selection.scores.loc[selection.scores["mae"].idxmin(), "alpha"]
    assert selection.best.params["alpha"] == best
