"""Tests for the metric definitions.

Every metric here is checked against a hand-computed value on a small example
rather than against another implementation. The point is that the numbers in
``reports/baselines.md`` mean what the report says they mean — comparing one
implementation to another would only show the two agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from california_housing_pulse.evaluation.metrics import (
    CLASS_ORDER,
    by_group,
    confusion_matrix,
    directional_metrics,
    evaluate,
    forward_growth_metrics,
    magnitude_metrics,
    per_class_metrics,
    probabilistic_metrics,
    skill_score,
)


def test_magnitude_metrics_match_hand_computation():
    actual = pd.Series([0.0, 2.0, -4.0, 6.0])
    predicted = pd.Series([1.0, 0.0, -1.0, 6.0])
    # errors: +1, -2, +3, 0
    got = magnitude_metrics(actual, predicted)

    assert got["n"] == 4
    assert got["mae"] == pytest.approx((1 + 2 + 3 + 0) / 4)
    assert got["rmse"] == pytest.approx(np.sqrt((1 + 4 + 9 + 0) / 4))
    assert got["mean_error"] == pytest.approx((1 - 2 + 3 + 0) / 4)
    assert got["medae"] == pytest.approx(1.5)


def test_mean_error_distinguishes_bias_from_noise():
    """The reason mean error is reported beside MAE and never instead of it."""
    actual = pd.Series([0.0, 0.0, 0.0, 0.0])
    unbiased = pd.Series([3.0, -3.0, 3.0, -3.0])
    biased = pd.Series([3.0, 3.0, 3.0, 3.0])

    left, right = magnitude_metrics(actual, unbiased), magnitude_metrics(actual, biased)
    assert left["mae"] == right["mae"] == pytest.approx(3.0)
    assert left["mean_error"] == pytest.approx(0.0)
    assert right["mean_error"] == pytest.approx(3.0)


def test_magnitude_metrics_ignore_unpaired_rows():
    actual = pd.Series([1.0, np.nan, 3.0])
    predicted = pd.Series([1.0, 5.0, np.nan])
    assert magnitude_metrics(actual, predicted)["n"] == 1


def test_skill_score_is_a_fractional_reduction():
    assert skill_score(4.0, 5.0) == pytest.approx(0.2)
    assert skill_score(5.0, 5.0) == pytest.approx(0.0)
    assert skill_score(10.0, 5.0) == pytest.approx(-1.0)
    assert np.isnan(skill_score(1.0, 0.0))


def test_confusion_matrix_keeps_classes_the_model_never_predicts():
    actual = pd.Series(["cooling", "stable", "heating", "heating"])
    predicted = pd.Series(["cooling", "cooling", "heating", "cooling"])

    matrix = confusion_matrix(actual, predicted)
    assert list(matrix.index) == list(CLASS_ORDER)
    assert list(matrix.columns) == list(CLASS_ORDER)
    # 'stable' is never predicted; its column must survive as zeros.
    assert matrix["stable"].sum() == 0
    assert matrix.loc["cooling", "cooling"] == 1
    assert matrix.loc["heating", "cooling"] == 1
    assert matrix.to_numpy().sum() == 4


def test_per_class_metrics_match_hand_computation():
    actual = pd.Series(["cooling"] * 3 + ["heating"] * 2)
    predicted = pd.Series(["cooling", "cooling", "heating", "heating", "cooling"])

    table = per_class_metrics(actual, predicted).set_index("class")
    # cooling: predicted 3 times, 2 correct -> precision 2/3; actual 3, recall 2/3
    assert table.loc["cooling", "precision"] == pytest.approx(2 / 3)
    assert table.loc["cooling", "recall"] == pytest.approx(2 / 3)
    # heating: predicted twice, 1 correct -> precision 1/2; actual twice -> recall 1/2
    assert table.loc["heating", "precision"] == pytest.approx(0.5)
    assert table.loc["heating", "recall"] == pytest.approx(0.5)
    assert table.loc["stable", "f1"] == 0.0


def test_macro_f1_refuses_to_reward_always_guessing_the_majority():
    """The property that makes macro-F1 the primary directional metric."""
    actual = pd.Series(["cooling"] * 8 + ["stable"] + ["heating"])
    always_majority = pd.Series(["cooling"] * 10)

    got = directional_metrics(actual, always_majority)
    assert got["accuracy"] == pytest.approx(0.8)
    # Two of three classes score zero F1, so macro-F1 stays near a third of the
    # one class it gets right.
    assert got["macro_f1"] < 0.31
    assert got["balanced_accuracy"] == pytest.approx(1 / 3)


def test_probabilistic_metrics_match_hand_computation():
    actual = pd.Series(["cooling", "heating"])
    probabilities = pd.DataFrame(
        {"cooling": [0.7, 0.2], "stable": [0.2, 0.3], "heating": [0.1, 0.5]}
    )
    got = probabilistic_metrics(actual, probabilities)

    assert got["log_loss"] == pytest.approx(-(np.log(0.7) + np.log(0.5)) / 2)
    first = (0.7 - 1) ** 2 + 0.2**2 + 0.1**2
    second = 0.2**2 + 0.3**2 + (0.5 - 1) ** 2
    assert got["brier"] == pytest.approx((first + second) / 2)


def test_a_perfect_forecast_scores_zero_on_both_probabilistic_metrics():
    actual = pd.Series(["stable", "heating"])
    probabilities = pd.DataFrame(
        {"cooling": [0.0, 0.0], "stable": [1.0, 0.0], "heating": [0.0, 1.0]}
    )
    got = probabilistic_metrics(actual, probabilities)
    assert got["log_loss"] == pytest.approx(0.0)
    assert got["brier"] == pytest.approx(0.0)


def test_brier_is_bounded_above_by_two():
    """Confidently wrong every time is the worst case, and it equals 2."""
    actual = pd.Series(["cooling", "cooling"])
    probabilities = pd.DataFrame(
        {"cooling": [0.0, 0.0], "stable": [0.0, 0.0], "heating": [1.0, 1.0]}
    )
    assert probabilistic_metrics(actual, probabilities)["brier"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The base-effect diagnostic
# ---------------------------------------------------------------------------


def _forward_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    b = rng.normal(0, 7, 300)
    f = rng.normal(1.5, 6, 300)
    return pd.DataFrame({"price_smoothed__log_diff3_o9": b, "target_dg": f - b})


def test_forward_metrics_leave_mae_unchanged():
    """Adding the same b to prediction and outcome cannot move any error."""
    frame = _forward_frame()
    frame["predicted_dg"] = 1.5 - frame["price_smoothed__log_diff3_o9"]

    got = forward_growth_metrics(frame)
    direct = magnitude_metrics(frame["target_dg"], frame["predicted_dg"])
    assert got["mae_forward"] == pytest.approx(direct["mae"])


def test_a_pure_base_effect_forecast_looks_skilful_on_dg_and_is_not():
    """The whole reason the diagnostic exists.

    A forecaster who only subtracts the observable base effect correlates
    strongly with the target, because both contain -b. Measured against the part
    that had not happened yet, it knows nothing.
    """
    frame = _forward_frame()
    frame["predicted_dg"] = 1.5 - frame["price_smoothed__log_diff3_o9"]

    got = forward_growth_metrics(frame)
    assert got["corr_dg"] > 0.6
    assert abs(got["corr_forward"]) < 0.15


def test_correlation_is_na_when_a_prediction_is_constant():
    frame = _forward_frame()
    frame["predicted_dg"] = 0.0
    got = forward_growth_metrics(frame)
    assert np.isnan(got["corr_dg"])


def test_forward_metrics_are_skipped_without_the_base_column():
    frame = pd.DataFrame({"target_dg": [1.0], "predicted_dg": [1.0]})
    assert forward_growth_metrics(frame) == {}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_evaluate_counts_rows_for_a_classifier_with_an_empty_magnitude_column():
    """Predictions from every model share one frame, so a classifier's rows still
    carry an all-NA `predicted_dg`. The row count must come from what scored."""
    frame = pd.DataFrame(
        {
            "target_dg": [1.0, 2.0],
            "predicted_dg": [np.nan, np.nan],
            "target_label": ["cooling", "heating"],
            "predicted_label": ["cooling", "cooling"],
        }
    )
    assert evaluate(frame)["n"] == 2


def test_evaluate_reports_only_what_the_model_predicted():
    magnitude_only = pd.DataFrame({"target_dg": [1.0], "predicted_dg": [1.5]})
    got = evaluate(magnitude_only)
    assert "mae" in got
    assert "macro_f1" not in got


def test_by_group_partitions_the_rows_it_was_given():
    frame = pd.DataFrame(
        {
            "volume_tier": ["thin", "thin", "large", "large"],
            "target_dg": [0.0, 0.0, 0.0, 0.0],
            "predicted_dg": [10.0, 10.0, 1.0, 1.0],
        }
    )
    table = by_group(frame, "volume_tier").set_index("volume_tier")
    assert table.loc["thin", "mae"] == pytest.approx(10.0)
    assert table.loc["large", "mae"] == pytest.approx(1.0)
    assert table["n"].sum() == len(frame)
