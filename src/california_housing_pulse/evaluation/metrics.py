"""Magnitude, directional and probabilistic metrics, with grouped reporting.

The plan asks for MAE and RMSE in percentage points *with mean error*, macro-F1,
balanced accuracy, per-class precision and recall, a three-class confusion
matrix, multiclass log loss and Brier score — and Milestone 2 Decisions 2 and 3
add that every one of these must be reportable **within volume tier** and
**within period**, not only pooled.

That last requirement is why this module exists rather than calling
``sklearn.metrics.classification_report`` directly. A pooled macro-F1 describes
neither a large county nor a thin one: measured prevalence of ``stable`` runs
from 47.7% in large counties to 9.1% in thin ones. Everything here therefore
returns a tidy frame keyed by group, and :func:`by_group` is the only way any
caller aggregates.

Two deliberate choices:

*Mean error is reported beside MAE, never instead of it.* MAE says how wrong the
forecast is; mean error says whether it is systematically high or low. A model
that predicts +3 half the time and -3 the other half has the same MAE as one that
always predicts +3, and they are very different failures.

*The primary metrics are named in code.* :data:`PRIMARY_MAGNITUDE` and
:data:`PRIMARY_DIRECTIONAL` are what the plan committed to before any result was
seen, so a later table cannot quietly promote whichever number came out best.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed once, in the plan, before any model was fitted.
PRIMARY_MAGNITUDE = "mae"
PRIMARY_DIRECTIONAL = "macro_f1"

# The frozen class order. Used for the confusion matrix axes and the columns of
# any probability matrix, so those two can never fall out of step.
CLASS_ORDER = ("cooling", "stable", "heating")


# ---------------------------------------------------------------------------
# Magnitude
# ---------------------------------------------------------------------------


def magnitude_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MAE, RMSE, mean error and median absolute error, all in percentage points."""
    actual = pd.Series(actual).astype("float64")
    predicted = pd.Series(predicted).astype("float64")
    both = actual.notna() & predicted.notna()
    actual, predicted = actual[both], predicted[both]

    if actual.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mean_error": np.nan, "medae": np.nan}

    error = predicted - actual
    return {
        "n": int(len(actual)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        # Signed, so a systematic bias is visible rather than cancelled by MAE.
        "mean_error": float(error.mean()),
        "medae": float(error.abs().median()),
    }


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """Fractional MAE reduction against a baseline. Negative means worse.

    The plan requires comparison against naive baselines rather than against
    random guessing, and a ratio makes that comparison legible: 0.10 means the
    model removed a tenth of the naive error, and 0 means it achieved nothing.
    """
    if not np.isfinite(baseline_mae) or baseline_mae == 0:
        return np.nan
    return float(1.0 - model_mae / baseline_mae)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


def confusion_matrix(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Three-class confusion matrix with rows = actual, columns = predicted.

    Reindexed onto :data:`CLASS_ORDER` so a class the model never predicts still
    appears as a column of zeros. Silently dropping it would make the matrix look
    better than the model is.
    """
    matrix = pd.crosstab(
        pd.Series(actual, name="actual").astype("object"),
        pd.Series(predicted, name="predicted").astype("object"),
    )
    return matrix.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0).astype(int)


def per_class_metrics(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Precision, recall, F1 and support for each of the three classes."""
    matrix = confusion_matrix(actual, predicted)
    rows = []
    for label in CLASS_ORDER:
        true_positive = int(matrix.loc[label, label])
        predicted_positive = int(matrix[label].sum())
        actual_positive = int(matrix.loc[label].sum())

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
        rows.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": actual_positive,
                "predicted": predicted_positive,
            }
        )
    return pd.DataFrame(rows)


def directional_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Accuracy, macro-F1 and balanced accuracy.

    Macro-F1 is the primary directional metric: it weights all three classes
    equally, so a model that reaches high accuracy by always answering with the
    prevailing class is not rewarded for it.
    """
    actual = pd.Series(actual).astype("object")
    predicted = pd.Series(predicted).astype("object")
    both = actual.notna() & predicted.notna()
    actual, predicted = actual[both], predicted[both]

    if actual.empty:
        return {
            "n": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
        }

    per_class = per_class_metrics(actual, predicted)
    # Balanced accuracy is mean recall over the classes actually present, so an
    # absent class does not drag the score toward zero for a reason unrelated to
    # the model.
    present = per_class.loc[per_class["support"] > 0]
    return {
        "n": int(len(actual)),
        "accuracy": float((actual.to_numpy() == predicted.to_numpy()).mean()),
        "macro_f1": float(per_class["f1"].mean()),
        "balanced_accuracy": float(present["recall"].mean()),
    }


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


def probabilistic_metrics(
    actual: pd.Series,
    probabilities: pd.DataFrame,
    epsilon: float = 1e-15,
) -> dict[str, float]:
    """Multiclass log loss and the multiclass (Brier) quadratic score.

    ``probabilities`` must carry one column per class in :data:`CLASS_ORDER`.
    Brier is the sum of squared differences between the predicted vector and the
    one-hot outcome, averaged over rows — the standard multiclass form, which
    sklearn does not provide, and which ranges from 0 (perfect) to 2 (confidently
    wrong every time).
    """
    actual = pd.Series(actual).astype("object")
    probabilities = probabilities.reindex(columns=list(CLASS_ORDER))
    both = actual.notna() & probabilities.notna().all(axis=1)
    actual, probabilities = actual[both], probabilities.loc[both]

    if actual.empty:
        return {"n": 0, "log_loss": np.nan, "brier": np.nan}

    values = probabilities.to_numpy(dtype="float64")
    # Renormalise defensively: a caller may hand over rounded probabilities, and
    # a row summing to 0.999 would otherwise bias both scores.
    values = values / values.sum(axis=1, keepdims=True)

    one_hot = np.zeros_like(values)
    index = {label: i for i, label in enumerate(CLASS_ORDER)}
    positions = actual.map(index).to_numpy(dtype="int64")
    one_hot[np.arange(len(actual)), positions] = 1.0

    clipped = np.clip(values, epsilon, 1.0)
    return {
        "n": int(len(actual)),
        "log_loss": float(-np.log(clipped[np.arange(len(actual)), positions]).mean()),
        "brier": float(((values - one_hot) ** 2).sum(axis=1).mean()),
    }


# ---------------------------------------------------------------------------
# Everything at once, and grouped
# ---------------------------------------------------------------------------


def evaluate(
    frame: pd.DataFrame,
    *,
    actual: str = "target_dg",
    predicted: str = "predicted_dg",
    actual_label: str = "target_label",
    predicted_label: str = "predicted_label",
    probability_prefix: str | None = "prob_",
) -> dict[str, float]:
    """Every metric a single set of predictions supports.

    Magnitude, direction and probability are all optional: a majority-class
    baseline has no magnitude prediction and a ridge has no probabilities, and
    each simply contributes nothing rather than raising. The keys are stable
    across models so the results table has one shape.
    """
    result: dict[str, float] = {}

    if predicted in frame.columns:
        result.update(magnitude_metrics(frame[actual], frame[predicted]))
    if predicted_label in frame.columns:
        directional = directional_metrics(frame[actual_label], frame[predicted_label])
        # `n` is shared; keep the magnitude count if one is already present.
        result.setdefault("n", directional["n"])
        result.update({k: v for k, v in directional.items() if k != "n"})
    if probability_prefix:
        columns = [f"{probability_prefix}{label}" for label in CLASS_ORDER]
        if all(column in frame.columns for column in columns):
            probabilities = frame[columns]
            probabilities.columns = list(CLASS_ORDER)
            probabilistic = probabilistic_metrics(frame[actual_label], probabilities)
            result.update({k: v for k, v in probabilistic.items() if k != "n"})

    return result


def by_group(frame: pd.DataFrame, group: str | list[str], **kwargs) -> pd.DataFrame:
    """Evaluate within each group — the only aggregation this project uses.

    Milestone 2 Decision 2: a pooled score is dominated by whichever segment is
    largest or noisiest, and a model can appear to improve simply by getting
    better at the counties that matter least. Reporting by ``volume_tier`` and by
    period is how that stays visible.
    """
    keys = [group] if isinstance(group, str) else list(group)
    rows = []
    for values, part in frame.groupby(keys, dropna=False, observed=True):
        values = values if isinstance(values, tuple) else (values,)
        rows.append({**dict(zip(keys, values, strict=True)), **evaluate(part, **kwargs)})
    return pd.DataFrame(rows).sort_values(keys, ignore_index=True)


def by_period(frame: pd.DataFrame, freq: str = "YE", **kwargs) -> pd.DataFrame:
    """Evaluate within each calendar period of the reference month.

    The definition of done requires variation across time rather than a single
    aggregate, and Milestone 2 Finding 3 is the reason: the class mix swings from
    70.5% cooling in 2022 to 54.8% heating in 2023, far more than any model will.
    """
    working = frame.copy()
    working["period"] = working["reference_month"].dt.to_period(freq[0]).astype("string")
    return by_group(working, "period", **kwargs)
