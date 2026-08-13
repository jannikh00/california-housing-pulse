"""The learned baselines: regularized linear regression and multinomial logistic.

Both are wrapped in a :class:`sklearn.pipeline.Pipeline` whose first two steps
are imputation and standardisation. That is not decoration — it is how the
definition of done's "preprocessing is fit on training data only" becomes
structural rather than a rule someone remembers. A pipeline fitted on the
training frame holds the training medians and the training means and standard
deviations; calling ``predict`` on the test frame applies those stored values and
has no way to consult test statistics even if the caller wanted it to.

The imputer is where the split's ``require_complete: false`` decision is
honoured. Roughly 0.4% of feature cells are missing after the lead-in, almost all
of them sporadic Redfin gaps in thin counties, and filling them with a training
median keeps those counties in the model instead of deleting the segment
Milestone 2 asked to keep visible.

**Hyperparameters are chosen on validation, never on test.** :func:`select` scans
a grid, scores each fit on the validation split, and returns the winner. The test
split is not passed to it and the function has no parameter that would accept it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluation.metrics import CLASS_ORDER, directional_metrics, magnitude_metrics
from .baselines import classify

# Scanned on validation. Wide and log-spaced: with 59 correlated features over
# ~6,300 rows the useful amount of shrinkage is not obvious in advance, and a
# grid that bottoms out at its own edge would be hiding the answer.
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
LOGISTIC_CS = (0.001, 0.01, 0.1, 1.0, 10.0)

RANDOM_STATE = 20260811


def _preprocessor() -> list[tuple[str, object]]:
    """Impute then standardise — both fitted on whatever frame is passed to fit.

    Standardisation matters for more than convergence here: the features are in
    wildly different units (percentage points, days, unitless ratios), and a
    single ridge penalty applied to unscaled coefficients would effectively
    regularise the small-scale features far harder than the large-scale ones.
    """
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]


@dataclass
class LearnedModel:
    """A fitted pipeline plus the metadata the results table needs."""

    name: str
    kind: str
    pipeline: Pipeline
    features: list[str]
    params: dict[str, float] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = frame[self.features]
        if self.kind == "magnitude":
            predicted = pd.Series(self.pipeline.predict(matrix), index=frame.index, dtype="float64")
            return pd.DataFrame(
                {"predicted_dg": predicted, "predicted_label": classify(predicted)},
                index=frame.index,
            )

        labels = pd.Series(self.pipeline.predict(matrix), index=frame.index, dtype="string")
        out = pd.DataFrame({"predicted_label": labels}, index=frame.index)
        probabilities = self.pipeline.predict_proba(matrix)
        for position, label in enumerate(self.pipeline.classes_):
            out[f"prob_{label}"] = probabilities[:, position]
        # A class absent from training would otherwise have no column at all,
        # and the probabilistic metrics would fail on a missing key rather than
        # on the real problem.
        for label in CLASS_ORDER:
            if f"prob_{label}" not in out.columns:
                out[f"prob_{label}"] = 0.0
        return out

    def coefficients(self) -> pd.DataFrame:
        """Standardised coefficients, largest absolute effect first.

        Readable as effect sizes precisely because the scaler ran first: every
        coefficient is "percentage points of Dg per standard deviation of this
        feature", so they can be compared across features in different units.
        """
        estimator = self.pipeline.named_steps["model"]
        raw = np.asarray(estimator.coef_)
        if raw.ndim == 1:
            frame = pd.DataFrame({"feature": self.features, "coefficient": raw})
        else:
            frame = pd.DataFrame(raw.T, index=self.features, columns=estimator.classes_)
            frame = frame.reset_index(names="feature").melt(
                id_vars="feature", var_name="class", value_name="coefficient"
            )
        frame["abs"] = frame["coefficient"].abs()
        return frame.sort_values("abs", ascending=False, ignore_index=True).drop(columns="abs")


def fit_ridge(
    train: pd.DataFrame,
    features: list[str],
    alpha: float,
) -> LearnedModel:
    """Fit L2-regularized linear regression on the training split."""
    pipeline = Pipeline([*_preprocessor(), ("model", Ridge(alpha=alpha, random_state=None))])
    pipeline.fit(train[features], train["target_dg"].astype("float64"))
    return LearnedModel(
        name="ridge",
        kind="magnitude",
        pipeline=pipeline,
        features=list(features),
        params={"alpha": alpha},
    )


def fit_logistic(
    train: pd.DataFrame,
    features: list[str],
    C: float,
) -> LearnedModel:
    """Fit multinomial logistic regression on the training split.

    ``class_weight`` is left at its default. Milestone 2 measured prevalence at
    36.6 / 31.2 / 32.2, which is close enough to balanced that reweighting would
    be tuning rather than correcting, and macro-F1 already declines to reward a
    model for riding the majority class.
    """
    pipeline = Pipeline(
        [
            *_preprocessor(),
            (
                "model",
                LogisticRegression(
                    C=C,
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(train[features], train["target_label"].astype("object"))
    return LearnedModel(
        name="multinomial_logistic",
        kind="direction",
        pipeline=pipeline,
        features=list(features),
        params={"C": C},
    )


@dataclass
class Selection:
    """The outcome of a hyperparameter scan, including the losers."""

    best: LearnedModel
    scores: pd.DataFrame
    metric: str

    def summary(self) -> str:
        name, value = self.best.name, self.scores[self.metric].dropna()
        chosen = self.best.params
        best_value = value.max() if self.metric == "macro_f1" else value.min()
        return (
            f"{name}: chose {chosen} on validation "
            f"{self.metric}={best_value:.4f} from {len(self.scores)} candidates"
        )


def select_ridge(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    alphas=RIDGE_ALPHAS,
) -> Selection:
    """Choose ``alpha`` by validation MAE — the plan's primary magnitude metric."""
    rows, fitted = [], {}
    for alpha in alphas:
        model = fit_ridge(train, features, alpha)
        scored = validation.assign(**model.predict(validation))
        metrics = magnitude_metrics(scored["target_dg"], scored["predicted_dg"])
        rows.append({"alpha": alpha, **metrics})
        fitted[alpha] = model

    scores = pd.DataFrame(rows)
    best_alpha = scores.loc[scores["mae"].idxmin(), "alpha"]
    return Selection(best=fitted[best_alpha], scores=scores, metric="mae")


def select_logistic(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    values=LOGISTIC_CS,
) -> Selection:
    """Choose ``C`` by validation macro-F1 — the primary directional metric."""
    rows, fitted = [], {}
    for C in values:
        model = fit_logistic(train, features, C)
        scored = validation.assign(**model.predict(validation))
        metrics = directional_metrics(scored["target_label"], scored["predicted_label"])
        rows.append({"C": C, **metrics})
        fitted[C] = model

    scores = pd.DataFrame(rows)
    best_c = scores.loc[scores["macro_f1"].idxmax(), "C"]
    return Selection(best=fitted[best_c], scores=scores, metric="macro_f1")


def refit_on(
    model: LearnedModel,
    frame: pd.DataFrame,
) -> LearnedModel:
    """Refit the chosen configuration on train and validation combined.

    Standard practice once hyperparameters are settled, and safe: the selection
    that produced those hyperparameters never saw the test split, and this
    function is only ever handed train + validation. It is kept separate from
    ``select`` so that the boundary is visible in the call site rather than
    buried in a flag.
    """
    if model.kind == "magnitude":
        return fit_ridge(frame, model.features, model.params["alpha"])
    return fit_logistic(frame, model.features, model.params["C"])
