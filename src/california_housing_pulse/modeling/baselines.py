"""Naive baselines — the bar the learned models actually have to clear.

The plan is explicit that results must be compared against naive magnitude and
class baselines rather than against random guessing. That is not a formality
here. Milestone 2 measured a target standard deviation near 9.7 pp driven mostly
by market thinness, and against noise of that size "predict no change" is a
genuinely strong forecast. If ridge cannot beat it, the honest finding is that it
cannot beat it, and Milestone 4's definition of done explicitly allows for that.

Four naive predictors, each a different theory of what happens next:

``zero_change``
    Nothing changes: ``Dg = 0``. Under the frozen threshold this always predicts
    ``stable``, which makes it simultaneously the magnitude baseline and a
    degenerate directional one.

``persistence``
    Recent change continues: ``Dg(t) = growth_yoy(t) - growth_yoy(t-3)``. Note
    this is the most recent *fully observable* momentum change — nothing about it
    reaches past the forecast cutoff.

``mean_reversion``
    Momentum returns to its own recent norm: growth sitting far above a county's
    twelve-month average falls back. The strength ``k`` is fitted on training
    data, so this is the one baseline with a parameter, and the fitted value is
    reported because its sign is itself a finding.

``majority_class``
    Always answer with the most common training label. Direction only — it makes
    no magnitude claim. This is the bar macro-F1 must clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.target import TargetContract, load_contract

# Features the naive predictors read. They are ordinary columns of the feature
# matrix, so they inherit the same leakage guarantees as everything else.
MOMENTUM_CHANGE = "growth_yoy__diff3"
MOMENTUM_LEVEL = "growth_yoy__lag0"
MOMENTUM_NORM = "growth_yoy__rollmean12"


def classify(values: pd.Series, contract: TargetContract | None = None) -> pd.Series:
    """Map a predicted magnitude onto a directional class at the frozen tau.

    Deliberately the same rule the label itself uses, imported from the frozen
    contract rather than restated. A model is not allowed a different threshold
    from the target it is scored against.
    """
    from ..features.target import classify as classify_target

    return classify_target(values, contract or load_contract())


@dataclass
class Baseline:
    """Common shape for the naive predictors.

    ``fit`` exists on all of them even though most have nothing to learn, so the
    run harness can treat every model identically and so that the one baseline
    which *does* learn something cannot accidentally learn it from the test set.
    """

    name: str
    kind: str = "magnitude"
    params: dict[str, float] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> Baseline:
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _with_labels(self, predicted: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {"predicted_dg": predicted, "predicted_label": classify(predicted)},
            index=predicted.index,
        )


class ZeroChange(Baseline):
    """``Dg = 0`` — momentum is where it was."""

    def __init__(self) -> None:
        super().__init__(name="zero_change", kind="both")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._with_labels(pd.Series(0.0, index=frame.index))


class Persistence(Baseline):
    """``Dg(t) = growth_yoy(t) - growth_yoy(t-3)`` — the last observable change."""

    def __init__(self) -> None:
        super().__init__(name="persistence", kind="both")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        # A county with too little history has no recent change to extrapolate;
        # falling back to zero is the honest reading of "no information", and it
        # keeps the baseline defined on every row the models are scored on.
        return self._with_labels(frame[MOMENTUM_CHANGE].astype("float64").fillna(0.0))


class MeanReversion(Baseline):
    """``Dg = -k * (growth_yoy - its own 12-month mean)``, with ``k`` fit on train.

    The opposite bet to persistence, and worth making because Milestone 2 found
    the regime shifts to be the dominant feature of this series: the mean target
    moved +7.44 pp at 2023-02 and -6.90 pp at 2021-05. If growth reverts, ``k``
    comes out positive; if it extrapolates, ``k`` comes out negative and this
    degenerates into damped persistence on the level. Either way the fitted value
    is reported rather than assumed.
    """

    def __init__(self) -> None:
        super().__init__(name="mean_reversion", kind="both")

    def fit(self, train: pd.DataFrame) -> MeanReversion:
        gap = self._gap(train)
        actual = train["target_dg"].astype("float64")
        usable = gap.notna() & actual.notna()

        # Least squares through the origin: k = -sum(gap * y) / sum(gap^2).
        denominator = float((gap[usable] ** 2).sum())
        numerator = float((gap[usable] * actual[usable]).sum())
        self.params = {"k": -numerator / denominator if denominator else 0.0}
        return self

    @staticmethod
    def _gap(frame: pd.DataFrame) -> pd.Series:
        return frame[MOMENTUM_LEVEL].astype("float64") - frame[MOMENTUM_NORM].astype("float64")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        k = self.params.get("k", 0.0)
        return self._with_labels(-k * self._gap(frame).fillna(0.0))


class MajorityClass(Baseline):
    """Always predict the most common training label — the directional floor."""

    def __init__(self) -> None:
        super().__init__(name="majority_class", kind="direction")
        self.label: str | None = None
        self.shares: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> MajorityClass:
        counts = train["target_label"].value_counts()
        self.label = str(counts.idxmax())
        self.shares = train["target_label"].value_counts(normalize=True)
        self.params = {"share": float(counts.max() / counts.sum())}
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        from ..evaluation.metrics import CLASS_ORDER

        out = pd.DataFrame(
            {"predicted_label": pd.Series(self.label, index=frame.index, dtype="string")}
        )
        # Training prevalence is the honest probability statement for a model
        # that ignores its inputs, and it gives log loss and Brier something
        # meaningful to score rather than a degenerate one-hot vector.
        for label in CLASS_ORDER:
            share = float(self.shares.get(label, 0.0)) if self.shares is not None else 0.0
            out[f"prob_{label}"] = share
        return out


def naive_baselines() -> list[Baseline]:
    """Every naive predictor, in the order the results table reports them."""
    return [ZeroChange(), Persistence(), MeanReversion(), MajorityClass()]


def climatology(train: pd.DataFrame) -> float:
    """Mean training target — the constant a regression must beat to be useful.

    Not registered as a baseline of its own because it is within a whisker of
    zero_change on this data, but exposed so the evaluation report can state the
    number instead of implying zero_change was chosen arbitrarily.
    """
    value = train["target_dg"].astype("float64").mean()
    return float(value) if np.isfinite(value) else 0.0
