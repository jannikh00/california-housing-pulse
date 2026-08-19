"""The measurements behind the Milestone 4 story, computed from saved predictions.

This module is the reason ``chp report`` exists as a separate command. Everything
here reads ``data/processed/predictions.parquet`` — the file the Milestone 3 run
already wrote — and recomputes metrics from it. Nothing refits a model, so the
README, its figures and its results table can be regenerated and restyled without
touching the frozen test split again.

That separation is a leakage control as much as a convenience: a presentation
layer that could refit is a presentation layer that could quietly tune against
the holdout while someone was "just fixing a chart".

The model names are pinned as constants rather than inferred from whichever row
scored best, for the same reason ``metrics.PRIMARY_MAGNITUDE`` is pinned: the
story names its models before it reads their numbers.
"""

from __future__ import annotations

import pandas as pd

from ..evaluation.metrics import by_group, evaluate, forward_growth_metrics
from ..io import read_parquet
from ..modeling.baselines import BASE_EFFECT, PRIMARY_MAGNITUDE_BASELINE
from ..modeling.split import TEST
from ..paths import PROCESSED_DIR

PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"

# The two models Milestone 3 selected: one per primary metric.
MAGNITUDE_MODEL = "ridge"
DIRECTIONAL_MODEL = "multinomial_logistic"

# Reading order for the baseline comparison: naive first, learned last, so the
# figure is read as "here is the bar, and here is what clearing it looks like".
MAGNITUDE_MODELS = (
    "persistence",
    "zero_change",
    "mean_reversion",
    PRIMARY_MAGNITUDE_BASELINE,
    MAGNITUDE_MODEL,
)
DIRECTIONAL_MODELS = (
    "majority_class",
    "persistence",
    "mean_reversion",
    PRIMARY_MAGNITUDE_BASELINE,
    MAGNITUDE_MODEL,
    DIRECTIONAL_MODEL,
)


def load_predictions(path=None) -> pd.DataFrame:
    """Read the saved predictions, or raise with the command that produces them."""
    path = path or PREDICTIONS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `chp baselines` before `chp report`: the report "
            "renders saved predictions and never refits a model."
        )
    return read_parquet(path)


def scored(predictions: pd.DataFrame, split: str = TEST) -> pd.DataFrame:
    return predictions.loc[predictions["split"] == split]


def headline(predictions: pd.DataFrame, split: str = TEST) -> pd.DataFrame:
    """One row per model: both primary metrics, skill against the naive bars.

    ``corr_forward`` travels with them because it is the number the Milestone 4
    story is actually about — see :func:`forward_skill`.
    """
    rows = []
    for name, part in scored(predictions, split).groupby("model", sort=False):
        rows.append({"model": name, **evaluate(part), **forward_growth_metrics(part)})
    table = pd.DataFrame(rows).set_index("model")

    base_mae = table.loc[PRIMARY_MAGNITUDE_BASELINE, "mae"]
    zero_mae = table.loc["zero_change", "mae"]
    majority_f1 = table.loc["majority_class", "macro_f1"]

    table["skill_vs_base_effect"] = 1.0 - table["mae"] / base_mae
    table["skill_vs_zero_change"] = 1.0 - table["mae"] / zero_mae
    table["macro_f1_vs_majority"] = table["macro_f1"] - majority_f1
    return table.reset_index()


def by_tier(predictions: pd.DataFrame, model: str, split: str = TEST) -> pd.DataFrame:
    """Metrics within volume tier, in thin-to-large order.

    Milestone 2 Decision 2 and Milestone 3 Decision 5: never pool these. The
    pooled macro-F1 of 0.569 describes no tier in the resulting table.
    """
    from ..viz import TIER_ORDER

    frame = scored(predictions, split)
    table = by_group(frame.loc[frame["model"] == model], "volume_tier")
    order = {tier: i for i, tier in enumerate(TIER_ORDER)}
    table["_order"] = table["volume_tier"].map(order)
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def by_county(predictions: pd.DataFrame, model: str, split: str = TEST) -> pd.DataFrame:
    """Per-county magnitude error, worst first, carrying the tier for colour.

    Sorted rather than alphabetical because the ordering *is* the finding: the
    ranking sorts itself almost perfectly by market thinness.
    """
    frame = scored(predictions, split)
    frame = frame.loc[frame["model"] == model].copy()
    frame["absolute_error"] = (frame["predicted_dg"] - frame["target_dg"]).abs()
    table = (
        frame.groupby(["county_fips", "county_name", "volume_tier"], observed=True)
        .agg(mae=("absolute_error", "mean"), n=("absolute_error", "size"))
        .reset_index()
        .sort_values("mae", ascending=False, ignore_index=True)
    )
    table["county"] = table["county_name"].str.replace(" County", "", regex=False)
    return table


def base_effect_correlation(predictions: pd.DataFrame, split: str = TEST) -> float:
    """``corr(Δg, −b)`` — how much of the target the base effect alone explains.

    The number the whole Milestone 4 story turns on, so it is computed rather
    than quoted: a later data vintage that changed it would otherwise leave the
    prose asserting a correlation the data no longer shows.
    """
    frame = scored(predictions, split)
    frame = frame.loc[frame["model"] == MAGNITUDE_MODEL]
    return float(frame["target_dg"].corr(-frame[BASE_EFFECT]))


def forward_skill(predictions: pd.DataFrame, model: str, split: str = TEST) -> pd.DataFrame:
    """Predicted and actual forward growth ``f(t)`` for one model.

    ``f = Δg + b``. Adding the observable base effect back to both sides moves the
    comparison out of the space where the model is credited for arithmetic, and
    into the space where only a real forecast can score.
    """
    frame = scored(predictions, split)
    frame = frame.loc[frame["model"] == model]
    usable = frame[[BASE_EFFECT, "target_dg", "predicted_dg"]].notna().all(axis=1)
    frame = frame.loc[usable]
    return pd.DataFrame(
        {
            "actual_f": frame["target_dg"] + frame[BASE_EFFECT],
            "predicted_f": frame["predicted_dg"] + frame[BASE_EFFECT],
            "volume_tier": frame["volume_tier"].to_numpy(),
        }
    )
