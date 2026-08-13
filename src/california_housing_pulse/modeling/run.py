"""The baseline run: fit everything, evaluate once, write the results down.

The order here encodes the definition of done, and each step is separated so the
boundary it protects is visible at the call site rather than hidden in a flag.

1. Assign the frozen split. Nothing downstream re-derives it.
2. Fit the naive baselines on training rows only. Three have nothing to learn;
   ``mean_reversion`` learns one scalar, and it learns it here.
3. Scan hyperparameters, scoring each candidate on **validation**. The test frame
   is not in scope in this step.
4. Freeze the chosen configurations and refit them on train + validation.
5. Only now, predict on the test split — once.
6. Ablate feature families, measured on validation, so the question "which
   families carry the signal?" costs no additional look at the test set.

Step 6 is deliberately after step 5 in the file and before it in importance: the
ablation is model *understanding*, and answering it on the test split would be
a second look at the holdout under a different name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from ..evaluation import bootstrap as bootstrap_mod
from ..evaluation.metrics import evaluate
from ..features.build import feature_names
from ..features.spec import load_feature_contract
from ..paths import MODELS_DIR, PROCESSED_DIR, relative
from .baselines import climatology, naive_baselines
from .models import (
    LearnedModel,
    Selection,
    fit_logistic,
    fit_ridge,
    refit_on,
    select_logistic,
    select_ridge,
)
from .split import (
    SPLIT_ORDER,
    TEST,
    TRAIN,
    VALIDATION,
    assign_split,
    load_split_contract,
    summarize_split,
)

PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"
MODEL_CONFIG_PATH = MODELS_DIR / "baseline_config.json"

# Carried onto every prediction row so the results can be sliced by tier and by
# period without re-joining anything.
CONTEXT = ["county_fips", "county_name", "reference_month", "volume_tier", "split"]
TRUTH = ["target_dg", "target_label"]


@dataclass
class RunResult:
    """Everything one baseline run produced."""

    predictions: pd.DataFrame
    split_summary: str
    selections: dict[str, Selection] = field(default_factory=dict)
    baseline_params: dict[str, dict] = field(default_factory=dict)
    ablation: pd.DataFrame | None = None
    intervals: pd.DataFrame | None = None
    coefficients: pd.DataFrame | None = None
    climatology_dg: float = 0.0
    feature_count: int = 0
    paths: dict[str, str] = field(default_factory=dict)

    def scored(self, split: str) -> pd.DataFrame:
        return self.predictions.loc[self.predictions["split"] == split]

    def table(self, split: str = TEST) -> pd.DataFrame:
        """Pooled metrics for every model on one split, primary metrics first."""
        rows = []
        for name, part in self.scored(split).groupby("model", sort=False):
            rows.append({"model": name, **evaluate(part)})
        table = pd.DataFrame(rows)
        return table.sort_values("mae", ignore_index=True) if "mae" in table else table


def _predictions_for(name: str, frame: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    out = frame[CONTEXT + TRUTH].copy()
    out.insert(0, "model", name)
    for column in predicted.columns:
        out[column] = predicted[column].to_numpy()
    return out


def run(
    features: pd.DataFrame,
    *,
    resamples: int = bootstrap_mod.DEFAULT_RESAMPLES,
    write: bool = True,
) -> RunResult:
    """Fit every baseline and model, evaluate, and persist the results."""
    contract = load_split_contract()
    columns = feature_names()

    frame = features.assign(split=assign_split(features))
    split_report = summarize_split(features, frame["split"])
    parts = {name: frame.loc[frame["split"] == name].copy() for name in SPLIT_ORDER}
    train, validation = parts[TRAIN], parts[VALIDATION]

    print(split_report.summary())
    print(f"  contract: {contract.describe()}")

    collected: list[pd.DataFrame] = []
    baseline_params: dict[str, dict] = {}

    print("\n[1/4] Naive baselines …")
    for baseline in naive_baselines():
        baseline.fit(train)
        baseline_params[baseline.name] = dict(baseline.params)
        for part in parts.values():
            collected.append(
                _predictions_for(baseline.name, part, baseline.predict(part).set_axis(part.index))
            )
        detail = ", ".join(f"{k}={v:.4g}" for k, v in baseline.params.items()) or "no parameters"
        print(f"  {baseline.name:<20} {detail}")

    print("\n[2/4] Selecting hyperparameters on validation …")
    selections = {
        "ridge": select_ridge(train, validation, columns),
        "multinomial_logistic": select_logistic(train, validation, columns),
    }
    for selection in selections.values():
        print(f"  {selection.summary()}")

    print("\n[3/4] Refitting the chosen configurations on train + validation …")
    fitted: dict[str, LearnedModel] = {}
    development = pd.concat([train, validation], ignore_index=True)
    for name, selection in selections.items():
        model = refit_on(selection.best, development)
        fitted[name] = model
        for part in parts.values():
            collected.append(_predictions_for(name, part, model.predict(part).set_axis(part.index)))

    predictions = pd.concat(collected, ignore_index=True)

    print("\n[4/4] Ablating feature families on validation …")
    ablation = _ablate(train, validation, columns, selections)
    for row in ablation.itertuples(index=False):
        print(f"  without {row.family:<10} mae {row.mae:+.4f}  macro_f1 {row.macro_f1:+.4f}")

    intervals = _intervals(predictions, resamples=resamples)
    coefficients = fitted["ridge"].coefficients().head(20)

    result = RunResult(
        predictions=predictions,
        split_summary=split_report.summary(),
        selections=selections,
        baseline_params=baseline_params,
        ablation=ablation,
        intervals=intervals,
        coefficients=coefficients,
        climatology_dg=climatology(train),
        feature_count=len(columns),
    )

    if write:
        result.paths = _persist(result, fitted, contract)
    return result


def _ablate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    selections: dict[str, Selection],
) -> pd.DataFrame:
    """Re-fit with each family removed; report the change in validation metrics.

    Reported as a *delta against the full model*, so a positive MAE entry means
    dropping the family made the error worse — that is, the family was carrying
    signal. Measured on validation, never on test.

    Hyperparameters are held at the values the full model selected rather than
    re-scanned per subset. That isolates the question being asked: re-tuning each
    ablated model would let a lucky alpha on one subset masquerade as evidence
    about a family, and the comparison would no longer be like for like.
    """
    contract = load_feature_contract()
    families: dict[str, list[str]] = {}
    for spec in contract.all_specs():
        families.setdefault(spec.family, []).append(spec.name)

    alpha = selections["ridge"].best.params["alpha"]
    C = selections["multinomial_logistic"].best.params["C"]

    full_ridge = fit_ridge(train, columns, alpha)
    full_logistic = fit_logistic(train, columns, C)
    base_mae = evaluate(validation.assign(**full_ridge.predict(validation)))["mae"]
    base_f1 = evaluate(validation.assign(**full_logistic.predict(validation)))["macro_f1"]

    rows = []
    for family, names in sorted(families.items()):
        kept = [column for column in columns if column not in set(names)]
        if not kept:
            continue
        ridge = fit_ridge(train, kept, alpha)
        logistic = fit_logistic(train, kept, C)
        mae = evaluate(validation.assign(**ridge.predict(validation)))["mae"]
        f1 = evaluate(validation.assign(**logistic.predict(validation)))["macro_f1"]
        rows.append(
            {
                "family": family,
                "features_removed": len(names),
                "mae": mae - base_mae,
                "macro_f1": f1 - base_f1,
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=False, ignore_index=True)


def _intervals(predictions: pd.DataFrame, *, resamples: int) -> pd.DataFrame:
    """Bootstrap intervals for every model's primary test metrics."""
    test = predictions.loc[predictions["split"] == TEST]
    frames = []
    for name, part in test.groupby("model", sort=False):
        table = bootstrap_mod.intervals_for(part, resamples=resamples)
        if len(table):
            table.insert(0, "model", name)
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _persist(result: RunResult, fitted: dict[str, LearnedModel], contract) -> dict[str, str]:
    """Write predictions and the reproducible model configuration."""
    from ..io import write_parquet

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {"predictions": relative(write_parquet(result.predictions, PREDICTIONS_PATH))}

    config = {
        "frozen": {"split": contract.frozen_date, "status": contract.frozen_status},
        "split": {
            name: {
                "start": str(contract.window(name).start or ""),
                "end": str(contract.window(name).end),
            }
            for name in SPLIT_ORDER
        },
        "embargo_months": contract.embargo_months,
        "feature_count": result.feature_count,
        "models": {
            **{name: {"params": params} for name, params in result.baseline_params.items()},
            **{
                name: {
                    "params": model.params,
                    "selected_on": "validation",
                    "refit_on": "train+validation",
                }
                for name, model in fitted.items()
            },
        },
        "hyperparameter_scans": {
            name: selection.scores.to_dict(orient="records")
            for name, selection in result.selections.items()
        },
    }
    MODEL_CONFIG_PATH.write_text(json.dumps(config, indent=2, default=str))
    paths["config"] = relative(MODEL_CONFIG_PATH)
    return paths
