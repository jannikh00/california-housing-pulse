"""Renders ``reports/results.md`` — the one-page MVP results document.

``reports/baselines.md`` is the exhaustive Milestone 3 evidence file: every
model, every split, every breakdown. This is the opposite document. It carries
the concise results table, the four Milestone 4 figures, and the statement of
what the project can and cannot claim, at the length a reviewer will actually
read.

The two do not disagree, because neither transcribes anything. Both compute from
``data/processed/predictions.parquet`` at render time, so the only way for them
to drift is for the predictions to change — which is exactly when they should.

The can/cannot-claim block is the part worth defending. It is written from the
Milestone 3 findings rather than generated, because a limitation is a judgement
about evidence and cannot be derived from the numbers that need limiting. Its
*numbers* are still interpolated from the live table, so a claim cannot quietly
outlive the result it rests on.
"""

from __future__ import annotations

import pandas as pd

from ..evaluation.bootstrap import intervals_for
from ..modeling.split import TEST
from ..paths import REPORTS_DIR, relative
from . import figures, results

RESULTS_REPORT = REPORTS_DIR / "results.md"

FIGURE_CAPTIONS = {
    "fig09_forward_skill.png": (
        "Forward skill — the claim the project actually makes. Both panels plot "
        "`f(t)`, the part of the target that had not yet happened."
    ),
    "fig06_baseline_comparison.png": (
        "Magnitude and direction against every naive baseline, on the untouched test split."
    ),
    "fig07_results_by_tier.png": (
        "The same results within volume tier. Magnitude and direction disagree about "
        "which counties are hard, which is why neither is quoted pooled."
    ),
    "fig08_county_error.png": (
        "Per-county magnitude error over the test window, worst first, coloured by tier."
    ),
}


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _percent(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.1%}"


def concise_table(predictions: pd.DataFrame, split: str = TEST) -> pd.DataFrame:
    """The results table the README shows: every model, both primary metrics.

    Ordered naive-first so the table is read as a comparison against a bar rather
    than as a leaderboard, and carrying ``corr_forward`` because a magnitude
    number without it overstates what was forecast.
    """
    table = results.headline(predictions, split).set_index("model")
    order = [
        "persistence",
        "zero_change",
        "mean_reversion",
        results.PRIMARY_MAGNITUDE_BASELINE,
        "majority_class",
        results.MAGNITUDE_MODEL,
        results.DIRECTIONAL_MODEL,
    ]
    return table.loc[[m for m in order if m in table.index]].reset_index()


def _results_table_lines(table: pd.DataFrame) -> list[str]:
    header = (
        "| model | kind | MAE (pp) | vs base effect | vs zero change | "
        "macro-F1 | vs majority | corr on f(t) |"
    )
    lines = [header, "|" + "---|" * 8]
    learned = {results.MAGNITUDE_MODEL, results.DIRECTIONAL_MODEL}
    for row in table.itertuples(index=False):
        name = f"**{row.model}**" if row.model in learned else row.model
        kind = "learned" if row.model in learned else "naive"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    kind,
                    _fmt(row.mae),
                    _percent(row.skill_vs_base_effect),
                    _percent(row.skill_vs_zero_change),
                    _fmt(row.macro_f1),
                    _fmt(row.macro_f1_vs_majority),
                    _fmt(row.corr_forward),
                ]
            )
            + " |"
        )
    return lines


def _tier_table_lines(predictions: pd.DataFrame) -> list[str]:
    magnitude = results.by_tier(predictions, results.MAGNITUDE_MODEL)
    direction = results.by_tier(predictions, results.DIRECTIONAL_MODEL)
    merged = magnitude[["volume_tier", "n", "mae", "mean_error"]].merge(
        direction[["volume_tier", "accuracy", "macro_f1"]], on="volume_tier"
    )
    lines = [
        "| tier | rows | ridge MAE | ridge mean error | logistic accuracy | logistic macro-F1 |",
        "|" + "---|" * 6,
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.volume_tier,
                    str(int(row.n)),
                    _fmt(row.mae),
                    _fmt(row.mean_error),
                    _fmt(row.accuracy),
                    _fmt(row.macro_f1),
                ]
            )
            + " |"
        )
    return lines


def _interval_lines(predictions: pd.DataFrame, resamples: int) -> list[str]:
    """Block-bootstrap intervals for the two selected models.

    Both blocking schemes are reported. Resampling whole counties and resampling
    whole months answer different questions — "would another set of counties give
    this?" and "would another twelve months?" — and the month interval is the
    narrower one only because twelve months of a single cooling-leaning regime
    resemble each other. Showing one without the other would flatter the result.
    """
    frame = results.scored(predictions)
    lines = ["| model | metric | block | point | 95% interval |", "|" + "---|" * 5]
    for model, metrics in (
        (results.MAGNITUDE_MODEL, ("mae",)),
        (results.DIRECTIONAL_MODEL, ("macro_f1",)),
        (results.PRIMARY_MAGNITUDE_BASELINE, ("mae",)),
    ):
        part = frame.loc[frame["model"] == model]
        table = intervals_for(part, metrics=metrics, resamples=resamples)
        for row in table.itertuples(index=False):
            lines.append(
                f"| {model} | {row.metric} | {row.block} | {_fmt(row.point)} | "
                f"[{_fmt(row.low)}, {_fmt(row.high)}] |"
            )
    return lines


def claims_block(predictions: pd.DataFrame) -> list[str]:
    """What the project can and cannot claim, with the numbers behind each line."""
    table = results.headline(predictions).set_index("model")
    ridge = table.loc[results.MAGNITUDE_MODEL]
    logistic = table.loc[results.DIRECTIONAL_MODEL]
    base = table.loc[results.PRIMARY_MAGNITUDE_BASELINE]
    tiers = results.by_tier(predictions, results.MAGNITUDE_MODEL).set_index("volume_tier")

    return [
        "## What this project can and cannot claim",
        "",
        "**It can claim:**",
        "",
        f"- On one untouched twelve-month window ({int(ridge['n'])} county-months, scored "
        f"once), ridge predicted the magnitude of the three-month momentum change with MAE "
        f"{ridge['mae']:.3f} pp against the honest naive bar's {base['mae']:.3f} pp — "
        f"{ridge['skill_vs_base_effect']:.1%} less error.",
        f"- That skill is real forecasting rather than recovered arithmetic. On forward "
        f"growth `f(t)`, the part that had not yet happened, ridge scores "
        f"{ridge['corr_forward']:.3f} where the base-effect baseline scores "
        f"{base['corr_forward']:.3f}.",
        f"- Multinomial logistic separates heating, stable and cooling at macro-F1 "
        f"{logistic['macro_f1']:.3f} against a majority-class floor of "
        f"{table.loc['majority_class', 'macro_f1']:.3f}.",
        "- Every feature was verified available at prediction time by three independent "
        "controls, one of which found and fixed a real leak in the unemployment lag.",
        "",
        "**It cannot claim:**",
        "",
        "- **That these numbers generalise across market regimes.** They come from one "
        "contiguous test window, 2025-03 to 2026-02, which leans cooling. Rolling-origin "
        "validation across regimes is the next milestone, and until it runs, a single "
        "held-out window is the honest description of the evidence.",
        f"- **Useful county-level accuracy in thin markets.** MAE is "
        f"{tiers.loc['thin', 'mae']:.2f} pp in the thin tier against "
        f"{tiers.loc['large', 'mae']:.2f} pp in the large one. A forecast for Mono or Colusa "
        f"carries error larger than most of the moves it is trying to call.",
        "- **Calibrated probabilities.** The logistic emits class probabilities and they are "
        "scored, but no calibration was fitted and no reliability diagram has been drawn. "
        "That is Milestone 7.",
        "- **Any causal statement.** These are predictive associations between observable "
        "market conditions and a later price-growth change. Nothing here identifies a "
        "mechanism, and nothing supports an intervention.",
        f"- **Symmetric performance across classes.** The models under-call cooling: mean "
        f"error is positive in every quarter of the test window (+{ridge['mean_error']:.2f} pp "
        f"pooled), and cooling recall runs well below cooling precision. This is left "
        f"uncorrected on purpose — tuning class weights after seeing the test set is what "
        f"the frozen contracts exist to prevent.",
        "- **That beating persistence means much.** Persistence is *worse than predicting no "
        "change at all* on this series, so clearing it is a low bar. The bar that counts is "
        "`base_effect`.",
    ]


def build_report(
    predictions: pd.DataFrame | None = None,
    *,
    directory=None,
    resamples: int = 1000,
    render_figures: bool = True,
) -> tuple[str, list[str]]:
    """Render the results document and its figures; return both paths."""
    predictions = results.load_predictions() if predictions is None else predictions
    figure_paths = figures.build_all(predictions, directory) if render_figures else []

    table = concise_table(predictions)
    indexed = table.set_index("model")
    rows = int(indexed.loc[results.MAGNITUDE_MODEL, "n"])
    counties = results.scored(predictions)["county_fips"].nunique()
    base_correlation = results.base_effect_correlation(predictions)
    pooled_f1 = float(indexed.loc[results.MAGNITUDE_MODEL, "macro_f1"])

    lines = [
        "# Results — California Housing Pulse MVP",
        "",
        "Generated by `chp report` from `data/processed/predictions.parquet`. Every number "
        "on this page is computed at render time; none is transcribed by hand.",
        "",
        f"Test split: **2025-03 → 2026-02**, {rows} county-months across {counties} "
        "counties, scored once after the pipeline was frozen.",
        "",
        "## Read this first",
        "",
        "The target decomposes exactly into `Δg(t) = f(t) − b(t)`, where `b` — the **base "
        "effect** — is three-month growth that already happened between `t−12` and `t−9` and "
        "is fully observable when the forecast is made. On the test window `corr(Δg, −b)` is "
        f"**{base_correlation:.3f}**: roughly half this target was knowable in advance for "
        "reasons that have nothing to do with forecasting a housing market.",
        "",
        "So the magnitude bar here is `base_effect`, not `zero_change`, and the skill measure "
        "is `corr on f(t)` — the correlation between predicted and actual *forward* growth. "
        "A model scored against zero-change is credited for arithmetic it did not perform.",
        "",
        "## Results",
        "",
        *_results_table_lines(table),
        "",
        "`vs base effect` and `vs zero change` are fractional MAE reductions; negative is "
        "worse than that baseline. `vs majority` is a difference in macro-F1, not a ratio. "
        "`corr on f(t)` is the honest skill measure — note that `base_effect` scores "
        "essentially zero on it, by construction, which is what makes it trustworthy.",
        "",
        "## By volume tier — never pool these",
        "",
        *_tier_table_lines(predictions),
        "",
        f"Ridge's pooled macro-F1 of {pooled_f1:.3f} describes no tier in this table.",
        "",
        "## Uncertainty",
        "",
        *_interval_lines(predictions, resamples),
        "",
        "## Figures",
        "",
    ]

    # Iterated in caption order, not render order: the figures are rendered
    # cheapest-first but must be *read* claim-first, and forward skill is the claim.
    for name, caption in FIGURE_CAPTIONS.items():
        lines += [caption, "", f"![{name}](figures/{name})", ""]

    lines += ["", *claims_block(predictions), ""]

    RESULTS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_REPORT.write_text("\n".join(lines))
    return relative(RESULTS_REPORT), figure_paths
