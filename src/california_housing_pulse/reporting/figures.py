"""The four Milestone 4 results figures.

Milestone 2's figures argued about the *data*; these argue about the *results*,
and each one exists to carry a claim the README makes in words:

* :func:`baseline_comparison` — the models clear the honest naive bar, and
  persistence does not clear doing nothing.
* :func:`results_by_tier` — magnitude and direction disagree about which counties
  are hard, so neither may be quoted pooled.
* :func:`county_error` — where the magnitude error actually lives.
* :func:`forward_skill` — the milestone's real claim: skill on the part of the
  target that had not happened yet.

Palette and chrome come from :mod:`california_housing_pulse.viz`, shared with the
Milestone 2 figures, because the README shows both sets on one page.

**Emphasis.** Learned models take the strong series hue; naive baselines take the
muted grey. That is a deliberate reading order rather than decoration — every one
of these figures is a comparison against a bar, and the bar should recede.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import FIGURES_DIR
from ..viz import (
    AXIS,
    COOLING,
    GRID,
    HEATING,
    INK,
    INK_SECONDARY,
    MUTED,
    SERIES,
    TIER_COLORS,
    plt,
)
from ..viz import figure as _figure
from ..viz import figure_header as _figure_header
from ..viz import panel_title as _panel_title
from ..viz import panels as _panels
from ..viz import save as _save
from ..viz import style_axes as _style
from ..viz import tier_legend as _tier_legend
from ..viz import title as _title
from . import results

LEARNED = SERIES[0]
NAIVE = "#b9b8b0"

# Printed under a panel whose metric runs the opposite way to the one beside it.
LOWER_BETTER = "lower is better"
HIGHER_BETTER = "higher is better"


def _bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    *,
    highlight: tuple[str, ...],
    reference: str,
    digits: int = 3,
) -> None:
    """Horizontal bars with values printed at the bar end.

    The value labels are not redundant with the axis: these figures are read at
    README width, where a reader cannot resolve 3.823 from 4.699 by bar length.
    """
    positions = np.arange(len(frame))
    colors = [LEARNED if m in highlight else NAIVE for m in frame["model"]]
    ax.barh(positions, frame[value], color=colors, height=0.68, zorder=2)

    reference_value = float(frame.loc[frame["model"] == reference, value].iloc[0])
    ax.axvline(reference_value, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)

    span = float(frame[value].max())
    for position, (model, magnitude) in enumerate(zip(frame["model"], frame[value], strict=True)):
        ax.text(
            magnitude + span * 0.02,
            position,
            f"{magnitude:.{digits}f}",
            va="center",
            fontsize=9,
            color=INK if model in highlight else INK_SECONDARY,
            fontweight="bold" if model in highlight else "normal",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([m.replace("_", " ") for m in frame["model"]], fontsize=9.5, color=INK)
    ax.set_xlim(0, span * 1.16)
    ax.invert_yaxis()


def baseline_comparison(predictions: pd.DataFrame, directory: Path) -> str:
    """Claim: the models clear the honest bar, and persistence is worse than nothing.

    Two panels rather than two files. The plan asks for a magnitude *and*
    direction comparison, and splitting them across figures would let a reader
    take one without the other — which is exactly the misreading Milestone 3
    Decision 5 warns about.
    """
    table = results.headline(predictions).set_index("model")

    magnitude = table.loc[list(results.MAGNITUDE_MODELS), ["mae"]].reset_index().sort_values("mae")
    direction = (
        table.loc[list(results.DIRECTIONAL_MODELS), ["macro_f1"]]
        .reset_index()
        .sort_values("macro_f1", ascending=False)
    )

    fig, (left, right) = _panels(12.4, 4.4)
    _bars(
        left,
        magnitude,
        "mae",
        highlight=(results.MAGNITUDE_MODEL,),
        reference=results.PRIMARY_MAGNITUDE_BASELINE,
    )
    _bars(
        right,
        direction,
        "macro_f1",
        highlight=(results.MAGNITUDE_MODEL, results.DIRECTIONAL_MODEL),
        reference="majority_class",
    )

    base_mae = float(table.loc[results.PRIMARY_MAGNITUDE_BASELINE, "mae"])
    ridge_skill = float(table.loc[results.MAGNITUDE_MODEL, "skill_vs_base_effect"])
    logistic_f1 = float(table.loc[results.DIRECTIONAL_MODEL, "macro_f1"])
    majority_f1 = float(table.loc["majority_class", "macro_f1"])
    persistence_penalty = -float(table.loc["persistence", "skill_vs_zero_change"])

    _panel_title(left, f"Magnitude — MAE in percentage points · {LOWER_BETTER}")
    _panel_title(right, f"Direction — macro-F1 across three classes · {HIGHER_BETTER}")
    for ax in (left, right):
        _style(ax, xgrid=True)
        ax.grid(axis="y", visible=False)

    # Anchored above the top bar rather than below the bottom one: the axes are
    # inverted, so the bottom is where the longest bar and its value label sit.
    left.annotate(
        "base effect — the honest bar",
        xy=(base_mae, -0.72),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
        va="center",
        annotation_clip=False,
    )
    right.annotate(
        "majority class",
        xy=(float(table.loc["majority_class", "macro_f1"]), -0.72),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
        va="center",
        annotation_clip=False,
    )
    for ax, frame in ((left, magnitude), (right, direction)):
        ax.set_ylim(len(frame) - 0.4, -1.0)

    rows = int(table.loc[results.MAGNITUDE_MODEL, "n"])
    _figure_header(
        fig,
        "Both models clear the naive bar — and persistence is worse than doing nothing",
        f"Test split, {rows} county-months, scored once. Ridge removes {ridge_skill:.1%} of "
        f"the base-effect baseline's error and the logistic reaches macro-F1 "
        f"{logistic_f1:.3f} against a majority-class floor of {majority_f1:.3f}. "
        f"Persistence — the conventional naive forecast — is {persistence_penalty:.0%} worse "
        f"than predicting no change at all.",
    )
    return _save(fig, "fig06_baseline_comparison.png", directory)


def results_by_tier(predictions: pd.DataFrame, directory: Path) -> str:
    """Claim: magnitude and direction disagree about which counties are hard.

    The single most important reason this project never quotes one pooled score.
    Thin counties are four times worse on magnitude and *better* on direction,
    so any headline number describes a tier that does not exist.
    """
    magnitude = results.by_tier(predictions, results.MAGNITUDE_MODEL)
    direction = results.by_tier(predictions, results.DIRECTIONAL_MODEL)

    fig, (left, right) = _panels(12.4, 4.3)
    for ax, table, column, digits in (
        (left, magnitude, "mae", 2),
        (right, direction, "macro_f1", 3),
    ):
        positions = np.arange(len(table))
        colors = [TIER_COLORS[t] for t in table["volume_tier"]]
        ax.bar(positions, table[column], color=colors, width=0.62, zorder=2)
        for position, value in zip(positions, table[column], strict=True):
            ax.text(
                position,
                value + table[column].max() * 0.035,
                f"{value:.{digits}f}",
                ha="center",
                fontsize=9.5,
                color=INK,
                fontweight="bold",
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{t}\n{int(n)} rows" for t, n in zip(table["volume_tier"], table["n"], strict=True)],
            fontsize=9.5,
            color=INK,
        )
        ax.set_ylim(0, table[column].max() * 1.22)
        _style(ax)

    _panel_title(left, f"Magnitude — ridge MAE, percentage points · {LOWER_BETTER}")
    _panel_title(right, f"Direction — logistic macro-F1 · {HIGHER_BETTER}")

    def _at(table, tier, column):
        return float(table.loc[table["volume_tier"] == tier, column].iloc[0])

    thin_mae, large_mae = _at(magnitude, "thin", "mae"), _at(magnitude, "large", "mae")
    thin_f1, large_f1 = _at(direction, "thin", "macro_f1"), _at(direction, "large", "macro_f1")
    _figure_header(
        fig,
        "The hardest counties for magnitude are not the hardest for direction",
        f"Test split by volume tier, thin to large. Magnitude error runs "
        f"{thin_mae / large_mae:.1f}× worse in thin counties — and direction runs the other "
        f"way: large counties post the best MAE ({large_mae:.2f}) and the worst macro-F1 "
        f"({large_f1:.3f}), while thin counties manage {thin_f1:.3f}. The ±2 pp band is fixed "
        f"while noise scales with thinness, so large counties sit in the stable class about "
        f"half the time and the model must separate three genuinely close classes. No single "
        f"pooled score describes any of these four tiers.",
    )
    return _save(fig, "fig07_results_by_tier.png", directory)


def county_error(predictions: pd.DataFrame, directory: Path) -> str:
    """Claim: the error is concentrated, and it is concentrated by market thinness.

    A ranking rather than a map. A choropleth would shade by area, and California's
    largest counties by area are its emptiest by transaction volume — the reader
    would see Modoc and Inyo dominate a figure whose entire point is that error
    tracks *sales*, not land.
    """
    table = results.by_county(predictions, results.MAGNITUDE_MODEL)
    pooled = results.headline(predictions).set_index("model").loc[results.MAGNITUDE_MODEL, "mae"]

    fig, ax = _figure(9.6, 0.208 * len(table) + 2.3)
    positions = np.arange(len(table))
    ax.barh(
        positions,
        table["mae"],
        color=[TIER_COLORS[t] for t in table["volume_tier"]],
        height=0.74,
        zorder=2,
    )
    ax.axvline(float(pooled), color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.annotate(
        f"pooled MAE {pooled:.2f}",
        xy=(float(pooled), -1.1),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
        va="center",
        annotation_clip=False,
    )

    for position, value in zip(positions, table["mae"], strict=True):
        ax.text(
            value + table["mae"].max() * 0.012,
            position,
            f"{value:.1f}",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(table["county"], fontsize=8.5, color=INK)
    ax.set_ylim(len(table) - 0.4, -1.6)
    ax.set_xlim(0, float(table["mae"].max()) * 1.08)
    ax.set_xlabel(
        "mean absolute error over the test window (pp)", fontsize=9.5, color=INK_SECONDARY
    )
    _style(ax, xgrid=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    worst, best = table.iloc[0], table.iloc[-1]
    _title(
        ax,
        "Error is a property of the county, not of the month",
        f"Ridge, test split, one bar per modelled county over twelve months, worst first. "
        f"{worst['county']} at {worst['mae']:.1f} pp against {best['county']} at "
        f"{best['mae']:.1f} pp — a {worst['mae'] / best['mae']:.0f}× spread that the pooled "
        f"figure hides. Colour is volume tier, and the ranking sorts itself almost entirely "
        f"by it.",
        wrap=104,
    )
    _tier_legend(ax, loc="lower right", bbox_to_anchor=(1.0, 0.005), ncol=2)
    return _save(fig, "fig08_county_error.png", directory)


def forward_skill(predictions: pd.DataFrame, directory: Path) -> str:
    """Claim: the real forecasting skill, measured where arithmetic cannot help.

    The most important figure in the milestone. Both panels plot forward growth
    ``f(t) = Δg(t) + b(t)`` — the part of the target that had genuinely not
    happened when the forecast was made. The base-effect baseline scores 0.707
    against the *raw* target and near zero here, which is what makes ridge's
    score in the same space believable rather than merely large.
    """
    table = results.headline(predictions).set_index("model")
    fig, axes = _panels(11.6, 5.2, sharex=True, sharey=True)

    models = (results.PRIMARY_MAGNITUDE_BASELINE, results.MAGNITUDE_MODEL)
    frames = [results.forward_skill(predictions, model) for model in models]
    combined = pd.concat(frames)
    low = float(min(combined["actual_f"].min(), combined["predicted_f"].min()))
    high = float(max(combined["actual_f"].max(), combined["predicted_f"].max()))
    pad = (high - low) * 0.04

    for ax, model, frame in zip(axes, models, frames, strict=True):
        is_learned = model == results.MAGNITUDE_MODEL
        ax.plot([low, high], [low, high], color=AXIS, linewidth=1.0, zorder=1)
        ax.scatter(
            frame["actual_f"],
            frame["predicted_f"],
            s=15,
            color=LEARNED if is_learned else MUTED,
            alpha=0.55,
            linewidth=0,
            zorder=3,
        )
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("actual forward growth f(t), pp", fontsize=9.5, color=INK_SECONDARY)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)

        forward = float(table.loc[model, "corr_forward"])
        raw = float(table.loc[model, "corr_dg"])
        _panel_title(ax, model.replace("_", " "))
        ax.text(
            0.04,
            0.95,
            f"corr on f(t)   {forward:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            color=HEATING if abs(forward) < 0.1 else COOLING,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.04,
            0.885,
            f"corr on Δg     {raw:.3f}",
            transform=ax.transAxes,
            fontsize=9.5,
            color=INK_SECONDARY,
            va="top",
        )

    axes[0].set_ylabel("predicted forward growth, pp", fontsize=9.5, color=INK_SECONDARY)
    base_correlation = results.base_effect_correlation(predictions)
    ridge_forward = float(table.loc[results.MAGNITUDE_MODEL, "corr_forward"])
    _figure_header(
        fig,
        "Half the target was knowable by arithmetic. This is the half that was not.",
        f"Both panels plot forward growth f(t) = Δg(t) + b(t), where b is the base effect — "
        f"three-month growth that had already happened between t−12 and t−9 and is fully "
        f"observable at forecast time. The base effect alone correlates "
        f"{base_correlation:.3f} with the raw target while knowing nothing about the future, "
        f"and its score collapses here. Ridge holds {ridge_forward:.3f}. That gap is the "
        f"forecasting claim this project makes.",
    )
    return _save(fig, "fig09_forward_skill.png", directory)


def build_all(predictions: pd.DataFrame, directory: Path | None = None) -> list[str]:
    """Render every Milestone 4 figure and return their project-relative paths."""
    directory = Path(directory) if directory else FIGURES_DIR
    return [
        baseline_comparison(predictions, directory),
        results_by_tier(predictions, directory),
        county_error(predictions, directory),
        forward_skill(predictions, directory),
    ]
