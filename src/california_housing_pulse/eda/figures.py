"""The five Milestone 2 figures.

Each figure exists to settle a decision, not to decorate the report; the module
docstring for each function names the decision it informs.

**Colour.** Two palettes, each chosen by the job the colour does:

*Polarity* (cooling / stable / heating) uses a diverging scale — blue and red
poles either side of a neutral grey. Grey is deliberate: a diverging midpoint
must read as "nothing happening", which no chromatic hue does. The pair was
checked for colour-vision deficiency separation (worst adjacent ΔE 8.7, against a
target of 8) and every step clears 3:1 against the surface.

*Identity* (one line per county) uses the first three categorical slots — blue,
orange, aqua — which clear the all-pairs CVD and normal-vision floors. Aqua sits
below 3:1 on this surface, so those series are directly labelled rather than
relying on the legend alone.

Class shares are printed onto the stacked segments and lines are labelled at
their right edge, so identity and magnitude never depend on colour by itself.
Figures render light-only: they are committed artifacts in a repository, not a
themed web page.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in CI or a plain terminal.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..features.target import TargetContract, load_contract  # noqa: E402
from ..paths import FIGURES_DIR, relative  # noqa: E402
from . import analysis  # noqa: E402

# Diverging: polarity of the market move.
COOLING = "#2a78d6"
STABLE = "#898781"
HEATING = "#e34948"
CLASS_COLORS = {"cooling": COOLING, "stable": STABLE, "heating": HEATING}

# Categorical: identity of a county.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

DPI = 160


def _style(ax: plt.Axes, *, xgrid: bool = False) -> None:
    """Recessive chrome: hairline grid, no box, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _figure(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _title(ax: plt.Axes, title: str, subtitle: str, *, wrap: int = 88) -> None:
    """Left-aligned title with a wrapped subtitle.

    The subtitle is wrapped rather than left to run long: ``bbox_inches="tight"``
    sizes the saved canvas around every artist, so a subtitle wider than the axes
    silently stretches the image and squashes the plot.
    """
    wrapped = textwrap.fill(subtitle, wrap)
    lines = wrapped.count("\n") + 1
    # Both are positioned in points above the axes rather than in axes fractions,
    # so the gap does not change with figure height and the title clears a
    # subtitle of any line count.
    subtitle_offset = 8.0
    line_height = 13.3
    ax.set_title(
        title,
        loc="left",
        fontsize=13,
        color=INK,
        fontweight="bold",
        pad=subtitle_offset + lines * line_height + 6,
    )
    ax.annotate(
        wrapped,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, subtitle_offset),
        textcoords="offset points",
        fontsize=9.5,
        color=INK_SECONDARY,
        va="bottom",
        linespacing=1.4,
    )


def _place_labels(ax: plt.Axes, entries: list[tuple[float, str, str]], x: float) -> None:
    """Annotate series at a common x, nudged apart so labels never overlap.

    Direct labels are the accessibility mechanism for these charts, so two of
    them landing on top of each other is a correctness problem, not a cosmetic one.
    """
    low, high = ax.get_ylim()
    gap = (high - low) * 0.055
    ordered = sorted(entries, key=lambda item: item[0])
    placed: list[float] = []
    for value, _, _ in ordered:
        candidate = value if not placed else max(value, placed[-1] + gap)
        placed.append(candidate)
    # Re-centre the block so the nudging does not drift everything upward.
    drift = (placed[-1] + placed[0]) / 2 - (ordered[-1][0] + ordered[0][0]) / 2
    for position, (_, text, color) in zip(placed, ordered, strict=True):
        ax.annotate(
            text,
            xy=(x, position - drift),
            fontsize=9,
            color=color,
            va="center",
            fontweight="bold",
            annotation_clip=False,
        )


def _save(fig: plt.Figure, name: str, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return relative(path)


def target_distribution(
    model: pd.DataFrame, directory: Path, contract: TargetContract | None = None
) -> str:
    """Decision: is the magnitude target dominated by artifacts or extremes?

    A histogram rather than a density: the reader needs to see where the frozen
    thresholds cut the distribution, and how much mass sits in each class.
    """
    contract = contract or load_contract()
    values = model["target_dg"].dropna()
    # Show the bulk, and *omit* the tails rather than clipping them. Clipping
    # piles every extreme value into the two end bins, which draws two tall bars
    # that a reader will reasonably mistake for real modes. The omitted count is
    # stated in the subtitle instead.
    limit = 30.0
    inside = values[values.abs() <= limit]
    omitted = len(values) - len(inside)

    fig, ax = _figure(9, 4.6)
    edges = np.linspace(-limit, limit, 61)
    counts, _ = np.histogram(inside, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [
        COOLING if c <= -contract.tau else HEATING if c >= contract.tau else STABLE for c in centers
    ]
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.88, color=colors, zorder=2)

    for edge in (-contract.tau, contract.tau):
        # Stop the rules below the class labels so they never strike through the text.
        ax.axvline(edge, ymax=0.80, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)

    shares = analysis.prevalence_overall(model, contract)
    ax.set_ylim(0, counts.max() * 1.32)
    for label, x, color in (
        ("cooling", -limit * 0.62, COOLING),
        ("stable", 0.0, STABLE),
        ("heating", limit * 0.62, HEATING),
    ):
        ax.text(
            x,
            counts.max() * 1.20,
            f"{label}\n{shares[label]:.1%}",
            ha="center",
            va="center",
            fontsize=10.5,
            color=color,
            fontweight="bold",
        )

    _title(
        ax,
        "The magnitude target is centred and roughly symmetric",
        f"target_dg over {len(values):,} modelling rows; dashed lines mark the frozen "
        f"±{contract.tau:g} pp threshold. {omitted} rows ({omitted / len(values):.1%}) "
        f"beyond ±{limit:g} pp are omitted from the plot, not clipped into the end bins.",
    )
    ax.set_xlabel(
        "Δg — three-month change in year-over-year price growth (pp)",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    ax.set_ylabel("county-months", fontsize=9.5, color=INK_SECONDARY)
    _style(ax)
    return _save(fig, "fig01_target_distribution.png", directory)


def prevalence_by_year(
    model: pd.DataFrame, directory: Path, contract: TargetContract | None = None
) -> str:
    """Decision: is the class mix stable enough to train one model across years?

    Stacked to 100% because the question is about *mix*, not volume; the row
    count per year is near-constant anyway and is reported in the memo.
    """
    contract = contract or load_contract()
    table = analysis.prevalence_by_year(model, contract)
    years = table.index.astype(str)

    fig, ax = _figure(9.5, 4.6)
    bottom = np.zeros(len(table))
    for label in contract.label_names:
        values = table[label].to_numpy() * 100
        ax.bar(
            years,
            values,
            bottom=bottom,
            color=CLASS_COLORS[label],
            width=0.72,
            zorder=2,
            # A 2px surface gap between stacked segments, per the mark spec.
            edgecolor=SURFACE,
            linewidth=2,
        )
        for x, (value, base) in enumerate(zip(values, bottom, strict=True)):
            if value >= 12:  # Only label segments with room for the text.
                ax.text(
                    x,
                    base + value / 2,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#ffffff",
                    fontweight="bold",
                )
        bottom += values

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    _title(
        ax,
        "Class mix swings hard with the market regime",
        "Share of modelling rows by directional class. 2022 is 70% cooling; "
        "2023 is 55% heating — the same model must handle both.",
    )
    _style(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[label]) for label in contract.label_names
    ]
    ax.legend(
        handles,
        list(contract.label_names),
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=9.5,
        labelcolor=INK_SECONDARY,
    )
    return _save(fig, "fig02_class_prevalence_by_year.png", directory)


def county_time_series(panel: pd.DataFrame, model: pd.DataFrame, directory: Path) -> str:
    """Decision: does the target behave the same way in small and large counties?

    Plots growth rather than the target itself: the reader needs to see the
    series the target is *built from* to judge whether its swings are market
    movement or thin-market noise.
    """
    chosen = analysis.representative_counties(model)
    # One county per tier, largest first, capped at three series.
    wanted = [t for t in ("large", "mid", "thin") if t in chosen][:3]

    fig, ax = _figure(10, 4.8)
    labels: list[tuple[float, str, str]] = []
    for color, tier in zip(SERIES, wanted, strict=False):
        fips, name = chosen[tier]
        series = panel.loc[panel["county_fips"] == fips].sort_values("reference_month")
        ax.plot(
            series["reference_month"],
            series["growth_yoy"],
            color=color,
            linewidth=2.0,
            zorder=3,
            solid_capstyle="round",
        )
        last = series.loc[series["growth_yoy"].notna()].iloc[-1]
        volume = series["homes_sold_median"].iloc[0]
        labels.append(
            (float(last["growth_yoy"]), f"{name.replace(' County', '')} · {volume:,.0f}/mo", color)
        )

    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    _title(
        ax,
        "Small counties swing far harder than large ones",
        "Year-over-year growth of the three-month smoothed median sale price, for the "
        "median county of each volume tier.",
    )
    ax.set_ylabel("year-over-year growth (pp)", fontsize=9.5, color=INK_SECONDARY)
    last_month = panel["reference_month"].max()
    ax.set_xlim(panel["reference_month"].min(), last_month + pd.DateOffset(months=1))
    _style(ax)
    # Direct labels sit just outside the axes, so no plot width is spent on
    # empty space. Identity never rests on colour alone.
    _place_labels(ax, labels, x=last_month + pd.DateOffset(months=3))
    return _save(fig, "fig03_county_time_series.png", directory)


def volume_versus_dispersion(
    panel: pd.DataFrame, directory: Path, contract: TargetContract | None = None
) -> str:
    """Decision: where should the county inclusion floor sit?

    The single most important figure in this milestone — it is the evidence for
    excluding four counties. Emphasis form: excluded counties in the warning
    hue, retained counties in one neutral series hue.
    """
    contract = contract or load_contract()
    labelled = panel.loc[panel["has_target"]]
    table = analysis.county_dispersion(labelled)

    fig, ax = _figure(9, 5.0)
    excluded = table["volume"] < contract.min_homes_sold
    ax.scatter(
        table.loc[~excluded, "volume"],
        table.loc[~excluded, "dg_sd"],
        s=42,
        color=SERIES[0],
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1.4,
        label="retained",
    )
    ax.scatter(
        table.loc[excluded, "volume"],
        table.loc[excluded, "dg_sd"],
        s=64,
        color=HEATING,
        zorder=4,
        edgecolor=SURFACE,
        linewidth=1.4,
        label="excluded",
        marker="D",
    )
    ax.axvline(contract.min_homes_sold, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.text(
        contract.min_homes_sold * 1.12,
        table["dg_sd"].max() * 0.95,
        f"inclusion floor\nmedian ≥ {contract.min_homes_sold:g} sales/month",
        fontsize=9,
        color=INK_SECONDARY,
        va="top",
    )

    for row in table.loc[excluded].itertuples():
        ax.annotate(
            f"  {row.county_name.replace(' County', '')}",
            xy=(row.volume, row.dg_sd),
            fontsize=8.5,
            color=HEATING,
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("median homes sold per month (log scale)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("standard deviation of Δg (pp)", fontsize=9.5, color=INK_SECONDARY)
    _title(
        ax,
        "Target volatility is a function of market thinness, not market drama",
        "One point per county. Below roughly 25 sales a month the target is mostly "
        "arithmetic noise from a handful of transactions.",
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, loc="upper right")
    return _save(fig, "fig04_volume_vs_dispersion.png", directory)


def seasonality(model: pd.DataFrame, directory: Path) -> str:
    """Decision: do we need calendar-month features in Milestone 3?

    Plotted with ±2 standard errors so the reader can see whether any monthly
    mean is distinguishable from zero at all, rather than reading noise as a
    seasonal pattern.
    """
    table = analysis.seasonality(model)

    fig, ax = _figure(9, 4.2)
    positions = np.arange(len(table))
    ax.bar(
        positions,
        table["mean"],
        width=0.62,
        color=SERIES[0],
        zorder=2,
        yerr=2 * table["stderr"],
        error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capsize": 3},
    )
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(table.index, fontsize=9)
    ax.set_ylabel("mean Δg (pp)", fontsize=9.5, color=INK_SECONDARY)
    _title(
        ax,
        "There is no seasonal pattern worth modelling",
        "Mean target by calendar month with ±2 standard errors. Every month sits "
        "within about one point of zero, against a target standard deviation near 10.",
    )
    _style(ax)
    return _save(fig, "fig05_seasonality.png", directory)


def build_all(
    panel: pd.DataFrame,
    model: pd.DataFrame,
    directory: Path | None = None,
) -> list[str]:
    """Render every Milestone 2 figure and return their project-relative paths."""
    directory = Path(directory) if directory else FIGURES_DIR
    contract = load_contract()
    return [
        target_distribution(model, directory, contract),
        prevalence_by_year(model, directory, contract),
        county_time_series(panel, model, directory),
        volume_versus_dispersion(panel, directory, contract),
        seasonality(model, directory),
    ]
