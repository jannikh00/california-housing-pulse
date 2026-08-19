"""Shared figure palette and chrome for every rendered figure in the project.

This module exists because Milestone 4 puts Milestone 2's exploratory figures and
Milestone 4's results figures side by side in one README. Two modules each
holding their own copy of the palette would drift the first time a colour was
adjusted in one of them, and a reader would see it: the same class hue meaning
the same thing across figures is the whole reason the palette is fixed.

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

Figures render light-only: they are committed artifacts in a repository, not a
themed web page.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in CI or a plain terminal.

import matplotlib.pyplot as plt  # noqa: E402

from .paths import relative  # noqa: E402

# Re-exported so consumers reach pyplot *through* this module. A module that did
# `import matplotlib.pyplot` on its own line would bind whatever backend is
# default before the `use("Agg")` above ever ran, and fail on a headless box.
__all__ = ["plt"]

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

# Volume tier is an *ordered* quantity, so it takes a sequential ramp rather than
# categorical hues: the reader should see thin-to-large as a progression, which
# is exactly the gradient Milestone 3 Finding 3 is about. Every step clears 3:1
# against the surface.
TIER_ORDER = ("thin", "small", "mid", "large")
TIER_COLORS = {
    "thin": "#9fc4ea",
    "small": "#6ba3dd",
    "mid": "#3d7fc4",
    "large": "#1d5490",
}


def style_axes(ax: plt.Axes, *, xgrid: bool = False) -> None:
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


def figure(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def panels(
    width: float, height: float, columns: int = 2, **kwargs
) -> tuple[plt.Figure, list[plt.Axes]]:
    """A row of panels sharing one figure surface.

    Used where a single claim has two measurements that must be read together —
    magnitude beside direction — and putting them in separate files would let a
    reader take one without the other.
    """
    fig, axes = plt.subplots(1, columns, figsize=(width, height), dpi=DPI, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, list(axes)


def title(ax: plt.Axes, headline: str, subtitle: str, *, wrap: int = 88) -> None:
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
        headline,
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


def figure_header(
    fig: plt.Figure,
    headline: str,
    subtitle: str,
    *,
    wrap: int = 110,
    panel_titles: bool = True,
) -> None:
    """Headline and subtitle for a *multi-panel* figure, in figure coordinates.

    :func:`title` cannot be reused here. It renders through ``ax.set_title``, and
    so does :func:`panel_title`, so a panel heading and a figure headline on the
    same axes silently overwrite each other — the figure still renders, with one
    of the two texts simply gone.

    Header height is measured in inches and converted to a figure fraction, so a
    tall figure and a short one leave the same visual gap rather than the same
    proportional one. ``panel_titles`` reserves the extra band the per-panel
    headings need.
    """
    wrapped = textwrap.fill(subtitle, wrap)
    lines = wrapped.count("\n") + 1

    height = fig.get_figheight()
    headline_inches = 0.30
    line_inches = 0.17
    gap_inches = 0.16
    panel_inches = 0.34 if panel_titles else 0.0
    header_inches = headline_inches + lines * line_inches + gap_inches + panel_inches

    fig.subplots_adjust(top=1 - header_inches / height)
    fig.text(
        0.0,
        1.0,
        headline,
        ha="left",
        va="top",
        fontsize=14,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.0,
        1 - headline_inches / height,
        wrapped,
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK_SECONDARY,
        linespacing=1.4,
    )


def panel_title(ax: plt.Axes, headline: str) -> None:
    """A quiet heading over one panel of a multi-panel figure.

    Deliberately lighter than :func:`title`: the figure has one headline, and a
    panel heading that competes with it makes the reader hunt for the claim.
    """
    ax.set_title(headline, loc="left", fontsize=10.5, color=INK_SECONDARY, pad=8)


def place_labels(ax: plt.Axes, entries: list[tuple[float, str, str]], x: float) -> None:
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


def tier_legend(ax: plt.Axes, **kwargs) -> None:
    """Legend for the sequential volume-tier ramp, in thin-to-large order."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[tier]) for tier in TIER_ORDER]
    defaults = {
        "frameon": False,
        "fontsize": 9,
        "labelcolor": INK_SECONDARY,
        "ncol": 4,
        "loc": "lower right",
    }
    ax.legend(handles, [f"{t} counties" for t in TIER_ORDER], **{**defaults, **kwargs})


def save(fig: plt.Figure, name: str, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return relative(path)
