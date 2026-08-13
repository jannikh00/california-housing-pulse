"""Block bootstrap confidence intervals for the test metrics.

A naive bootstrap that resamples individual rows would badly understate the
uncertainty here, because the rows are not independent in either direction. All
54 counties share a national mortgage-rate regime, so the errors within a month
are correlated; and each county's series is autocorrelated across months, so the
errors within a county are correlated too. Resampling rows would break both
structures and return an interval far too narrow to be believed.

So the resampling unit is a **block**, and two choices of block answer two
different questions:

``county``
    Resample whole county time series with replacement. 54 blocks, each keeping
    its internal time structure intact. This answers "how much does the result
    depend on *which counties* are in the sample?" — the relevant question given
    Milestone 2 found dispersion to be largely a function of county thinness.

``month``
    Resample whole months with replacement. Only 12 blocks in the test window,
    keeping the cross-county correlation intact. This answers "how much does the
    result depend on *which months* were tested?" With twelve blocks the interval
    is wide, and that width is the honest answer rather than a defect: Milestone 2
    Finding 3 showed the class mix swings enormously between regimes.

Both are reported. Where they disagree, they disagree informatively.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 20260811

BLOCK_COLUMNS = {"county": "county_fips", "month": "reference_month"}


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval for one metric."""

    metric: str
    block: str
    point: float
    low: float
    high: float
    resamples: int

    def describe(self, digits: int = 3) -> str:
        return f"{self.point:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def block_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    block: str = "county",
    metric: str = "metric",
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Percentile interval for ``statistic`` under resampling of whole blocks."""
    try:
        column = BLOCK_COLUMNS[block]
    except KeyError:
        raise KeyError(
            f"Unknown block '{block}'; choose one of {', '.join(sorted(BLOCK_COLUMNS))}."
        ) from None

    point = statistic(frame)
    groups = [part for _, part in frame.groupby(column, sort=True, observed=True)]
    if len(groups) < 2:
        return Interval(metric, block, point, np.nan, np.nan, 0)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[i] for i in picks], ignore_index=True)
        value = statistic(sample)
        if np.isfinite(value):
            draws.append(value)

    if not draws:
        return Interval(metric, block, point, np.nan, np.nan, 0)

    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(metric, block, point, float(low), float(high), len(draws))


def _statistic(metric: str) -> Callable[[pd.DataFrame], float]:
    """Build a callable returning one named metric from a scored frame."""
    from .metrics import directional_metrics, magnitude_metrics

    def compute(frame: pd.DataFrame) -> float:
        if metric in ("mae", "rmse", "mean_error", "medae"):
            return magnitude_metrics(frame["target_dg"], frame["predicted_dg"]).get(
                metric, np.nan
            )
        return directional_metrics(frame["target_label"], frame["predicted_label"]).get(
            metric, np.nan
        )

    return compute


def intervals_for(
    frame: pd.DataFrame,
    metrics: tuple[str, ...] = ("mae", "macro_f1"),
    blocks: tuple[str, ...] = ("county", "month"),
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Intervals for each requested metric under each blocking scheme."""
    rows = []
    for metric in metrics:
        if metric in ("mae", "rmse", "mean_error", "medae"):
            if "predicted_dg" not in frame.columns:
                continue
        elif "predicted_label" not in frame.columns:
            continue
        for block in blocks:
            interval = block_bootstrap(
                frame,
                _statistic(metric),
                block=block,
                metric=metric,
                resamples=resamples,
                seed=seed,
            )
            rows.append(
                {
                    "metric": interval.metric,
                    "block": interval.block,
                    "point": interval.point,
                    "low": interval.low,
                    "high": interval.high,
                    "resamples": interval.resamples,
                }
            )
    return pd.DataFrame(rows)


def paired_difference(
    frame: pd.DataFrame,
    other: pd.DataFrame,
    metric: str = "mae",
    *,
    block: str = "county",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Interval for the *difference* between two models on the same rows.

    Paired rather than comparing two separate intervals, which is the mistake
    this function exists to avoid: two overlapping confidence intervals do not
    imply the difference is indistinguishable from zero. Both models saw the same
    counties and months, so resampling the blocks jointly preserves that pairing
    and answers the question actually being asked — did this model beat that one?
    """
    column = BLOCK_COLUMNS[block]
    keys = ["county_fips", "reference_month"]
    merged = frame[keys + [c for c in frame.columns if c not in keys]].merge(
        other[keys + [c for c in other.columns if c not in keys]],
        on=keys,
        suffixes=("_a", "_b"),
    )

    compute = _statistic(metric)

    def difference(sample: pd.DataFrame) -> float:
        left = sample.rename(columns=lambda c: c[:-2] if c.endswith("_a") else c)
        right = sample.rename(columns=lambda c: c[:-2] if c.endswith("_b") else c)
        return compute(left) - compute(right)

    point = difference(merged)
    groups = [part for _, part in merged.groupby(column, sort=True, observed=True)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        picks = rng.integers(0, len(groups), size=len(groups))
        value = difference(pd.concat([groups[i] for i in picks], ignore_index=True))
        if np.isfinite(value):
            draws.append(value)

    if not draws:
        return Interval(f"{metric}_difference", block, point, np.nan, np.nan, 0)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(f"{metric}_difference", block, point, float(low), float(high), len(draws))
