"""Decision-oriented measurements over the target panel.

Every function here returns a table or a plain summary object and draws nothing.
Keeping the arithmetic separate from the rendering is what lets the EDA findings
be asserted in tests: a claim in the memo is only worth making if the number
behind it is reproducible.

The distinction that matters throughout: the **panel** is all 10,034 county-months
including excluded counties and unlabelled lead-in rows, while the **modelling
rows** are what Milestone 3 may actually train on. Coverage questions are asked of
the panel; target questions are asked of the modelling rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.columns import bounded_columns
from ..features.target import TargetContract, load_contract

# Calendar months, for seasonality tables.
MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class Distribution:
    """Summary of the continuous target."""

    rows: int
    mean: float
    std: float
    median: float
    iqr: float
    abs_median: float
    quantiles: dict[float, float]
    start: pd.Timestamp
    end: pd.Timestamp


def describe_target(model: pd.DataFrame, column: str = "target_dg") -> Distribution:
    """Summarize the continuous target over the modelling rows."""
    values = model[column].dropna()
    probs = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
    return Distribution(
        rows=len(values),
        mean=float(values.mean()),
        std=float(values.std()),
        median=float(values.median()),
        iqr=float(values.quantile(0.75) - values.quantile(0.25)),
        abs_median=float(values.abs().median()),
        quantiles={p: float(values.quantile(p)) for p in probs},
        start=model["reference_month"].min(),
        end=model["reference_month"].max(),
    )


def _prevalence(frame: pd.DataFrame, by: pd.Series | None, contract: TargetContract):
    """Class shares, always with all three classes present as columns."""
    labels = contract.label_names
    if by is None:
        counts = frame["target_label"].value_counts()
        shares = (counts / counts.sum()).reindex(labels).fillna(0.0)
        return shares
    table = pd.crosstab(by, frame["target_label"], normalize="index")
    return table.reindex(columns=labels).fillna(0.0)


def prevalence_overall(model: pd.DataFrame, contract: TargetContract | None = None) -> pd.Series:
    return _prevalence(model, None, contract or load_contract())


def prevalence_by_year(model: pd.DataFrame, contract: TargetContract | None = None) -> pd.DataFrame:
    """Class shares per calendar year, with the row count behind each."""
    contract = contract or load_contract()
    years = model["reference_month"].dt.year.rename("year")
    table = _prevalence(model, years, contract)
    table["rows"] = years.value_counts().sort_index()
    return table


def prevalence_by_tier(model: pd.DataFrame, contract: TargetContract | None = None) -> pd.DataFrame:
    """Class shares per county-volume tier — the thin-market distortion."""
    contract = contract or load_contract()
    order = [tier.name for tier in sorted(contract.volume_tiers, key=lambda t: t.min)]
    present = [name for name in order if name in set(model["volume_tier"])]
    table = _prevalence(model, model["volume_tier"], contract).reindex(present)
    grouped = model.groupby("volume_tier", observed=True)
    table["counties"] = grouped["county_fips"].nunique()
    table["rows"] = grouped.size()
    table["dg_sd"] = grouped["target_dg"].std()
    return table


def prevalence_by_county(
    model: pd.DataFrame, contract: TargetContract | None = None
) -> pd.DataFrame:
    """Per-county class shares, ordered by the rarest class.

    Counties whose rarest class is very small are the ones a directional model
    will effectively never predict correctly for.
    """
    contract = contract or load_contract()
    table = _prevalence(model, model["county_name"], contract)
    table["smallest_class"] = table[list(contract.label_names)].min(axis=1)
    table["dg_sd"] = model.groupby("county_name")["target_dg"].std()
    table["volume"] = model.groupby("county_name")["homes_sold_median"].first()
    return table.sort_values("smallest_class")


def county_dispersion(model: pd.DataFrame) -> pd.DataFrame:
    """Per-county target volatility against volume — the inclusion-rule evidence."""
    grouped = model.groupby(["county_fips", "county_name"], as_index=False)
    table = grouped.agg(
        volume=("homes_sold_median", "first"),
        tier=("volume_tier", "first"),
        rows=("target_dg", "size"),
        dg_sd=("target_dg", "std"),
        dg_abs_median=("target_dg", lambda s: s.abs().median()),
        dg_max_abs=("target_dg", lambda s: s.abs().max()),
    )
    return table.sort_values("dg_sd", ascending=False, ignore_index=True)


def excluded_county_dispersion(panel: pd.DataFrame) -> pd.DataFrame:
    """The same volatility measure for the counties the volume floor removed.

    Reported so the exclusion is justified by evidence rather than asserted.
    """
    excluded = panel.loc[~panel["is_included"] & panel["has_target"]]
    if not len(excluded):
        return pd.DataFrame()
    return county_dispersion(excluded)


def coverage_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-county source coverage and how many labelled rows survive."""
    grouped = panel.groupby(["county_fips", "county_name"], as_index=False)
    table = grouped.agg(
        months=("reference_month", "size"),
        months_with_price=("median_sale_price", "count"),
        months_with_unemployment=("unemployment_rate", "count"),
        labelled_rows=("has_target", "sum"),
        volume=("homes_sold_median", "first"),
        tier=("volume_tier", "first"),
        included=("is_included", "first"),
    )
    table["price_coverage"] = table["months_with_price"] / table["months"]
    return table.sort_values("price_coverage", ignore_index=True)


def missingness(panel: pd.DataFrame) -> pd.DataFrame:
    """Null rate per bounded column, worst first."""
    measured = [name for name in bounded_columns() if name in panel.columns]
    table = pd.DataFrame(
        {
            "column": measured,
            "null_rate": [panel[name].isna().mean() for name in measured],
            "nulls": [int(panel[name].isna().sum()) for name in measured],
        }
    )
    return table.sort_values("null_rate", ascending=False, ignore_index=True)


def feature_outliers(panel: pd.DataFrame) -> pd.DataFrame:
    """Values outside each column's declared plausible range.

    Uses the Milestone 1 registry rather than a fresh rule of thumb, so the
    outlier definition is the one already published in the data dictionary.
    """
    rows = []
    for name, spec in bounded_columns().items():
        if spec.plausible is None or name not in panel.columns:
            continue
        low, high = spec.plausible
        values = pd.to_numeric(panel[name], errors="coerce")
        outside = values.notna() & ((values < low) | (values > high))
        if not outside.any():
            continue
        affected = panel.loc[outside]
        top = (
            affected["county_name"].value_counts().head(3).index.tolist()
            if "county_name" in affected
            else []
        )
        rows.append(
            {
                "column": name,
                "plausible_low": low,
                "plausible_high": high,
                "extreme_values": int(outside.sum()),
                "share": float(outside.sum() / values.notna().sum()),
                "counties": ", ".join(str(name_) for name_ in top),
            }
        )
    return pd.DataFrame(rows).sort_values("extreme_values", ascending=False, ignore_index=True)


def seasonality(model: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of the target by calendar month.

    A real seasonal effect would show up as a mean that is large relative to the
    standard error, not merely non-zero.
    """
    month = model["reference_month"].dt.month.rename("month")
    grouped = model.groupby(month)["target_dg"]
    table = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(),
            "rows": grouped.size(),
        }
    )
    table["stderr"] = table["std"] / np.sqrt(table["rows"])
    # How many standard errors the monthly mean sits from zero.
    table["z"] = table["mean"] / table["stderr"]
    table.index = [MONTH_ABBR[i - 1] for i in table.index]
    return table


def monthly_series(model: pd.DataFrame) -> pd.DataFrame:
    """Cross-county mean target per reference month — the regime series."""
    grouped = model.groupby("reference_month")["target_dg"]
    table = pd.DataFrame({"mean": grouped.mean(), "std": grouped.std(), "rows": grouped.size()})
    table["rolling_12m_mean"] = table["mean"].rolling(12, min_periods=12).mean()
    table["rolling_12m_sd"] = table["mean"].rolling(12, min_periods=12).std()
    return table


def regime_shifts(model: pd.DataFrame, window: int = 12, top: int = 5) -> pd.DataFrame:
    """A deliberately simple change-point scan.

    For every candidate month with ``window`` months either side, measure the
    difference between the mean target after it and before it. The largest
    differences are the structural breaks. This is a descriptive scan, not a
    hypothesis test: it names *when* the market regime turned, and makes no claim
    about significance or cause.
    """
    series = monthly_series(model)["mean"]
    rows = []
    for position in range(window, len(series) - window):
        before = series.iloc[position - window : position]
        after = series.iloc[position : position + window]
        rows.append(
            {
                "month": series.index[position],
                "mean_before": float(before.mean()),
                "mean_after": float(after.mean()),
                "shift": float(after.mean() - before.mean()),
            }
        )
    table = pd.DataFrame(rows)
    if not len(table):
        return table
    table["abs_shift"] = table["shift"].abs()
    ranked = table.sort_values("abs_shift", ascending=False, ignore_index=True)
    # Suppress near-duplicate detections: one regime turn produces a run of
    # adjacent months that all score highly.
    kept: list[pd.Timestamp] = []
    selected = []
    for row in ranked.itertuples():
        if any(abs((row.month - seen).days) < 30 * window for seen in kept):
            continue
        kept.append(row.month)
        selected.append(row.Index)
        if len(selected) == top:
            break
    return ranked.loc[selected].reset_index(drop=True)


def rate_regime_prevalence(
    model: pd.DataFrame, contract: TargetContract | None = None
) -> pd.DataFrame:
    """Class prevalence by mortgage-rate band.

    The financing environment is the one macro driver in the dataset, so this is
    the cheapest available check on whether the target behaves differently across
    market regimes.
    """
    contract = contract or load_contract()
    bands = pd.cut(
        model["mortgage_rate_30y"],
        bins=[0, 4, 5, 6, 7, 100],
        labels=["<4%", "4-5%", "5-6%", "6-7%", ">7%"],
        right=False,
    )
    table = _prevalence(model, bands.rename("rate_band"), contract)
    grouped = model.groupby(bands.rename("rate_band"), observed=True)
    table["rows"] = grouped.size()
    table["dg_mean"] = grouped["target_dg"].mean()
    return table.dropna(subset=["rows"])


def representative_counties(model: pd.DataFrame, contract: TargetContract | None = None) -> dict:
    """One county per volume tier, for the illustrative time-series figure.

    The median-volume county within each tier is chosen rather than the largest,
    so the picture is typical of the tier instead of flattering to it.
    """
    contract = contract or load_contract()
    order = [tier.name for tier in sorted(contract.volume_tiers, key=lambda t: t.min)]
    volumes = model.groupby(["volume_tier", "county_fips", "county_name"], observed=True)[
        "homes_sold_median"
    ].first()

    chosen: dict[str, tuple[str, str]] = {}
    for tier in order:
        if tier not in volumes.index.get_level_values("volume_tier"):
            continue
        band = volumes.xs(tier, level="volume_tier").sort_values()
        fips, name = band.index[len(band) // 2]
        chosen[tier] = (fips, name)
    return chosen
