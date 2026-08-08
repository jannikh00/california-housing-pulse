"""Construction of the frozen modelling target.

The contract itself lives in ``configs/target.yaml``; this module only executes
it. Keeping the two apart means the definition of what the project predicts is a
reviewable, diffable artifact rather than a set of literals buried in code, and a
guard test asserts the config still matches the values frozen at Milestone 2.

**Timing.** For a county at reference month *t*:

``price_smoothed(t)``
    Mean of ``median_sale_price`` over ``[t-2, t]``. ``min_periods`` equals the
    window, so a partially observed window yields NA instead of a mean over
    fewer months — coverage gaps must stay visible, not be quietly smoothed over.

``growth_yoy(t)``
    ``100 * ln(price_smoothed(t) / price_smoothed(t-12))``, in percentage points.
    Uses only months ``<= t``, so it is a legitimate prediction-time feature.

``target_dg(t)``
    ``growth_yoy(t+3) - growth_yoy(t)``. This one reaches *forward* on purpose:
    it is the label, observable only at ``t+3``. Nothing on the feature side may
    ever be derived from it.

Every calculation is grouped by county and evaluated on a panel sorted by
``(county_fips, reference_month)``. That ordering matters: the shifts are
positional, and the complete county-month spine built in Milestone 1 is what
makes positional shifts equivalent to calendar shifts. A gap in the spine would
silently reinterpret a 12-row shift as something other than twelve months, which
is why :func:`add_target` re-sorts rather than trusting its input.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..paths import CONFIG_DIR

TARGET_CONFIG = CONFIG_DIR / "target.yaml"

PANEL_KEY = ["county_fips", "reference_month"]

# Columns this module adds to the panel, in the order they are produced.
TARGET_COLUMNS = (
    "homes_sold_median",
    "volume_tier",
    "is_included",
    "price_smoothed",
    "growth_yoy",
    "target_dg",
    "target_label",
    "has_target",
)


@dataclass(frozen=True)
class VolumeTier:
    """One county-volume band, keyed on median monthly ``homes_sold``."""

    name: str
    min: float
    max: float | None


@dataclass(frozen=True)
class TargetContract:
    """The frozen definition of what the project predicts."""

    price_column: str
    target_name: str
    label_name: str
    smoothing_window: int
    smoothing_statistic: str
    smoothing_min_periods: int
    growth_method: str
    growth_lag_months: int
    horizon_months: int
    tau: float
    min_homes_sold: float
    excluded_tier: str
    volume_tiers: tuple[VolumeTier, ...]
    frozen_status: str
    frozen_date: str

    @property
    def label_names(self) -> tuple[str, str, str]:
        return ("cooling", "stable", "heating")

    def describe(self) -> str:
        """One-line human summary, used in generated reports."""
        return (
            f"{self.smoothing_window}-month rolling {self.smoothing_statistic} of "
            f"{self.price_column} -> {self.growth_method} "
            f"{self.growth_lag_months}-month growth -> "
            f"{self.horizon_months}-month change, tau = +/-{self.tau:g} pp, "
            f"counties with median homes_sold < {self.min_homes_sold:g} excluded"
        )


@lru_cache(maxsize=1)
def load_contract(config_path: Path | None = None) -> TargetContract:
    """Parse ``configs/target.yaml`` into the frozen :class:`TargetContract`."""
    path = Path(config_path) if config_path else TARGET_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Target contract not found at {path}")

    raw = yaml.safe_load(path.read_text())
    target = raw["target"]
    inclusion = raw["inclusion"]
    frozen = raw["frozen"]

    tiers = tuple(
        VolumeTier(name=entry["name"], min=float(entry["min"]), max=entry["max"])
        for entry in inclusion["volume_tiers"]
    )

    return TargetContract(
        price_column=target["price_column"],
        target_name=target["name"],
        label_name=target["label_name"],
        smoothing_window=int(target["smoothing"]["window_months"]),
        smoothing_statistic=target["smoothing"]["statistic"],
        smoothing_min_periods=int(target["smoothing"]["min_periods"]),
        growth_method=target["growth"]["method"],
        growth_lag_months=int(target["growth"]["lag_months"]),
        horizon_months=int(target["horizon_months"]),
        tau=float(target["thresholds"]["tau"]),
        min_homes_sold=float(inclusion["min_homes_sold"]),
        excluded_tier=inclusion["excluded_tier"],
        volume_tiers=tiers,
        frozen_status=frozen["status"],
        frozen_date=str(frozen["date"]),
    )


def smooth_price(panel: pd.DataFrame, contract: TargetContract) -> pd.Series:
    """Rolling smoothed price per county, ending at the reference month."""
    if contract.smoothing_statistic != "mean":
        raise ValueError(
            f"Unsupported smoothing statistic '{contract.smoothing_statistic}'; "
            "the frozen contract specifies 'mean'."
        )
    return panel.groupby("county_fips", sort=False)[contract.price_column].transform(
        lambda series: series.rolling(
            contract.smoothing_window,
            min_periods=contract.smoothing_min_periods,
        ).mean()
    )


def year_over_year_growth(
    panel: pd.DataFrame,
    smoothed: pd.Series,
    contract: TargetContract,
) -> pd.Series:
    """Growth of the smoothed price against the same month a year earlier.

    Expressed in percentage points. Log growth is used because the target is a
    difference of two growth rates and must be symmetric under a symmetric
    threshold; see the rationale recorded in ``configs/target.yaml``.
    """
    if contract.growth_method != "log":
        raise ValueError(
            f"Unsupported growth method '{contract.growth_method}'; "
            "the frozen contract specifies 'log'."
        )
    lagged = smoothed.groupby(panel["county_fips"], sort=False).shift(
        contract.growth_lag_months
    )
    ratio = smoothed / lagged
    # A non-positive ratio cannot be logged. Prices are bounded above zero by the
    # column registry, so this guards against a future source change rather than
    # anything present today; NA is the honest answer either way.
    ratio = ratio.where(ratio > 0)
    return 100.0 * np.log(ratio)


def forward_change(
    panel: pd.DataFrame,
    growth: pd.Series,
    contract: TargetContract,
) -> pd.Series:
    """``growth(t + horizon) - growth(t)`` — the label, in percentage points."""
    ahead = growth.groupby(panel["county_fips"], sort=False).shift(-contract.horizon_months)
    return ahead - growth


def classify(values: pd.Series, contract: TargetContract) -> pd.Series:
    """Map the continuous target onto cooling / stable / heating.

    Rows with no target stay NA rather than collapsing into ``stable``: an
    unobservable outcome is not a prediction of no change.
    """
    labels = pd.Series(pd.NA, index=values.index, dtype="string")
    observed = values.notna()
    labels[observed & (values <= -contract.tau)] = "cooling"
    labels[observed & (values >= contract.tau)] = "heating"
    labels[observed & (values > -contract.tau) & (values < contract.tau)] = "stable"
    return labels


def county_volume(panel: pd.DataFrame) -> pd.Series:
    """Median monthly ``homes_sold`` per county, the inclusion statistic."""
    return panel.groupby("county_fips")["homes_sold"].median()


def assign_tier(volume: pd.Series, contract: TargetContract) -> pd.Series:
    """Bin a county-volume series into the contract's volume tiers.

    Binning uses each tier's lower bound only, so the bands are exhaustive and a
    fractional median (Lassen County's 24.5, for instance) cannot fall in a gap
    between two integer ranges.
    """
    tiers = sorted(contract.volume_tiers, key=lambda tier: tier.min)
    edges = [tier.min for tier in tiers] + [float("inf")]
    names = [tier.name for tier in tiers]
    binned = pd.cut(volume, bins=edges, labels=names, right=False, include_lowest=True)
    return pd.Series(binned, index=volume.index).astype("string")


def add_target(
    panel: pd.DataFrame,
    contract: TargetContract | None = None,
) -> pd.DataFrame:
    """Return the panel with the frozen target and inclusion columns attached."""
    contract = contract or load_contract()
    out = panel.sort_values(PANEL_KEY, ignore_index=True).copy()

    volume = county_volume(out)
    tiers = assign_tier(volume, contract)
    out["homes_sold_median"] = out["county_fips"].map(volume).astype("float64")
    out["volume_tier"] = out["county_fips"].map(tiers).astype("string")
    out["is_included"] = out["homes_sold_median"] >= contract.min_homes_sold

    out["price_smoothed"] = smooth_price(out, contract)
    out["growth_yoy"] = year_over_year_growth(out, out["price_smoothed"], contract)
    out["target_dg"] = forward_change(out, out["growth_yoy"], contract)
    out["target_label"] = classify(out["target_dg"], contract)
    out["has_target"] = out["target_dg"].notna()

    return out


def modeling_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """The rows Milestone 3 may train and evaluate on.

    A row qualifies when its county clears the volume floor *and* the label is
    observable. Both conditions remain visible as columns on the full panel, so
    this filter is a convenience, never the only record of what was dropped.
    """
    return panel.loc[panel["is_included"] & panel["has_target"]].reset_index(drop=True)


@dataclass
class TargetReport:
    """Row accounting for the target-construction step."""

    panel_rows: int = 0
    counties: int = 0
    excluded_counties: tuple[str, ...] = ()
    included_rows: int = 0
    rows_with_target: int = 0
    modeling_rows: int = 0
    target_start: pd.Timestamp | None = None
    target_end: pd.Timestamp | None = None
    prevalence: dict[str, float] | None = None

    def summary(self) -> str:
        excluded = ", ".join(self.excluded_counties) or "none"
        lines = [
            f"target: {self.modeling_rows:,} modelling rows from {self.panel_rows:,} panel rows",
            f"  excluded {len(self.excluded_counties)} of {self.counties} counties "
            f"below the volume floor: {excluded}",
            f"  label observable for {self.rows_with_target:,} rows "
            f"({self.target_start:%Y-%m} to {self.target_end:%Y-%m})",
        ]
        if self.prevalence:
            shares = ", ".join(f"{name} {share:.1%}" for name, share in self.prevalence.items())
            lines.append(f"  class prevalence: {shares}")
        return "\n".join(lines)


def summarize(panel: pd.DataFrame, contract: TargetContract | None = None) -> TargetReport:
    """Measure what target construction produced."""
    contract = contract or load_contract()
    model = modeling_rows(panel)
    excluded = (
        panel.loc[~panel["is_included"], ["county_fips", "county_name"]]
        .drop_duplicates()
        .sort_values("county_fips")
    )
    with_target = panel.loc[panel["has_target"]]

    prevalence = None
    if len(model):
        counts = model["target_label"].value_counts(normalize=True)
        prevalence = {name: float(counts.get(name, 0.0)) for name in contract.label_names}

    return TargetReport(
        panel_rows=len(panel),
        counties=panel["county_fips"].nunique(),
        excluded_counties=tuple(excluded["county_name"].astype(str)),
        included_rows=int(panel["is_included"].sum()),
        rows_with_target=len(with_target),
        modeling_rows=len(model),
        target_start=with_target["reference_month"].min() if len(with_target) else None,
        target_end=with_target["reference_month"].max() if len(with_target) else None,
        prevalence=prevalence,
    )
