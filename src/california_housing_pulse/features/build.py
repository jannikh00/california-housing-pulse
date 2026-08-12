"""Execution of the feature specification against the panel.

``spec`` declares, ``transforms`` computes, this module assembles. The order of
operations is the part that matters and it is fixed:

1. sort onto the county-month spine and check it has no holes;
2. apply each source's missing-data policy to the *raw* columns, recording an
   indicator wherever a value was filled;
3. run every transform, each carrying its source's publication lag;
4. carry the key, context and label columns through untouched.

Step 2 precedes step 3 deliberately. Filling after transforming would let a
carried-forward value be treated as an observation by some features and as a gap
by others depending on which window happened to straddle it, and the indicator
would no longer mean one thing.

Nothing in this module drops rows. Early months carry NA features because the
history is not there yet; that is a property of the data, and the split and the
model pipeline decide what to do about it downstream, in the open.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..paths import REPORTS_DIR
from .spec import FeatureContract, FeatureSpec, load_feature_contract
from .transforms import GROUP_KEY, apply_transform, assert_complete_spine

PANEL_KEY = ["county_fips", "reference_month"]

# Carried onto the feature frame unchanged. These are keys, reporting dimensions
# and labels — never model inputs. `volume_tier` in particular is a reporting
# dimension only: it is derived from a whole-panel median and would leak.
CONTEXT_COLUMNS = (
    "county_name",
    "prediction_as_of",
    "volume_tier",
    "is_included",
    "has_target",
    "growth_yoy",
    "target_dg",
    "target_label",
)

AVAILABILITY_REPORT = REPORTS_DIR / "feature_availability.md"


@dataclass
class FeatureReport:
    """Row and coverage accounting for feature construction."""

    rows: int = 0
    feature_count: int = 0
    families: dict[str, int] | None = None
    imputed_cells: dict[str, int] | None = None
    max_reach_months: int = 0
    complete_rows: int = 0
    complete_modeling_rows: int = 0
    modeling_rows: int = 0
    first_complete_month: pd.Timestamp | None = None

    def summary(self) -> str:
        lines = [
            f"features: {self.feature_count} columns over {self.rows:,} panel rows",
            f"  deepest feature reads {self.max_reach_months} months of history",
        ]
        if self.families:
            spread = ", ".join(f"{name} {count}" for name, count in sorted(self.families.items()))
            lines.append(f"  families: {spread}")
        if self.imputed_cells:
            filled = ", ".join(f"{col} {n:,}" for col, n in sorted(self.imputed_cells.items()))
            lines.append(f"  carried forward: {filled}")
        lines.append(
            f"  complete feature rows: {self.complete_modeling_rows:,} of "
            f"{self.modeling_rows:,} modelling rows"
            + (
                f" (from {self.first_complete_month:%Y-%m})"
                if self.first_complete_month is not None
                else ""
            )
        )
        return "\n".join(lines)


def apply_missing_policy(
    panel: pd.DataFrame,
    contract: FeatureContract,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill declared gaps in place and flag every cell that was filled.

    Only sources with a policy in ``configs/features.yaml`` are touched. Filling
    is forward-only within a county, so it uses no information the forecaster
    would not have had, and it is capped at ``max_gap_months`` so that a future
    multi-year outage fails the coverage check instead of quietly propagating a
    stale value across years of panel.
    """
    out = panel.copy()
    filled: dict[str, int] = {}

    for policy in contract.missing.values():
        columns = [
            spec.column for spec in contract.features if spec.source == policy.source_id
        ]
        columns = list(dict.fromkeys(columns))
        if not columns:
            continue
        if policy.method != "ffill_within_county":
            raise ValueError(
                f"Unsupported missing-data method '{policy.method}' for "
                f"'{policy.source_id}'; this module implements 'ffill_within_county'."
            )

        was_missing = out[columns].isna().any(axis=1)
        for column in columns:
            out[column] = (
                out.groupby(GROUP_KEY, sort=False)[column]
                .ffill(limit=policy.max_gap_months or None)
                .astype("float64")
            )
        now_present = out[columns].notna().all(axis=1)
        imputed = was_missing & now_present

        if policy.indicator:
            out[policy.indicator] = imputed.astype("float64")
            filled[policy.indicator] = int(imputed.sum())

    return out, filled


def _indicator_specs(contract: FeatureContract) -> list[FeatureSpec]:
    """Synthesise a spec for each missing-data indicator.

    The indicator is a feature like any other and must carry its source's
    publication lag: at reference month *t* the model reads the unemployment
    value from ``t - 2``, so what it needs to know is whether *that* month was
    filled, not whether *t* was.
    """
    specs = []
    for policy in contract.missing.values():
        if not policy.indicator:
            continue
        timing = contract.sources[policy.source_id]
        specs.append(
            FeatureSpec(
                name=policy.indicator,
                column=policy.indicator,
                source=policy.source_id,
                family="quality",
                transform="lag",
                param=0,
                offset=0,
                release_lag_months=timing.release_lag_months,
            )
        )
    return specs


def all_specs(contract: FeatureContract) -> list[FeatureSpec]:
    """Declared features followed by the synthesised missingness indicators."""
    return [*contract.features, *_indicator_specs(contract)]


def build_features(
    panel: pd.DataFrame,
    contract: FeatureContract | None = None,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Return the feature matrix and an accounting of what it contains."""
    contract = contract or load_feature_contract()

    out = panel.sort_values(PANEL_KEY, ignore_index=True).copy()
    assert_complete_spine(out)

    missing_from_panel = [
        spec.column for spec in contract.features if spec.column not in out.columns
    ]
    if missing_from_panel:
        raise KeyError(
            "Feature specification references columns absent from the panel: "
            f"{', '.join(sorted(set(missing_from_panel)))}."
        )

    prepared, filled = apply_missing_policy(out, contract)
    specs = all_specs(contract)

    columns = {
        key: out[key] for key in PANEL_KEY + [c for c in CONTEXT_COLUMNS if c in out.columns]
    }
    for spec in specs:
        columns[spec.name] = apply_transform(
            prepared,
            spec.column,
            spec.transform,
            spec.param,
            lag_months=spec.effective_lag,
        )
    features = pd.DataFrame(columns)

    names = [spec.name for spec in specs]
    complete = features[names].notna().all(axis=1)
    modeling = features["is_included"] & features["has_target"]

    families: dict[str, int] = {}
    for spec in specs:
        families[spec.family] = families.get(spec.family, 0) + 1

    report = FeatureReport(
        rows=len(features),
        feature_count=len(names),
        families=families,
        imputed_cells=filled,
        max_reach_months=max(spec.reach_months for spec in specs),
        complete_rows=int(complete.sum()),
        modeling_rows=int(modeling.sum()),
        complete_modeling_rows=int((complete & modeling).sum()),
        first_complete_month=(
            features.loc[complete & modeling, "reference_month"].min()
            if (complete & modeling).any()
            else None
        ),
    )
    return features, report


def feature_names(contract: FeatureContract | None = None) -> list[str]:
    """The model-input columns, in build order. Everything else is context."""
    contract = contract or load_feature_contract()
    return [spec.name for spec in all_specs(contract)]


def availability_table(
    features: pd.DataFrame,
    contract: FeatureContract | None = None,
) -> pd.DataFrame:
    """The feature-availability table Milestone 3 requires as a deliverable.

    One row per feature, stating the oldest month it reads, the earliest
    reference month at which it can be computed, and the observed null rate.
    Declared intent and measured reality side by side, as in the data dictionary.
    """
    contract = contract or load_feature_contract()
    panel_start = features["reference_month"].min()

    rows = []
    for spec in all_specs(contract):
        earliest = panel_start + pd.DateOffset(months=spec.reach_months)
        observed = features[spec.name]
        first_valid = features.loc[observed.notna(), "reference_month"]
        rows.append(
            {
                "feature": spec.name,
                "family": spec.family,
                "source": spec.source,
                "definition": spec.describe(),
                "release_lag_months": spec.release_lag_months,
                "reads_back_months": spec.reach_months,
                "earliest_month_expected": earliest,
                "earliest_month_observed": first_valid.min() if len(first_valid) else pd.NaT,
                "null_rate": float(observed.isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def _render_availability(
    table: pd.DataFrame,
    contract: FeatureContract,
    report: FeatureReport,
) -> str:
    lines = [
        "# Feature availability",
        "",
        "Generated by `chp features` from `configs/features.yaml`. Every row states "
        "the oldest month the feature reads and the earliest reference month at "
        "which it can be computed, so a reviewer can check the leakage claim "
        "without reading the code.",
        "",
        "Each row of the panel carries `prediction_as_of`, the 15th of the month "
        "after its reference month. That is the forecast cutoff. A feature is "
        "leakage-safe when every input it reads was published on or before that "
        "date; `release_lag_months` is how this file enforces it, and "
        "`tests/test_features.py` checks both that claim and the stronger one that "
        "recomputing a feature on a truncated panel reproduces it exactly.",
        "",
        "## Publication lags",
        "",
        "| Source | Release lag | Why |",
        "|---|---|---|",
    ]
    for timing in contract.sources.values():
        rationale = " ".join(timing.rationale.split())
        lines.append(f"| `{timing.source_id}` | {timing.release_lag_months} mo | {rationale} |")

    lines += ["", "## Missing-data policy", "", "| Column / source | Policy | Why |", "|---|---|---|"]
    for policy in contract.missing.values():
        rationale = " ".join(policy.rationale.split())
        detail = f"{policy.method}, max {policy.max_gap_months} mo"
        if policy.indicator:
            detail += f", flagged by `{policy.indicator}`"
        lines.append(f"| `{policy.source_id}` | {detail} | {rationale} |")
    for column, rationale in contract.excluded_columns.items():
        lines.append(f"| `{column}` | excluded | {' '.join(rationale.split())} |")

    lines += [
        "",
        "## Features",
        "",
        f"{report.feature_count} columns. The deepest reads "
        f"{report.max_reach_months} months of history, which is the feature-side "
        "lead-in: no row earlier than that many months into the panel can have a "
        "complete feature vector.",
        "",
        "| Feature | Family | Source | Definition | Reads back | First available | Null rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        observed = (
            f"{row.earliest_month_observed:%Y-%m}"
            if pd.notna(row.earliest_month_observed)
            else "never"
        )
        lines.append(
            f"| `{row.feature}` | {row.family} | `{row.source}` | {row.definition} "
            f"| {row.reads_back_months} mo | {observed} | {row.null_rate:.1%} |"
        )

    lines += ["", "## Coverage", "", report.summary(), ""]
    return "\n".join(lines)


def write_availability_report(
    features: pd.DataFrame,
    report: FeatureReport,
    contract: FeatureContract | None = None,
) -> str:
    """Render `reports/feature_availability.md` and return its relative path."""
    from ..paths import relative

    contract = contract or load_feature_contract()
    table = availability_table(features, contract)
    AVAILABILITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    AVAILABILITY_REPORT.write_text(_render_availability(table, contract, report))
    return relative(AVAILABILITY_REPORT)
