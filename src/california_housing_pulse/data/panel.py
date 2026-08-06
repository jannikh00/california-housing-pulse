"""The joined county-month panel.

The panel is built on a **complete spine**: every California county crossed with
every month in the coverage window, before any source is joined. Sources are then
left-joined onto that spine. A county-month with no Redfin observation therefore
appears as a row with missing values rather than as an absent row, which is what
makes coverage gaps countable instead of invisible.

The coverage window is driven by the Redfin housing series, because that source
defines the modelling target; the mortgage-rate and unemployment series extend
further back and are simply clipped to the window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..io import write_parquet
from ..paths import PROCESSED_DIR
from .normalize import month_range

PANEL_KEY = ["county_fips", "reference_month"]

# Milestone 0 fixes the forecast cutoff at the 15th day of the month following
# the reference month. It is a deterministic function of the key, so it is
# materialized here and carried through to feature construction.
PREDICTION_CUTOFF_DAY = 15


@dataclass
class JoinReport:
    """Row accounting for the panel build."""

    spine_rows: int = 0
    counties: int = 0
    months: int = 0
    coverage_start: pd.Timestamp | None = None
    coverage_end: pd.Timestamp | None = None
    joins: dict[str, dict[str, int]] = field(default_factory=dict)

    def add_join(self, name: str, *, matched: int, unmatched_source_rows: int) -> None:
        self.joins[name] = {
            "spine_rows_matched": int(matched),
            "spine_rows_unmatched": int(self.spine_rows - matched),
            "source_rows_outside_spine": int(unmatched_source_rows),
        }

    def summary(self) -> str:
        lines = [
            f"spine: {self.spine_rows:,} rows "
            f"({self.counties} counties x {self.months} months, "
            f"{self.coverage_start:%Y-%m} to {self.coverage_end:%Y-%m})"
        ]
        for name, counts in self.joins.items():
            matched = counts["spine_rows_matched"]
            pct = 100 * matched / self.spine_rows if self.spine_rows else 0
            lines.append(
                f"  {name}: matched {matched:,} spine rows ({pct:.1f}%), "
                f"{counts['spine_rows_unmatched']:,} spine rows without data, "
                f"{counts['source_rows_outside_spine']:,} source rows outside the spine"
            )
        return "\n".join(lines)


def build_spine(
    counties: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Cross-join every county with every month in ``[start, end]``."""
    months = month_range(start, end)
    spine = (
        counties[["county_fips", "county_name"]]
        .merge(pd.DataFrame({"reference_month": months}), how="cross")
        .sort_values(PANEL_KEY, ignore_index=True)
    )
    return spine


def prediction_as_of(reference_month: pd.Series) -> pd.Series:
    """The documented forecast cutoff: the 15th of the following month."""
    following = reference_month + pd.offsets.MonthBegin(1)
    return following + pd.Timedelta(days=PREDICTION_CUTOFF_DAY - 1)


def _left_join(
    panel: pd.DataFrame,
    source: pd.DataFrame,
    *,
    on: list[str],
    name: str,
    report: JoinReport,
    validate: str,
) -> pd.DataFrame:
    """Left-join a staged source onto the panel and record what matched.

    ``validate`` is passed to pandas so an unexpected fan-out raises instead of
    silently multiplying rows: county-month sources must be ``one_to_one`` with
    the spine, while the national mortgage series is ``many_to_one`` because one
    monthly rate applies to all 58 counties.
    """
    indicator = source[on].drop_duplicates().assign(_matched=True)
    probe = panel[on].merge(indicator, on=on, how="left")
    matched = int(probe["_matched"].fillna(False).sum())

    keys_in_panel = panel[on].drop_duplicates()
    outside = len(source.merge(keys_in_panel, on=on, how="left", indicator=True).query(
        "_merge == 'left_only'"
    ))
    report.add_join(name, matched=matched, unmatched_source_rows=outside)

    return panel.merge(source, on=on, how="left", validate=validate)


def build_panel(
    tables: dict[str, pd.DataFrame],
    *,
    write: bool = True,
) -> tuple[pd.DataFrame, JoinReport]:
    """Join every staged table onto a complete county-month spine."""
    counties = tables["counties"]
    redfin = tables["redfin"]
    mortgage = tables["mortgage_rate"]
    unemployment = tables["unemployment"]

    report = JoinReport()
    start = redfin["reference_month"].min()
    end = redfin["reference_month"].max()

    spine = build_spine(counties, start, end)
    report.spine_rows = len(spine)
    report.counties = spine["county_fips"].nunique()
    report.months = spine["reference_month"].nunique()
    report.coverage_start = start
    report.coverage_end = end

    panel = spine.copy()
    panel = _left_join(
        panel,
        redfin.drop(columns=["county_name"]),
        on=PANEL_KEY,
        name="redfin",
        report=report,
        validate="one_to_one",
    )
    panel = _left_join(
        panel,
        mortgage,
        on=["reference_month"],
        name="mortgage_rate",
        report=report,
        validate="many_to_one",
    )
    panel = _left_join(
        panel,
        unemployment,
        on=PANEL_KEY,
        name="unemployment",
        report=report,
        validate="one_to_one",
    )

    panel["prediction_as_of"] = prediction_as_of(panel["reference_month"])
    panel["has_redfin"] = panel["median_sale_price"].notna()
    panel["has_unemployment"] = panel["unemployment_rate"].notna()

    panel = panel.sort_values(PANEL_KEY, ignore_index=True)

    if write:
        write_parquet(panel, PROCESSED_DIR / "county_month_panel.parquet")

    return panel, report
