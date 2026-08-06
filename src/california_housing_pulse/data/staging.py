"""Staged tables: one normalized table per source, written to ``data/interim/``.

Each ``stage_*`` function reads exactly one raw snapshot, applies the project's
identifier/date/unit conventions, and returns both the table and a
:class:`StagingReport` recording how many rows were read and why rows were
dropped. Those counts are what later turn "we filtered the data" into a
measured, reviewable statement rather than an assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..io import write_parquet
from ..paths import INTERIM_DIR
from .normalize import (
    CALIFORNIA_STATE_FIPS,
    normalize_county_name,
    snake_case,
    to_county_fips,
    to_reference_month,
)
from .sources import SourceRegistry, load_registry

# Redfin publishes several period durations and property types in one file.
# Both filters are mandatory: without them a county-month appears many times.
REDFIN_PERIOD_DURATION_DAYS = 30
REDFIN_PROPERTY_TYPE = "All Residential"

# Level indicators kept from Redfin. The source's own _MOM/_YOY derivatives are
# deliberately excluded: the project derives its own growth measures under an
# explicit prediction-time cutoff, and mixing in vendor-computed changes whose
# timing is undocumented would risk leakage.
REDFIN_MEASURES = [
    "median_sale_price",
    "median_list_price",
    "median_ppsf",
    "homes_sold",
    "pending_sales",
    "new_listings",
    "inventory",
    "months_of_supply",
    "median_dom",
    "avg_sale_to_list",
    "sold_above_list",
    "price_drops",
    "off_market_in_two_weeks",
]

# BLS LAUS measure codes, from the tail of the series id.
BLS_MEASURES = {
    "03": "unemployment_rate",
    "04": "unemployed",
    "05": "employed",
    "06": "labor_force",
}
BLS_COUNTY_AREA_TYPE = "F"


@dataclass
class StagingReport:
    """Row accounting for one staged source."""

    source_id: str
    rows_in: int = 0
    rows_out: int = 0
    drops: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def drop(self, reason: str, count: int) -> None:
        if count:
            self.drops[reason] = self.drops.get(reason, 0) + int(count)

    def summary(self) -> str:
        parts = [f"{self.source_id}: {self.rows_in:,} read -> {self.rows_out:,} staged"]
        for reason, count in self.drops.items():
            parts.append(f"    dropped {count:,} ({reason})")
        parts.extend(f"    note: {note}" for note in self.notes)
        return "\n".join(parts)


def stage_counties(registry: SourceRegistry | None = None) -> tuple[pd.DataFrame, StagingReport]:
    """California county identifiers from the Census ANSI code list."""
    registry = registry or load_registry()
    source = registry["census_county_fips"]
    report = StagingReport(source_id=source.source_id)

    raw = pd.read_csv(source.raw_path, sep="|", dtype=str)
    raw.columns = snake_case(list(raw.columns))
    report.rows_in = len(raw)

    california = raw[raw["statefp"] == CALIFORNIA_STATE_FIPS].copy()
    report.drop("not California", len(raw) - len(california))

    counties = pd.DataFrame(
        {
            "county_fips": to_county_fips(california["statefp"], california["countyfp"]),
            "county_name": california["countyname"].astype("string").str.strip(),
            "county_name_key": normalize_county_name(california["countyname"]),
        }
    ).sort_values("county_fips", ignore_index=True)

    report.rows_out = len(counties)
    report.notes.append(f"{len(counties)} California counties in the Census 2020 code list")
    return counties, report


def stage_redfin(
    registry: SourceRegistry | None = None,
    *,
    counties: pd.DataFrame | None = None,
    chunk_size: int = 500_000,
) -> tuple[pd.DataFrame, StagingReport]:
    """California county-month housing metrics from the Redfin bulk file.

    The 241 MB national file is streamed in chunks and filtered to California
    counties, monthly periods, and the All Residential aggregate before anything
    is held in memory.
    """
    registry = registry or load_registry()
    source = registry["redfin_county"]
    report = StagingReport(source_id=source.source_id)

    if counties is None:
        counties, _ = stage_counties(registry)

    usecols = [
        "PERIOD_BEGIN",
        "PERIOD_END",
        "PERIOD_DURATION",
        "REGION_TYPE",
        "REGION",
        "STATE_CODE",
        "PROPERTY_TYPE",
        "LAST_UPDATED",
        *[measure.upper() for measure in REDFIN_MEASURES],
    ]

    kept: list[pd.DataFrame] = []
    reader = pd.read_csv(
        source.raw_path,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        report.rows_in += len(chunk)
        chunk.columns = snake_case(list(chunk.columns))

        selected = chunk[
            (chunk["state_code"] == "CA")
            & (chunk["region_type"] == "county")
            & (chunk["period_duration"] == REDFIN_PERIOD_DURATION_DAYS)
            & (chunk["property_type"] == REDFIN_PROPERTY_TYPE)
        ]
        if not selected.empty:
            kept.append(selected.copy())

    if kept:
        california = pd.concat(kept, ignore_index=True)
    else:
        california = pd.DataFrame(columns=snake_case(usecols))

    report.drop(
        "not a California county-month All Residential row", report.rows_in - len(california)
    )

    california["reference_month"] = to_reference_month(california["period_begin"])
    california["county_name_key"] = normalize_county_name(california["region"])

    merged = california.merge(
        counties[["county_fips", "county_name", "county_name_key"]],
        on="county_name_key",
        how="left",
        validate="many_to_one",
    )

    unmatched = merged["county_fips"].isna()
    if unmatched.any():
        names = sorted(merged.loc[unmatched, "region"].dropna().unique())
        report.drop("county name not matched to a Census FIPS code", int(unmatched.sum()))
        report.notes.append(f"unmatched Redfin regions: {', '.join(names[:10])}")
        merged = merged[~unmatched]

    staged = merged[
        [
            "county_fips",
            "county_name",
            "reference_month",
            *REDFIN_MEASURES,
            "last_updated",
        ]
    ].copy()
    staged = staged.rename(columns={"last_updated": "redfin_last_updated"})
    staged["redfin_last_updated"] = pd.to_datetime(
        staged["redfin_last_updated"], errors="coerce", format="mixed"
    )
    staged = staged.sort_values(["county_fips", "reference_month"], ignore_index=True)

    report.rows_out = len(staged)
    if not staged.empty:
        report.notes.append(
            f"coverage {staged['reference_month'].min():%Y-%m} to "
            f"{staged['reference_month'].max():%Y-%m}, "
            f"{staged['county_fips'].nunique()} counties"
        )
    return staged, report


def stage_mortgage_rate(
    registry: SourceRegistry | None = None,
) -> tuple[pd.DataFrame, StagingReport]:
    """Weekly 30-year mortgage rate aggregated to a monthly national series.

    The weekly observations are retained alongside the monthly mean so that
    Milestone 3 can build cutoff-safe features from individual release dates
    rather than assuming a whole month was observable.
    """
    registry = registry or load_registry()
    source = registry["fred_mortgage30us"]
    report = StagingReport(source_id=source.source_id)

    weekly = pd.read_csv(source.raw_path)
    weekly.columns = snake_case(list(weekly.columns))
    report.rows_in = len(weekly)

    weekly = weekly.rename(columns={"mortgage30us": "mortgage_rate_30y"})
    weekly["observation_date"] = pd.to_datetime(weekly["observation_date"], errors="coerce")
    weekly["mortgage_rate_30y"] = pd.to_numeric(weekly["mortgage_rate_30y"], errors="coerce")

    missing = weekly["mortgage_rate_30y"].isna()
    report.drop("missing or non-numeric rate", int(missing.sum()))
    weekly = weekly[~missing]

    weekly["reference_month"] = to_reference_month(weekly["observation_date"])

    monthly = (
        weekly.groupby("reference_month", as_index=False)
        .agg(
            mortgage_rate_30y=("mortgage_rate_30y", "mean"),
            mortgage_rate_30y_last=("mortgage_rate_30y", "last"),
            mortgage_rate_weeks_observed=("mortgage_rate_30y", "size"),
            mortgage_rate_last_release=("observation_date", "max"),
        )
        .sort_values("reference_month", ignore_index=True)
    )

    report.rows_out = len(monthly)
    report.notes.append(
        f"weekly observations aggregated to {len(monthly):,} months, "
        f"{monthly['reference_month'].min():%Y-%m} to {monthly['reference_month'].max():%Y-%m}"
    )
    return monthly, report


def stage_unemployment(
    registry: SourceRegistry | None = None,
    *,
    counties: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, StagingReport]:
    """County-month unemployment from the BLS LAUS California extract."""
    registry = registry or load_registry()
    data_source = registry["bls_lau_california"]
    series_source = registry["bls_lau_series"]
    report = StagingReport(source_id=data_source.source_id)

    if counties is None:
        counties, _ = stage_counties(registry)

    # BLS pads both header and value fields with spaces, so column names cannot be
    # selected with usecols until they have been trimmed.
    series = pd.read_csv(series_source.raw_path, sep="\t", dtype=str)
    series.columns = snake_case(list(series.columns))
    series = series[["series_id", "area_type_code", "area_code", "measure_code"]]
    for column in ("series_id", "area_type_code", "area_code", "measure_code"):
        series[column] = series[column].astype("string").str.strip()

    # Area type F is "Counties and equivalents"; area codes look like CN0603700000000.
    county_series = series[
        (series["area_type_code"] == BLS_COUNTY_AREA_TYPE)
        & (series["area_code"].str.startswith("CN" + CALIFORNIA_STATE_FIPS))
        & (series["measure_code"].isin(BLS_MEASURES))
    ].copy()
    county_series["county_fips"] = county_series["area_code"].str.slice(2, 7)
    county_series["measure"] = county_series["measure_code"].map(BLS_MEASURES)

    observations = pd.read_csv(data_source.raw_path, sep="\t", dtype=str)
    observations.columns = snake_case(list(observations.columns))
    observations = observations[["series_id", "year", "period", "value"]]
    report.rows_in = len(observations)
    for column in observations.columns:
        observations[column] = observations[column].astype("string").str.strip()

    # M13 is the annual average, not a month.
    monthly_only = observations["period"].str.match(r"^M(0[1-9]|1[0-2])$")
    report.drop("annual average or non-monthly period", int((~monthly_only).sum()))
    observations = observations[monthly_only]

    merged = observations.merge(
        county_series[["series_id", "county_fips", "measure"]],
        on="series_id",
        how="inner",
        validate="many_to_one",
    )
    report.drop("not a California county unemployment series", len(observations) - len(merged))

    merged["reference_month"] = pd.to_datetime(
        merged["year"] + "-" + merged["period"].str.slice(1, 3) + "-01",
        errors="coerce",
    )
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")

    unusable = merged["reference_month"].isna() | merged["value"].isna()
    report.drop("unparseable date or suppressed value", int(unusable.sum()))
    merged = merged[~unusable]

    wide = (
        merged.pivot_table(
            index=["county_fips", "reference_month"],
            columns="measure",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    wide["county_fips"] = wide["county_fips"].astype("string")

    known = wide["county_fips"].isin(set(counties["county_fips"]))
    report.drop("county FIPS not in the Census California list", int((~known).sum()))
    wide = wide[known].sort_values(["county_fips", "reference_month"], ignore_index=True)

    report.rows_out = len(wide)
    if not wide.empty:
        report.notes.append(
            f"coverage {wide['reference_month'].min():%Y-%m} to "
            f"{wide['reference_month'].max():%Y-%m}, "
            f"{wide['county_fips'].nunique()} counties"
        )
    return wide, report


def build_staged_tables(
    registry: SourceRegistry | None = None,
    *,
    write: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[StagingReport]]:
    """Stage every source and optionally persist each table to ``data/interim/``."""
    registry = registry or load_registry()

    counties, counties_report = stage_counties(registry)
    redfin, redfin_report = stage_redfin(registry, counties=counties)
    mortgage, mortgage_report = stage_mortgage_rate(registry)
    unemployment, unemployment_report = stage_unemployment(registry, counties=counties)

    tables = {
        "counties": counties,
        "redfin": redfin,
        "mortgage_rate": mortgage,
        "unemployment": unemployment,
    }
    reports = [counties_report, redfin_report, mortgage_report, unemployment_report]

    if write:
        for name, table in tables.items():
            write_parquet(table, INTERIM_DIR / f"stg_{name}.parquet")

    return tables, reports
