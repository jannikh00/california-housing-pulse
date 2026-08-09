"""The EDA measurements must be reproducible, because the memo cites them.

These tests guard the properties a reader of the memo relies on: that class
shares are shares, that a class missing from a slice still appears as a zero
rather than vanishing from the table, and that the report and figures actually
render from a panel with the published schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from california_housing_pulse.eda import analysis, figures, report
from california_housing_pulse.features.target import add_target, load_contract, modeling_rows

MONTHS = 60


def _synthetic_panel() -> pd.DataFrame:
    """Three counties spanning every volume tier, with a real price trajectory."""
    months = pd.date_range("2019-01-01", periods=MONTHS, freq="MS")
    rng = np.random.default_rng(20260809)

    frames = []
    for fips, name, volume, noise in (
        ("06037", "Large County", 3000.0, 0.004),
        ("06055", "Mid County", 150.0, 0.02),
        ("06015", "Thin County", 14.0, 0.10),
        ("06049", "Excluded County", 3.0, 0.25),
    ):
        # A gently rising price with tier-scaled noise, so thin counties are
        # genuinely noisier — the property the inclusion rule exists to handle.
        drift = np.linspace(0, 0.35, MONTHS)
        shocks = rng.normal(0, noise, MONTHS)
        price = 400_000 * np.exp(drift + shocks)
        frames.append(
            pd.DataFrame(
                {
                    "county_fips": pd.array([fips] * MONTHS, dtype="string"),
                    "county_name": pd.array([name] * MONTHS, dtype="string"),
                    "reference_month": months,
                    "median_sale_price": price,
                    "homes_sold": np.full(MONTHS, volume),
                    "unemployment_rate": rng.uniform(3, 9, MONTHS),
                    "mortgage_rate_30y": np.linspace(3.5, 7.5, MONTHS),
                    "median_dom": rng.uniform(10, 60, MONTHS),
                }
            )
        )
    return add_target(pd.concat(frames, ignore_index=True))


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _synthetic_panel()


@pytest.fixture(scope="module")
def model(panel: pd.DataFrame) -> pd.DataFrame:
    return modeling_rows(panel)


def test_overall_prevalence_is_a_share(model: pd.DataFrame):
    shares = analysis.prevalence_overall(model)
    assert set(shares.index) == {"cooling", "stable", "heating"}
    assert shares.sum() == pytest.approx(1.0)


def test_every_class_column_survives_a_slice_that_lacks_it(model: pd.DataFrame):
    """A year with no cooling months must show 0%, not drop the column."""
    heating_only = model.loc[model["target_label"] == "heating"]
    table = analysis.prevalence_by_year(heating_only)
    for label in load_contract().label_names:
        assert label in table.columns
    assert (table["cooling"] == 0).all()
    assert (table["heating"] == 1).all()


def test_prevalence_rows_sum_to_one(model: pd.DataFrame):
    labels = list(load_contract().label_names)
    for table in (
        analysis.prevalence_by_year(model),
        analysis.prevalence_by_tier(model),
        analysis.prevalence_by_county(model),
    ):
        assert table[labels].sum(axis=1).round(9).eq(1.0).all()


def test_excluded_counties_are_reported_separately(panel: pd.DataFrame, model: pd.DataFrame):
    """The exclusion must be evidenced, so its dispersion is measured too."""
    assert "Excluded County" not in set(model["county_name"])
    excluded = analysis.excluded_county_dispersion(panel)
    assert set(excluded["county_name"]) == {"Excluded County"}
    assert excluded["dg_sd"].iloc[0] > 0


def test_dispersion_falls_as_county_volume_rises(model: pd.DataFrame):
    """The core EDA finding — it should hold on data built to contain it."""
    table = analysis.county_dispersion(model).set_index("county_name")
    assert table.loc["Thin County", "dg_sd"] > table.loc["Mid County", "dg_sd"]
    assert table.loc["Mid County", "dg_sd"] > table.loc["Large County", "dg_sd"]


def test_describe_target_reports_the_window_and_spread(model: pd.DataFrame):
    dist = analysis.describe_target(model)
    assert dist.rows == int(model["target_dg"].notna().sum())
    assert dist.std > 0
    assert dist.start <= dist.end
    assert set(dist.quantiles) == {0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999}


def test_seasonality_covers_every_calendar_month_present(model: pd.DataFrame):
    table = analysis.seasonality(model)
    assert len(table) == 12
    assert table["rows"].sum() == int(model["target_dg"].notna().sum())
    # z is the mean in standard-error units; it must be finite where rows exist.
    assert np.isfinite(table["z"]).all()


def test_missingness_and_outliers_use_the_column_registry(panel: pd.DataFrame):
    missing = analysis.missingness(panel)
    assert "column" in missing.columns
    assert (missing["null_rate"].diff().dropna() <= 0).all(), "worst-first ordering"

    outliers = analysis.feature_outliers(panel)
    # median_dom is bounded 0–365 in the registry and the fixture stays inside it.
    assert "median_dom" not in set(outliers.get("column", []))


def test_regime_shift_scan_reports_before_and_after(model: pd.DataFrame):
    shifts = analysis.regime_shifts(model, window=6, top=3)
    if len(shifts):
        assert {"month", "mean_before", "mean_after", "shift"} <= set(shifts.columns)
        assert np.allclose(shifts["shift"], shifts["mean_after"] - shifts["mean_before"])
        # Detections must not cluster inside one window.
        months = sorted(shifts["month"])
        for earlier, later in zip(months, months[1:], strict=False):
            assert (later - earlier).days >= 30 * 6


def test_representative_counties_picks_one_per_tier(model: pd.DataFrame):
    chosen = analysis.representative_counties(model)
    assert chosen
    assert "excluded" not in chosen, "excluded counties are never representative"
    for _tier, (fips, name) in chosen.items():
        assert fips in set(model["county_fips"])
        assert isinstance(name, str)


def test_report_renders_every_section(panel: pd.DataFrame):
    markdown = report.render(panel, figure_paths=[])
    for heading in (
        "## The frozen target",
        "## Continuous target distribution",
        "## Directional class prevalence",
        "## County inclusion rule",
        "## Coverage and missingness",
        "## Seasonality",
        "## Structural breaks",
    ):
        assert heading in markdown
    # The report must cite the frozen contract rather than restating it loosely.
    assert load_contract().describe() in markdown


def test_figures_are_written(panel: pd.DataFrame, model: pd.DataFrame, tmp_path):
    paths = figures.build_all(panel, model, tmp_path)
    assert len(paths) == 5
    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == [
        "fig01_target_distribution.png",
        "fig02_class_prevalence_by_year.png",
        "fig03_county_time_series.png",
        "fig04_volume_vs_dispersion.png",
        "fig05_seasonality.png",
    ]
    assert all(p.stat().st_size > 0 for p in tmp_path.glob("*.png"))
