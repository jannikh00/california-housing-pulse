"""The configuration registries must stay coherent with the code."""

from __future__ import annotations

from california_housing_pulse.data.columns import bounded_columns, load_columns
from california_housing_pulse.data.dictionary import render
from california_housing_pulse.data.sources import load_registry


def test_every_column_declares_source_meaning_and_unit(column_specs):
    for name, spec in column_specs.items():
        assert spec.source, f"{name} has no declared source"
        assert spec.meaning, f"{name} has no declared meaning"
        assert spec.unit, f"{name} has no declared unit"


def test_declared_sources_exist_in_the_source_registry(column_specs):
    known = set(load_registry().ids) | {"derived"}
    for name, spec in column_specs.items():
        assert spec.source in known, f"{name} cites unknown source '{spec.source}'"


def test_hard_bounds_are_never_tighter_than_plausible_bounds():
    """Plausible values must sit inside the impossible-value envelope."""
    for name, spec in bounded_columns().items():
        if spec.plausible is None:
            continue
        assert spec.hard[0] <= spec.plausible[0], f"{name}: hard min exceeds plausible min"
        assert spec.hard[1] >= spec.plausible[1], f"{name}: hard max below plausible max"


def test_bounds_are_ordered():
    for name, spec in bounded_columns().items():
        assert spec.hard[0] < spec.hard[1], f"{name}: hard bounds are inverted"


def test_source_registry_entries_are_complete():
    for source in load_registry():
        assert source.url.startswith("https://"), f"{source.source_id} must use HTTPS"
        assert source.citation, f"{source.source_id} has no citation"
        assert source.license, f"{source.source_id} has no licence note"
        assert source.manual_procedure, (
            f"{source.source_id} has no manual fallback procedure, "
            "which the hybrid acquisition strategy requires"
        )


def test_dictionary_reports_declared_and_observed_side_by_side(panel):
    markdown = render(panel)
    assert "median_sale_price" in markdown
    # Declared intent.
    assert "Declared hard" in markdown and "Declared plausible" in markdown
    # Observed measurement.
    assert "Observed dtype" in markdown and "Null rate" in markdown
    # Every documented column present in the panel is rendered.
    for name in load_columns():
        if name in panel.columns:
            assert f"`{name}`" in markdown
