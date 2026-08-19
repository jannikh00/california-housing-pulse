"""Tests for the Milestone 4 presentation layer.

The reporting package makes claims *about* results rather than producing them,
so these tests are aimed at the two ways that goes wrong:

*A number in the prose stops matching the table.* The results document asserts
that nothing on it is transcribed by hand. That is only true if it stays true,
so :func:`test_the_report_prose_tracks_the_numbers_it_describes` re-renders
against altered predictions and checks the prose moved with them.

*An ordering silently becomes alphabetical.* ``groupby`` sorts its keys, and
"large, mid, small, thin" is a plausible-looking order that reverses the finding
the tier tables exist to show.

Everything runs on a synthetic prediction frame with the real schema. No test
here needs the 241 MB download or a fitted model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from california_housing_pulse.reporting import figures, report, results

# (fips, name, tier, target_dg, base effect b(t), ridge predicted_dg)
#
# ``b`` is chosen so the forward growth f = Δg + b sums to zero across the eight
# rows. The base-effect baseline then predicts Δg = 0 − b, which is what the real
# one does: forecast the climatological forward growth and subtract the observable
# base. Its predicted f is therefore constant, exactly as in the shipped model.
COUNTIES = [
    (
        "06001",
        "Alameda County",
        "large",
        [4.0, -4.0, 1.0, -1.0],
        [-2.0, 2.0, 0.0, 0.0],
        [3.0, -3.0, 1.0, -1.0],
    ),
    (
        "06003",
        "Alpine County",
        "thin",
        [8.0, -8.0, 2.0, -2.0],
        [-4.0, 4.0, -1.0, 1.0],
        [4.0, -4.0, 0.0, 0.0],
    ),
]

TAU = 2.0


def _label(value: float) -> str:
    if value <= -TAU:
        return "cooling"
    return "heating" if value >= TAU else "stable"


@pytest.fixture
def predictions() -> pd.DataFrame:
    """Eight test rows per model, with hand-chosen errors.

    Ridge halves its error in the large county and misses badly in the thin one,
    which is the real pattern in miniature and makes the tier tables meaningful.
    """
    months = pd.date_range("2025-03-01", periods=4, freq="MS")
    rows = []
    for fips, name, tier, target, bases, ridge in COUNTIES:
        for month, actual, base, predicted in zip(months, target, bases, ridge, strict=True):
            common = {
                "county_fips": fips,
                "county_name": name,
                "reference_month": month,
                "volume_tier": tier,
                "split": "test",
                "price_smoothed__log_diff3_o9": base,
                "target_dg": actual,
                "target_label": _label(actual),
            }
            for model, value in (
                ("ridge", predicted),
                ("base_effect", -base),
                ("zero_change", 0.0),
                ("persistence", -actual),
                ("mean_reversion", actual / 4),
            ):
                rows.append(
                    {
                        **common,
                        "model": model,
                        "predicted_dg": value,
                        "predicted_label": _label(value),
                    }
                )
            classifiers = (
                ("majority_class", "stable"),
                ("multinomial_logistic", _label(actual)),
            )
            for model, label in classifiers:
                rows.append(
                    {**common, "model": model, "predicted_dg": None, "predicted_label": label}
                )
    frame = pd.DataFrame(rows)
    frame["county_fips"] = frame["county_fips"].astype("string")
    return frame


def test_headline_scores_skill_against_the_base_effect_not_zero_change(predictions):
    """The distinction the whole milestone rests on, checked by hand.

    Absolute errors: ridge 1,1,0,0,4,4,2,2 -> MAE 1.75; base_effect 2,2,1,1,4,4,1,1
    -> 2.00; zero_change equals |target| -> 3.75. Quoting the zero-change number
    would advertise 53% skill for a model that earned 13%.
    """
    table = results.headline(predictions).set_index("model")

    assert table.loc["ridge", "mae"] == pytest.approx(1.75)
    assert table.loc["base_effect", "mae"] == pytest.approx(2.00)
    assert table.loc["zero_change", "mae"] == pytest.approx(3.75)

    assert table.loc["ridge", "skill_vs_base_effect"] == pytest.approx(1 - 1.75 / 2.00)
    assert table.loc["ridge", "skill_vs_zero_change"] == pytest.approx(1 - 1.75 / 3.75)
    # The honest number is the smaller one; a report must never silently swap them.
    assert table.loc["ridge", "skill_vs_base_effect"] < table.loc["ridge", "skill_vs_zero_change"]


def test_the_base_effect_baseline_knows_nothing_about_forward_growth(predictions):
    """``corr_forward`` near zero for base_effect is what validates the diagnostic.

    base_effect predicts Δg = c − b for a constant c, so predicted f = Δg + b is
    constant and correlates with nothing. If this ever became large, the metric
    would be measuring the base effect rather than removing it.
    """
    frame = results.forward_skill(predictions, "base_effect")
    assert frame["predicted_f"].std() == pytest.approx(0.0)
    # A constant predictor correlates with nothing, and the metric says so
    # rather than dividing by a zero standard deviation and returning a bare nan.
    table = results.headline(predictions).set_index("model")
    assert pd.isna(table.loc["base_effect", "corr_forward"])

    # f = Δg + b, exactly, for a model that does vary.
    ridge = results.forward_skill(predictions, "ridge")
    source = predictions.loc[predictions["model"] == "ridge"]
    expected = source["target_dg"] + source["price_smoothed__log_diff3_o9"]
    assert ridge["actual_f"].to_numpy() == pytest.approx(expected.to_numpy())


def test_tiers_are_ordered_thin_to_large_not_alphabetically(predictions):
    """Alphabetical order reverses the gradient the table exists to show."""
    table = results.by_tier(predictions, "ridge")
    present = list(table["volume_tier"])

    assert present == ["thin", "large"]
    # And the finding survives the ordering: thin is the harder tier on magnitude.
    assert (
        table.set_index("volume_tier").loc["thin", "mae"]
        > table.set_index("volume_tier").loc["large", "mae"]
    )


def test_county_errors_are_ranked_worst_first_with_readable_names(predictions):
    table = results.by_county(predictions, "ridge")

    assert list(table["county"]) == ["Alpine", "Alameda"]
    assert table.loc[0, "mae"] == pytest.approx(3.0)
    assert table.loc[1, "mae"] == pytest.approx(0.5)
    assert (table["n"] == 4).all()


def test_concise_table_reads_naive_first_then_learned(predictions):
    """Reading order is an argument: the bar comes before what clears it."""
    order = list(report.concise_table(predictions)["model"])

    assert order.index("base_effect") < order.index("ridge")
    assert order.index("majority_class") < order.index("multinomial_logistic")


def test_missing_predictions_name_the_command_that_produces_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="chp baselines"):
        results.load_predictions(tmp_path / "absent.parquet")


def test_figures_are_written(predictions, tmp_path):
    paths = figures.build_all(predictions, tmp_path)

    assert len(paths) == 4
    for path in paths:
        assert (tmp_path / path.rsplit("/", 1)[-1]).stat().st_size > 0


def test_the_report_prose_tracks_the_numbers_it_describes(predictions, tmp_path, monkeypatch):
    """The document claims nothing on it is transcribed. This is that claim, tested.

    Doubling every ridge error must move the MAE quoted in the can/cannot-claim
    block, not only the number in the table above it — a prose constant left
    behind would outlive the result it rests on.
    """
    destination = tmp_path / "results.md"
    monkeypatch.setattr(report, "RESULTS_REPORT", destination)

    report.build_report(predictions, directory=tmp_path, resamples=25)
    before = destination.read_text()

    worse = predictions.copy()
    is_ridge = worse["model"] == "ridge"
    worse.loc[is_ridge, "predicted_dg"] = (
        worse.loc[is_ridge, "target_dg"]
        + (worse.loc[is_ridge, "predicted_dg"] - worse.loc[is_ridge, "target_dg"]) * 2
    )
    report.build_report(worse, directory=tmp_path, resamples=25)
    after = destination.read_text()

    assert "1.750" in before and "3.500" in after
    # The claims block, not just the results table, carries the live number.
    claims_before = before.split("What this project can and cannot claim")[1]
    claims_after = after.split("What this project can and cannot claim")[1]
    assert claims_before != claims_after
    assert "3.500 pp" in claims_after
