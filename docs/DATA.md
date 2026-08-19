# Data

What the panel contains, how it is built and validated, and what the exploratory
analysis decided. Modelling choices live in [METHOD.md](METHOD.md); leakage
enforcement in [LEAKAGE.md](LEAKAGE.md).

## Sources

| Source | Role | Licence note |
|---|---|---|
| [Redfin Data Center](https://www.redfin.com/news/data-center/) | Target price series plus sales, inventory, days on market, sale-to-list | Use with attribution; cite and link on first reference |
| [Freddie Mac PMMS via FRED (`MORTGAGE30US`)](https://fred.stlouisfed.org/series/MORTGAGE30US) | National 30-year mortgage rate | Copyrighted, "as is", citation required |
| [BLS Local Area Unemployment Statistics](https://www.bls.gov/lau/) | County unemployment | Public domain; citation requested |
| [Census county/ANSI code list](https://www.census.gov/library/reference/code-lists/ansi.html) | County FIPS crosswalk | Federal reference data |

Full citations, retrieval timestamps and file hashes are recorded per source in
[DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## The panel

| | |
|---|---|
| Grain | one row per `(county_fips, reference_month)` |
| Rows | 10,034 |
| Counties | 58 (all California counties) |
| Coverage | 2012-01 → 2026-05, 173 consecutive months |
| Columns | 36, every one documented in `configs/columns.yaml` |
| Checks | 17 automated (0 errors, 5 warnings) |

### Modelling table

`data/processed/target_panel.parquet` holds the rows whose county clears the
volume floor *and* whose label is observable.

| | |
|---|---|
| Rows | 8,322 of 10,034 |
| Counties | 54 of 58 |
| Labelled window | 2013-03 → 2026-02 |
| Class prevalence | 36.6% cooling · 31.2% stable · 32.2% heating |
| Δg spread | mean −0.35 pp, standard deviation 9.71 pp |

`data/processed/features.parquet` adds the 61 leakage-safe features, and
`data/processed/predictions.parquet` holds one row per model × county × month.

## How the foundation works

**Acquisition is hybrid.** `chp fetch` downloads each declared source and records
its URL, SHA-256, byte size, retrieval time and the server's `Last-Modified` into
`data/raw/manifest.json`. If a host blocks or moves a URL, the failure prints
that source's documented manual procedure, and `chp register <source> <file>`
adopts a hand-downloaded file into the same manifest — so both paths produce
identical provenance records. `chp verify` re-hashes the snapshots against the
manifest.

**The panel is built on a complete spine.** All 58 counties are crossed with
every month in the coverage window *before* any source is joined. A county-month
with no Redfin observation is therefore a visible row with missing values, not an
absent row, which is what makes coverage gaps countable rather than invisible.
Join accounting is reported for every source.

**Validation separates impossible from merely extreme.**
[`configs/columns.yaml`](../configs/columns.yaml) declares two bound families per
column. Violating the `hard` bounds means a parsing or unit error and fails the
build. Violating the `plausible` bounds is reported as a warning: values like 466
months of supply in a county that sold two homes are real thin-market arithmetic,
and they are the evidence behind the county inclusion rule.

**The data dictionary cannot drift.** `configs/columns.yaml` declares intent
(source, meaning, unit, bounds); the generator measures dtype, null rate and
min/median/max from the built panel and publishes them side by side. An
ERROR-level check asserts that the panel's columns and the registry's entries
match exactly in both directions, so an undocumented column fails the build.

**The target is a contract, not a parameter.** Two ERROR-level checks guard its
timing: `target_leadin_months_unlabelled` fails if a growth value appears before
the 14-month lead-in has elapsed, and `target_horizon_within_coverage` fails if
any labelled row would resolve past the end of the data.

## What the exploratory analysis decided

Five findings from [`reports/eda.md`](../reports/eda.md), each driving a decision
that the modelling then had to honour:

- **Target volatility tracks market thinness, not the market.** Δg standard
  deviation runs from 20.6 pp in counties selling 10–25 homes a month to 3.8 pp
  in those selling over 500. Modoc County sells a median of *one*.
  → four counties excluded; volume tier retained as an evaluation dimension.
- **"Stable" is partly a volume artifact.** Large counties are stable 47.7% of
  the time, thin counties 9.1%. → report metrics within volume tier, never only
  pooled. This is the finding that the results later confirmed in the sharpest
  possible form: the tier with the worst magnitude error has among the best
  directional scores.
- **Regime shifts dwarf anything a model will do.** 2022 is 70.5% cooling; 2023
  is 54.8% heating. The sharpest turn is 2023-02, a +7.44 pp shift in the mean
  target. → per-period reporting, and rate *change* features rather than levels.
  This finding also predicted, correctly, that persistence would fail badly.
- **There is no seasonality left.** No calendar month's mean sits two standard
  errors from zero — the year-over-year construction already differences it out.
  → no calendar features in the baseline.
- **The lead-in and horizon cost 17% of the panel, predictably.** → split
  boundaries come from the labelled window, which ends 2026-02.

## Known data limitations

- The Redfin bulk file was last refreshed **2 June 2026**, so county coverage
  ends at **2026-05**, which caps the test window.
- **Redfin revises recent months** and carries an approximately four-week curing
  window. Historical vintages cannot be reconstructed from the current bulk
  download, so the most recent test labels may shift in a later vintage.
- **`price_drops` is missing for 23.9% of county-months.** Redfin documents some
  listing-history metrics as beginning around 2016. Any feature built on it
  silently restricts the training window, so it is **excluded from the feature
  set** rather than imputed.
- **113 county-months (1.1%) have no Redfin observation** and 58 (0.6%) no BLS
  unemployment value. The unemployment gap is exactly **2025-10** across all 58
  counties — one missing release, not a county-specific problem — and it falls
  inside the test window. It is carried forward within each county and flagged by
  the `unemployment_imputed` feature, which the ablation scores at exactly zero,
  as one imputed month should be.

## Generated artifacts

All are rebuilt by `make all`. Do not edit them by hand.

| File | Produced by |
|---|---|
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | `chp build` |
| [`reports/data_quality.md`](../reports/data_quality.md) | `chp build` |
| [`reports/feature_availability.md`](../reports/feature_availability.md) | `chp features` |
| [`reports/eda.md`](../reports/eda.md) | `chp eda` |
| [`reports/baselines.md`](../reports/baselines.md) | `chp baselines` |
| [`reports/results.md`](../reports/results.md) | `chp report` |
