# California Housing Pulse

Forecasting the **three-month change in California county price-growth momentum**:
for each county and reference month, predict the signed change in year-over-year
smoothed median-sale-price growth over the next three months, and translate that
value into a **heating / stable / cooling** label.

> **Status:** Milestone 2 (target construction and decision-oriented EDA) complete.
> The target is frozen; baselines arrive in Milestone 3 and the full narrative in Milestone 4.

## The frozen target

For a county at reference month *t*:

```text
price_smoothed(t) = mean of median_sale_price over [t-2, t]
growth_yoy(t)     = 100 · ln( price_smoothed(t) / price_smoothed(t-12) )      [pp]
target_dg(t)      = growth_yoy(t+3) − growth_yoy(t)                          [pp]
target_label(t)   = cooling if Δg ≤ −2 · stable if −2 < Δg < 2 · heating if Δg ≥ 2
```

The first two lines use only months up to *t*, so `growth_yoy` is available at
prediction time. The third reaches forward by design: it is the label, observable
only at *t+3*, and no feature may derive from it.

The definition is frozen in [`configs/target.yaml`](configs/target.yaml) and a test
fails if any value drifts. Counties selling fewer than 10 homes in a median month
are excluded from modelling but remain visible in the panel — see
[`docs/MILESTONE_2_EDA_MEMO.md`](docs/MILESTONE_2_EDA_MEMO.md) for the evidence.

## Quickstart

```bash
make setup   # create .venv and install dependencies
make all     # fetch raw sources, rebuild the panel, render the EDA, run the tests
```

`make all` is the single documented rebuild. The Makefile contains no pipeline
logic — it sets the environment and delegates to the CLI — so the direct
invocation is equivalent and is the fallback on Windows:

```bash
UV_NO_EDITABLE=1 uv run --reinstall-package california-housing-pulse chp all
```

> **Why `UV_NO_EDITABLE`:** uv's editable install writes a `.pth` file without a
> trailing newline, which Python's `site` machinery ignores, so the package
> silently fails to import. A non-editable install avoids this;
> `--reinstall-package` keeps the installed copy in step with the working tree.

### Individual commands

| Command | Purpose |
|---|---|
| `make fetch` | Download raw sources into `data/raw/` and record provenance |
| `make build` | Rebuild staged tables, the panel, the target, the quality report, and the dictionary |
| `make eda` | Render `reports/eda.md` and the five Milestone 2 figures |
| `make verify` | Re-hash raw snapshots against `data/raw/manifest.json` |
| `make test` | Run the test suite |
| `make lint` | Run ruff |

## Current panel

| | |
|---|---|
| Grain | one row per `(county_fips, reference_month)` |
| Rows | 10,034 |
| Counties | 58 (all California counties) |
| Coverage | 2012-01 → 2026-05, 173 consecutive months |
| Columns | 36, every one documented in `configs/columns.yaml` |
| Checks | 17 automated (0 errors, 5 warnings) |

### Modelling table

`data/processed/target_panel.parquet` is the subset Milestone 3 may train on:
rows whose county clears the volume floor *and* whose label is observable.

| | |
|---|---|
| Rows | 8,322 of 10,034 |
| Counties | 54 of 58 |
| Labelled window | 2013-03 → 2026-02 |
| Class prevalence | 36.6% cooling · 31.2% stable · 32.2% heating |
| Δg spread | mean −0.35 pp, standard deviation 9.71 pp |

Generated artifacts: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md),
[`reports/data_quality.md`](reports/data_quality.md), and
[`reports/eda.md`](reports/eda.md). All are rebuilt by `make all`; do not edit them
by hand.

## How the data foundation works

**Acquisition is hybrid.** `chp fetch` downloads each declared source and records
its URL, SHA-256, byte size, retrieval time, and the server's `Last-Modified`
into `data/raw/manifest.json`. If a host blocks or moves a URL, the failure
prints that source's documented manual procedure and `chp register <source> <file>`
adopts a hand-downloaded file into the same manifest — so both paths produce
identical provenance records.

**The panel is built on a complete spine.** All 58 counties are crossed with every
month in the coverage window *before* any source is joined. A county-month with
no Redfin observation is therefore a visible row with missing values, not an
absent row, which is what makes coverage gaps countable rather than invisible.
Join accounting is reported for every source.

**Validation separates impossible from merely extreme.** `configs/columns.yaml`
declares two bound families per column. Violating the `hard` bounds means a
parsing or unit error and fails the build. Violating the `plausible` bounds is
reported as a warning: values like 466 months of supply in a county that sold two
homes are real thin-market arithmetic, and they are the evidence Milestone 2 needs
for its county inclusion rule.

**The data dictionary cannot drift.** `configs/columns.yaml` declares intent
(source, meaning, unit, bounds); the generator measures dtype, null rate, and
min/median/max from the built panel and publishes them side by side. An
ERROR-level check asserts the panel's columns and the registry's entries match
exactly in both directions, so an undocumented column fails the build.

**The target is a contract, not a parameter.** `configs/target.yaml` holds the
frozen definition and a test asserts every value in it, so editing the config to
improve a score also breaks the build. Two ERROR-level checks guard the timing:
`target_leadin_months_unlabelled` fails if a growth value appears before the
14-month lead-in has elapsed, and `target_horizon_within_coverage` fails if any
labelled row would resolve past the end of the data.

## What the EDA decided

Five findings from [`reports/eda.md`](reports/eda.md), each driving a decision
recorded in [the memo](docs/MILESTONE_2_EDA_MEMO.md):

- **Target volatility tracks market thinness, not the market.** Δg standard
  deviation runs from 20.6 pp in counties selling 10–25 homes a month to 3.8 pp in
  those selling over 500. Modoc County sells a median of *one*. → four counties
  excluded; volume tier retained as an evaluation dimension.
- **"Stable" is partly a volume artifact.** Large counties are stable 47.7% of the
  time, thin counties 9.1%. → report metrics within volume tier, not only pooled.
- **Regime shifts dwarf anything a model will do.** 2022 is 70.5% cooling; 2023 is
  54.8% heating. The sharpest turn is 2023-02, a +7.44 pp shift in the mean target.
  → per-period reporting, and rate *change* features rather than levels.
- **There is no seasonality left.** No calendar month's mean sits two standard
  errors from zero — the year-over-year construction already differences it out.
  → no calendar features in the baseline.
- **The lead-in and horizon cost 17% of the panel, predictably.** → split
  boundaries come from the labelled window, which ends 2026-02.

## Repository layout

```text
Makefile                 thin wrapper: environment setup and target composition only
configs/
    sources.yaml         source registry: URLs, licences, citations, manual fallbacks
    columns.yaml         column registry: source, meaning, unit, hard/plausible bounds
    target.yaml          the frozen target contract, guarded by a test
data/raw/                upstream snapshots (gitignored; manifest.json is committed)
data/snapshots/          small California-only slices, committed for reproducibility
data/interim/            normalized staged tables (rebuildable)
data/processed/          joined county-month panel + target panel (rebuildable)
notebooks/               thin: calls project code, holds no pipeline logic
src/california_housing_pulse/
    cli.py               chp fetch / register / verify / build / eda / test / all
    paths.py             project root located by marker file, not by fixed depth
    io.py                typed Parquet helpers that protect the FIPS string contract
    data/                sources, manifest, fetch, normalize, staging, panel,
                         validate, dictionary, pipeline
    features/            target.py executes the frozen contract; features in Milestone 3
    eda/                 analysis (measures), figures (draws), report (renders)
    modeling/            baselines and models (Milestone 3+)
    evaluation/          metrics and error analysis (Milestone 3+)
tests/                   61 tests, no network or large-file dependency
reports/                 generated data-quality and EDA reports, plus figures
```

## Data sources

| Source | Role | Licence note |
|---|---|---|
| [Redfin Data Center](https://www.redfin.com/news/data-center/) | Target price series plus sales, inventory, days on market, sale-to-list | Use with attribution; cite and link on first reference |
| [Freddie Mac PMMS via FRED (`MORTGAGE30US`)](https://fred.stlouisfed.org/series/MORTGAGE30US) | National 30-year mortgage rate | Copyrighted, "as is", citation required |
| [BLS Local Area Unemployment Statistics](https://www.bls.gov/lau/) | County unemployment | Public domain; citation requested |
| [Census county/ANSI code list](https://www.census.gov/library/reference/code-lists/ansi.html) | County FIPS crosswalk | Federal reference data |

Full citations, retrieval timestamps, and file hashes are recorded per source in
[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## Known limitations

- The Redfin bulk file was last refreshed **2 June 2026**, so county coverage ends
  at **2026-05**. This caps the untouched test window in Milestone 3.
- Redfin values are revised and carry an approximately four-week curing window.
  Historical vintages cannot be reconstructed from the current bulk download, so
  the prediction-time contract relies on documented release lags rather than true
  point-in-time snapshots. This is disclosed rather than hidden.
- `price_drops` is missing for 23.9% of county-months; Redfin documents some
  listing-history metrics as beginning around 2016. Any feature built on it
  silently restricts the training window.
- 113 county-months (1.1%) have no Redfin observation and 58 (0.6%) no BLS
  unemployment value. The unemployment gap is exactly **2025-10** across all 58
  counties — one missing release, not a county-specific problem — and it falls
  inside the prospective test window.
- **Four counties are excluded from modelling** (Modoc, Alpine, Sierra, Inyo): they
  sell 1–5 homes in a median month, which cannot support a meaningful median price.
  Their rows stay in the panel flagged by `is_included`. The floor of 10 median
  monthly sales is a judgement — evidenced, published, and deliberately set before
  any test data was touched.
- **The six thinnest retained counties will drag the headline error metric.** Their
  Δg standard deviations run 13–28 pp against 3.8 pp for the largest counties. They
  are 10.7% of modelling rows; metrics are reported by volume tier so this stays
  visible rather than averaged away.
