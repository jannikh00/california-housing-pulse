# California Housing Pulse

Forecasting the **three-month change in California county price-growth momentum**:
for each county and reference month, predict the signed change in year-over-year
smoothed median-sale-price growth over the next three months, and translate that
value into a **heating / stable / cooling** label.

> **Status:** Milestone 3 (leakage-safe features, temporal split, baselines) complete.
> The target, the feature specification and the split are all frozen and guarded by
> tests. Ridge cuts MAE **18.6%** against the honest naive baseline and multinomial
> logistic reaches **macro-F1 0.638** against a majority-class floor of 0.179 — see
> [Baseline results](#baseline-results). The polished narrative and visuals arrive
> in Milestone 4.

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
make all     # fetch, rebuild the panel, build features, render the EDA,
             #   fit and evaluate the baselines, run the tests
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
| `make features` | Build the feature matrix, audit publication lags, render `reports/feature_availability.md` |
| `make eda` | Render `reports/eda.md` and the five Milestone 2 figures |
| `make baselines` | Fit the baselines and models, evaluate, render `reports/baselines.md` |
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

Generated artifacts: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md),
[`reports/data_quality.md`](reports/data_quality.md),
[`reports/eda.md`](reports/eda.md),
[`reports/feature_availability.md`](reports/feature_availability.md), and
[`reports/baselines.md`](reports/baselines.md). All are rebuilt by `make all`; do
not edit them by hand.

## Leakage controls

The project's central claim is that no feature uses information unavailable when
the forecast was made. Four mechanisms enforce it, each catching failures the
others cannot.

**Every row carries a cutoff.** `prediction_as_of` is the 15th of the month after
the reference month. A feature is safe when every input it reads was published on
or before that timestamp.

**Transforms are backward-only.** There is no forward shift anywhere in
`features/transforms.py`; the shift helper raises on a negative offset, so the
failure cannot be expressed rather than merely being absent.

**Truncation replay.** Every feature is recomputed on a panel cut off at month
*t* and must reproduce the full-panel value exactly. If any transform reached
forward, hiding the future would change the answer.

**A publication audit, which found a real problem.** Each source declares both
the lag *we chose* and how long the publisher actually takes, and the audit
checks the first against the second. BLS publishes county unemployment six to
eight weeks after the reference month, but the cutoff falls on the 15th of the
following month — so unemployment for month *t* is **not knowable** at *t*, and a
one-month lag is about a week short. `bls_lau_california` therefore carries a
**two-month** release lag, and `unemployment_rate__diff12` reads *t−14* → *t−2*.
A test forces the lag back to one month and asserts the audit objects, so the
check cannot degrade into restating its own configuration.

All 61 features currently clear the cutoff by at least 8 days.
[`reports/feature_availability.md`](reports/feature_availability.md) states the
oldest month every feature reads, generated from
[`configs/features.yaml`](configs/features.yaml) so the published table cannot
drift from the code.

## The split

Frozen in [`configs/split.yaml`](configs/split.yaml) on 11 August 2026 — before
any model was fitted — and asserted by a guard test, including the row counts.

| Split | Rows | Window | cooling / stable / heating |
|---|---|---|---|
| train | 6,271 | 2014-03 → 2023-11 | 35.3% / 31.5% / 33.2% |
| validation | 486 | 2024-03 → 2024-11 | 37.4% / 33.3% / 29.2% |
| test | 648 | 2025-03 → 2026-02 | 36.7% / 36.4% / 26.9% |

All 54 counties appear in all three splits.

**Each boundary carries a three-month embargo.** The label at month *t* is only
observable at *t+3*, so with contiguous windows a training row at 2024-02 would
carry an outcome that resolves inside the validation period. That is not feature
leakage — it is leakage into *model selection*. Three months at each boundary are
assigned to no split and used by nothing, costing 324 of 8,053 otherwise usable
rows.

Eligibility begins at 2014-03, the first month with enough history for the
deepest feature. It is a **date rather than a per-row completeness test**: since
Redfin's coverage of small counties is sporadic and one absent month blanks a
twelve-month rolling window, requiring every feature to be present deleted Del
Norte from training entirely and left Colusa with 22 of 117 months. Residual gaps
— 0.41% of feature cells — are imputed inside the model pipeline on training
statistics only.

## Baseline results

Test split, scored once after the pipeline was frozen. Full detail, including
per-tier and per-period breakdowns and confusion matrices, is in
[`reports/baselines.md`](reports/baselines.md).

| Model | MAE (pp) | vs base effect | macro-F1 | vs majority |
|---|---|---|---|---|
| **ridge** | **3.823** | **+18.6%** | 0.569 | +0.390 |
| base_effect | 4.699 | — | 0.515 | +0.336 |
| mean_reversion | 5.037 | −7.2% | 0.368 | +0.189 |
| zero_change | 5.434 | −15.6% | 0.178 | −0.001 |
| persistence | 9.257 | −97.0% | 0.270 | +0.091 |
| **multinomial_logistic** | — | — | **0.638** | **+0.459** |
| majority_class | — | — | 0.179 | — |

95% block-bootstrap intervals, resampling whole counties: ridge MAE
**[3.230, 4.490]**, logistic macro-F1 **[0.594, 0.681]**.

### Why the baseline is "base effect" and not zero-change

Expanding the frozen target shows it decomposes *exactly*:

```text
Δg = g(t+3) − g(t)
   = 100·ln( P(t+3) / P(t) )  −  100·ln( P(t−9) / P(t−12) )
   =        f(t)              −        b(t)
```

`f(t)` is the forward three-month growth and is genuinely unknown. `b(t)` — the
**base effect** — is growth that already happened between *t−12* and *t−9*, and
is fully observable when the forecast is made. The identity holds to machine
precision over 8,727 rows and is asserted by a test.

On the test window `corr(Δg, −b) = 0.707`, so roughly half the target's variance
was knowable in advance for reasons that have nothing to do with forecasting a
housing market. A predictor that only subtracts `b(t)` beats zero-change by
**13.5%** while knowing nothing at all. Quoting a result against zero-change
therefore credits the model for arithmetic it did not perform.

The distinction is measurable rather than rhetorical. `corr_forward` — the
correlation with the genuinely unknown part — is **0.017** for the base-effect
baseline and **0.580** for ridge. That is where the real forecasting claim lives.

### Three results worth stating plainly

- **Persistence is worse than doing nothing** (MAE 9.257 against zero-change's
  5.434). Milestone 2's regime reversals arriving in the results; the opposite
  bet, mean reversion, fitted *k* = +0.278 and beat both.
- **Error tracks county thinness, but direction inverts.** Ridge MAE runs 8.474
  in thin counties against 2.147 in large ones — yet thin counties score the
  *highest* directional accuracy and large counties the lowest macro-F1, because
  the fixed ±2 pp band makes `stable` rare in thin markets and common in large
  ones. Metrics are reported by tier for exactly this reason.
- **Both models under-call cooling** in a cooling test window: precision 0.822,
  recall 0.445. Recorded as a limitation rather than tuned away — adjusting
  thresholds after seeing the test set is what the frozen contracts prevent.

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

**So are the features and the split.** `configs/features.yaml` declares every
feature, each source's publication lag, and the missing-data policy; the
transforms live in Python because they are logic, but which features exist is a
diffable specification, and the availability table is generated from it.
`configs/split.yaml` holds the boundaries and the expected row counts, and its
guard test fails if either moves. If a newer Redfin file shifts the panel, the
split must be recomputed *before* the test set is read again — never after.

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
    features.yaml        the feature spec: publication lags, missing-data policy,
                         and every declared feature — guarded by an audit
    split.yaml           the frozen split: boundaries, embargo, expected row counts
data/raw/                upstream snapshots (gitignored; manifest.json is committed)
data/snapshots/          small California-only slices, committed for reproducibility
data/interim/            normalized staged tables (rebuildable)
data/processed/          panel, target panel, features, predictions (rebuildable)
models/                  baseline_config.json: chosen hyperparameters and the full scan
notebooks/               thin: calls project code, holds no pipeline logic
src/california_housing_pulse/
    cli.py               chp fetch / register / verify / build / features / eda /
                         baselines / test / all
    paths.py             project root located by marker file, not by fixed depth
    io.py                typed Parquet helpers that protect the FIPS string contract
    data/                sources, manifest, fetch, normalize, staging, panel,
                         validate, dictionary, pipeline
    features/            target.py executes the frozen contract; spec declares,
                         transforms computes, build assembles
    eda/                 analysis (measures), figures (draws), report (renders)
    modeling/            split, baselines, models, run
    evaluation/          metrics, bootstrap, report
tests/                   137 tests, no network or large-file dependency
reports/                 generated data-quality, EDA, feature-availability and
                         baseline reports, plus figures
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
  silently restricts the training window, so it is **excluded from the feature
  set** rather than imputed.
- 113 county-months (1.1%) have no Redfin observation and 58 (0.6%) no BLS
  unemployment value. The unemployment gap is exactly **2025-10** across all 58
  counties — one missing release, not a county-specific problem — and it falls
  inside the test window. It is carried forward within each county and flagged by
  the `unemployment_imputed` feature, which the ablation scores at exactly zero,
  as one imputed month should be.
- **Four counties are excluded from modelling** (Modoc, Alpine, Sierra, Inyo): they
  sell 1–5 homes in a median month, which cannot support a meaningful median price.
  Their rows stay in the panel flagged by `is_included`. The floor of 10 median
  monthly sales is a judgement — evidenced, published, and deliberately set before
  any test data was touched.
- **The six thinnest retained counties drag the headline error metric.** Their
  Δg standard deviations run 13–28 pp against 3.8 pp for the largest counties. They
  are 10.7% of modelling rows; metrics are reported by volume tier so this stays
  visible rather than averaged away. Measured: ridge MAE 8.474 in the thin tier
  against 2.147 in the large one.
- **Results rest on one test window in one regime.** 2025-03 → 2026-02 is
  cooling-leaning, and the model was trained mostly on warmer regimes — which is
  the likely reason it under-calls cooling. Rolling-origin validation across
  regimes is Milestone 5, and until then a single held-out window is the honest
  description of the evidence.
- **Validation is thin after the embargo:** 486 rows across nine months, and two
  hyperparameters were selected on it.
- **Roughly half the target is knowable by construction.** See
  [the base-effect section](#why-the-baseline-is-base-effect-and-not-zero-change).
  This is disclosed and measured against rather than hidden; it is the single most
  important caveat on every result above.
