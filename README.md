# California Housing Pulse

Forecasting the **three-month change in California county price-growth momentum**:
for each county and reference month, predict the signed change in year-over-year
smoothed median-sale-price growth over the next three months, and translate that
value into a **heating / stable / cooling** label.

> **Status:** Milestone 1 (reproducible data foundation) complete.
> The full project narrative, baseline results, and limitations arrive in Milestone 4.

## Quickstart

```bash
make setup   # create .venv and install dependencies
make all     # fetch raw sources, rebuild the panel, run the tests
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
| `make build` | Rebuild staged tables, the panel, the quality report, and the dictionary |
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
| Columns | 28, every one documented in `configs/columns.yaml` |
| Checks | 13 automated (0 errors, 4 warnings) |

Generated artifacts: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) and
[`reports/data_quality.md`](reports/data_quality.md). Both are rebuilt by
`make build`; do not edit them by hand.

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

## Repository layout

```text
Makefile                 thin wrapper: environment setup and target composition only
configs/
    sources.yaml         source registry: URLs, licences, citations, manual fallbacks
    columns.yaml         column registry: source, meaning, unit, hard/plausible bounds
data/raw/                upstream snapshots (gitignored; manifest.json is committed)
data/snapshots/          small California-only slices, committed for reproducibility
data/interim/            normalized staged tables (rebuildable)
data/processed/          joined county-month panel (rebuildable)
src/california_housing_pulse/
    cli.py               chp fetch / register / verify / build / test / all
    paths.py             project root located by marker file, not by fixed depth
    io.py                typed Parquet helpers that protect the FIPS string contract
    data/                sources, manifest, fetch, normalize, staging, panel,
                         validate, dictionary, pipeline
    features/            leakage-safe feature construction (Milestone 3)
    modeling/            baselines and models (Milestone 3+)
    evaluation/          metrics and error analysis (Milestone 3+)
tests/                   34 tests, no network or large-file dependency
reports/                 generated data-quality report and figures
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
  listing-history metrics as beginning around 2016.
- 113 county-months (1.1%) have no Redfin observation and 58 (0.6%) no BLS
  unemployment value. These are retained as visible rows; the inclusion rule is
  Milestone 2's decision.
