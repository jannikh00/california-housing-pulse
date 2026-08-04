# California Housing Pulse

Forecasting the **three-month change in California county price-growth momentum**:
for each county and reference month, predict the signed change in year-over-year
smoothed median-sale-price growth over the next three months, and translate that
value into a **heating / stable / cooling** label.

> Status: Milestone 1 (reproducible data foundation) in progress.
> The full project narrative, results, and limitations are written in Milestone 4.

## Quickstart

```bash
uv sync --extra dev     # create .venv and install the package
uv run chp --help       # pipeline entry point
```

## Repository layout

```text
configs/                 source definitions and pipeline configuration
data/raw/                upstream snapshots (gitignored; manifest.json is committed)
data/snapshots/          small California-only slices, committed for reproducibility
data/interim/            normalized staged tables (rebuildable)
data/processed/          joined county-month panel (rebuildable)
src/california_housing_pulse/
    data/                acquisition, normalization, staging, joining, validation
    features/            leakage-safe feature construction
    modeling/            baselines and models
    evaluation/          metrics and error analysis
tests/                   automated checks
reports/figures/         generated figures
```

## Data sources

See `docs/` for the data dictionary. Attribution requirements for Redfin, Freddie Mac/FRED, and BLS are recorded with each source's metadata and are reproduced before public release.
