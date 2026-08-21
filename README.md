# California Housing Pulse

Forecasting the **three-month change in California county price-growth momentum**:
for each county and reference month, predict the signed change in year-over-year
smoothed median-sale-price growth over the next three months, and translate that
value into a **heating / stable / cooling** label.

> **Status:** MVP, tagged `v0.1.0-mvp`. Target, features and split are frozen and
> guarded by tests. 145 tests, 61 features, one test window scored once.

## The finding

Half of this target was knowable before the model started, by arithmetic. We
found that, said so, measured against it anyway — and there is still real signal
left.

The target decomposes *exactly* into `Δg(t) = f(t) − b(t)`, where `b` is
three-month growth that had already happened between *t−12* and *t−9* and is
fully observable at forecast time. On the test window `corr(Δg, −b) = 0.707`. So
a predictor that knows nothing about housing, and merely subtracts `b`, already
beats the obvious zero-change baseline by 13.5%.

That makes the usual headline misleading. The honest question is how well a model
predicts `f(t)` — the part that had not yet happened.

![forward skill](reports/figures/fig09_forward_skill.png)

The base-effect baseline scores `corr_forward` **0.017**: it provably knows
nothing about the future, which is exactly what makes it a trustworthy bar.
Ridge scores **0.580**. That gap is the claim this project makes.

## Results

Test split **2025-03 → 2026-02**: 648 county-months, 54 counties, scored **once**
after the pipeline was frozen. Regenerate with `make report`; full detail in
[`reports/results.md`](reports/results.md) and
[`reports/baselines.md`](reports/baselines.md).

| Model | Kind | MAE (pp) | vs base effect | vs zero change | macro-F1 | vs majority | corr on f(t) |
|---|---|---|---|---|---|---|---|
| persistence | naive | 9.257 | −97.0% | −70.3% | 0.270 | +0.091 | −0.157 |
| zero_change | naive | 5.434 | −15.6% | — | 0.178 | −0.001 | 0.108 |
| mean_reversion | naive | 5.037 | −7.2% | +7.3% | 0.368 | +0.189 | 0.208 |
| base_effect | naive | 4.699 | — | +13.5% | 0.515 | +0.336 | 0.017 |
| majority_class | naive | — | — | — | 0.179 | — | — |
| **ridge** | learned | **3.823** | **+18.6%** | +29.6% | 0.569 | +0.390 | **0.580** |
| **multinomial_logistic** | learned | — | — | — | **0.638** | **+0.459** | — |

95% block-bootstrap intervals, resampling whole counties: ridge MAE
**[3.230, 4.490]**, logistic macro-F1 **[0.594, 0.681]**, base-effect MAE
[3.996, 5.488].

![baseline comparison](reports/figures/fig06_baseline_comparison.png)

**Persistence — the conventional naive forecast — is 70% worse than predicting no
change at all.** That is a finding about this series, not an embarrassment to
hide: the market reverses hard (the sharpest regime turn is +7.44 pp at 2023-02),
and a series that reverses punishes extrapolation. The opposite bet, mean
reversion, fitted *k* = +0.278 and beat it comfortably.

### Never pooled

| Tier | Rows | ridge MAE | ridge mean error | logistic accuracy | logistic macro-F1 |
|---|---|---|---|---|---|
| thin | 72 | 8.474 | +1.391 | 0.750 | 0.615 |
| small | 168 | 4.597 | +1.990 | 0.655 | 0.631 |
| mid | 240 | 3.060 | +1.901 | 0.583 | 0.591 |
| large | 168 | 2.147 | +1.841 | 0.655 | 0.491 |

![results by tier](reports/figures/fig07_results_by_tier.png)

Magnitude error runs **4× worse** in thin counties — and direction runs the *other
way*, with large counties posting the best MAE and the worst macro-F1. The
mechanism is the fixed ±2 pp band: noise scales with market thinness, so thin
counties rarely sit in `stable` and their direction is easy to call, while large
counties are `stable` about half the time and the model must separate three
genuinely close classes.

Ridge's pooled macro-F1 of 0.569 therefore describes **no tier in that table**.
This is why nothing here is quoted pooled.

![county error](reports/figures/fig08_county_error.png)

Two more figures cover the target itself:
[distribution and class thresholds](reports/figures/fig01_target_distribution.png)
and [the underlying series by tier](reports/figures/fig03_county_time_series.png).

## What this project can and cannot claim

**It can claim:**

- On one untouched twelve-month window, scored once, ridge predicted the
  magnitude of the three-month momentum change with MAE **3.823 pp** against the
  honest naive bar's 4.699 — **18.6% less error**.
- That skill is real forecasting rather than recovered arithmetic:
  `corr_forward` **0.580** against the base-effect baseline's **0.017**.
- Multinomial logistic separates the three classes at macro-F1 **0.638** against
  a majority-class floor of 0.179.
- Every feature was verified available at prediction time by three independent
  controls — one of which **found and fixed a real leak** in the unemployment lag.

**It cannot claim:**

- **That these numbers generalise across market regimes.** One contiguous test
  window, which leans cooling. Rolling-origin validation is the next milestone;
  until then, a single held-out window is the honest description of the evidence.
- **Useful county-level accuracy in thin markets.** MAE 8.47 pp in the thin tier.
  A forecast for Mono or Colusa carries error larger than most of the moves it is
  trying to call.
- **Calibrated probabilities.** The logistic emits class probabilities and they
  are scored (log loss 0.814, Brier 0.477), but no calibration was fitted and no
  reliability diagram drawn.
- **Any causal statement.** These are predictive associations. Nothing here
  identifies a mechanism, and nothing supports an intervention.
- **Symmetric performance across classes.** The models **under-call cooling** in
  a cooling window: cooling precision 0.822 but recall only 0.445, and mean error
  is positive in every quarter. Left uncorrected on purpose — tuning class
  weights after seeing the test set is what the frozen contracts exist to prevent.
- **That beating persistence means much.** Persistence is worse than doing
  nothing here, so clearing it is a low bar. The bar that counts is `base_effect`.

## Quickstart

```bash
make setup   # create .venv and install dependencies
make all     # fetch, rebuild the panel, build features, render the EDA,
             #   fit and evaluate the baselines, render the report, run the tests
```

`make all` is the single documented rebuild, and it runs from a clean checkout:
the repository commits small California-only slices in `data/snapshots/`, so no
241 MB upstream download is needed to reproduce the headline result. The Makefile
contains no pipeline logic — it sets the environment and delegates to the CLI —
so the direct invocation is equivalent and is the fallback on Windows:

```bash
UV_NO_EDITABLE=1 uv run --reinstall-package california-housing-pulse chp all
```

> **Why `UV_NO_EDITABLE`:** uv's editable install writes a `.pth` file without a
> trailing newline, which Python's `site` machinery ignores, so the package
> silently fails to import. A non-editable install avoids this;
> `--reinstall-package` keeps the installed copy in step with the working tree.

| Command | Purpose |
|---|---|
| `make fetch` | Download raw sources into `data/raw/` and record provenance |
| `make build` | Rebuild staged tables, the panel, the target, the quality report, the dictionary |
| `make features` | Build the feature matrix, audit publication lags, render the availability table |
| `make eda` | Render `reports/eda.md` and the five exploratory figures |
| `make baselines` | Fit the baselines and models, evaluate, render `reports/baselines.md` |
| `make report` | Render `reports/results.md` and the four results figures |
| `make verify` | Re-hash raw snapshots against `data/raw/manifest.json` |
| `make test` | Run the test suite |
| `make lint` | Run ruff |

`make report` reads saved predictions and never refits a model, so the story and
its figures can be regenerated without touching the frozen test split again.

## How it works

| Document | Covers |
|---|---|
| [docs/METHOD.md](docs/METHOD.md) | The frozen target and why it is shaped that way, the base-effect decomposition, county eligibility, the 61 features, the split, what was fitted, the metrics, the bootstrap |
| [docs/LEAKAGE.md](docs/LEAKAGE.md) | The four leakage controls, the real BLS leak the publication audit caught, the embargo, and what the controls do *not* cover |
| [docs/DATA.md](docs/DATA.md) | Sources and licences, the panel, provenance and validation, what the exploratory analysis decided, data limitations |
| [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Every column: source, meaning, unit, bounds, measured null rate and range |

In brief: the target is a frozen contract in `configs/target.yaml`, guarded by a
test that fails if any value drifts. Features are declared in
`configs/features.yaml` with per-source publication lags, and no transform may
shift forward — the helper raises on a negative offset. The split is frozen in
`configs/split.yaml` with a three-month embargo at each boundary. Hyperparameters
were selected on validation, refit on train + validation, and the test split was
scored once.

## Repository layout

```text
Makefile                 thin wrapper: environment setup and target composition only
configs/                 sources, columns, target, features, split — all frozen specs
data/raw/                upstream snapshots (gitignored; manifest.json is committed)
data/snapshots/          small California-only slices, committed for reproducibility
data/interim/            normalized staged tables (rebuildable)
data/processed/          panel, target panel, features, predictions (rebuildable)
models/                  baseline_config.json: chosen hyperparameters and the full scan
docs/                    method, leakage, data, and the generated data dictionary
reports/                 generated reports and figures
src/california_housing_pulse/
    cli.py               chp fetch / register / verify / build / features / eda /
                         baselines / report / test / all
    paths.py             project root located by marker file, not by fixed depth
    io.py                typed Parquet helpers that protect the FIPS string contract
    viz.py               shared figure palette and chrome
    data/                sources, manifest, fetch, normalize, staging, panel,
                         validate, dictionary, pipeline
    features/            target.py executes the frozen contract; spec declares,
                         transforms computes, build assembles
    modeling/            split, baselines, models, run
    evaluation/          metrics, bootstrap, report
    reporting/           results, figures, report — the MVP story layer
tests/                   145 tests, no network or large-file dependency
```

## Data sources

Redfin Data Center (target price series and market activity, used with
attribution), Freddie Mac PMMS via FRED (`MORTGAGE30US`, copyrighted, "as is"),
BLS Local Area Unemployment Statistics, and the Census county/ANSI code list.
Full citations, retrieval timestamps and file hashes are in
[docs/DATA.md](docs/DATA.md) and
[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

## Next

Rolling-origin cross-validation across market regimes, replacing dependence on a
single test window; fold-by-fold intervals including one for `corr_forward`,
which currently carries none; and proper feature selection — the 13 `tightness`
features scored a *negative* contribution in the ablation and were deliberately
**not** pruned, because the gain sits inside bootstrap noise and was measured on
486 validation rows.

## Licence

[MIT](LICENSE). Upstream data carries its own terms; see
[docs/DATA.md](docs/DATA.md).
