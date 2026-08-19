# Method

How the target is defined, how the features are built, how the data is split, and
what was fitted. Leakage enforcement has its own document,
[LEAKAGE.md](LEAKAGE.md); the data foundation has [DATA.md](DATA.md).

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

Three choices in that definition are worth defending:

**A three-month smoothing window** before the year-over-year ratio. County median
sale prices are noisy at monthly resolution in thin markets, and an unsmoothed
target would measure sampling variation in the numerator as if it were market
movement.

**Log growth rather than percentage change**, so that a rise and the fall that
undoes it are equal and opposite. A +10% move followed by −10% is not a round
trip in percentage terms; in log points it is.

**A ±2 pp band** for the directional label, fixed across all counties. A band
scaled per county would make the classes mean different things in different
places and would make the class prevalences incomparable — but the fixed band has
a real cost, which shows up in the results: `stable` is common in large counties
and rare in thin ones. That cost is reported rather than tuned away.

The definition is frozen in [`configs/target.yaml`](../configs/target.yaml) and a
test asserts every value in it, so editing the config to improve a score also
breaks the build.

### The base effect

Expanding the target shows that it decomposes *exactly*:

```text
Δg = g(t+3) − g(t)
   = 100·ln( P(t+3) / P(t) )  −  100·ln( P(t−9) / P(t−12) )
   =        f(t)              −        b(t)
```

`f(t)` is forward three-month growth and is genuinely unknown at forecast time.
`b(t)` — the **base effect** — is growth that already happened between *t−12* and
*t−9*, and is fully observable when the forecast is made. The identity holds to
machine precision over 8,727 rows and is asserted by a test.

On the test window `corr(Δg, −b) = 0.707`. Roughly half the target's variance is
knowable in advance for reasons that have nothing to do with forecasting a
housing market, and a predictor that only subtracts `b(t)` beats zero-change by
13.5% while knowing nothing at all.

This is not leakage: `b(t)` is legitimately observable, and a real forecaster
would exploit it. Three things follow, and all three are load-bearing.

1. **The magnitude bar is `base_effect`, not `zero_change`.** A result quoted
   against zero-change credits the model for arithmetic it did not perform.
2. **`b(t)` is an explicit feature** (`price_smoothed__log_diff3_o9`) rather than
   something the model reassembles from momentum lags — so the mechanical
   component is visible in the coefficients and can be ablated.
3. **`corr_forward` is the honest skill measure.** It is the correlation between
   predicted and actual `f(t)`. `corr_dg` is inflated because both sides share
   the `−b` term.

The diagnostic validates itself: `base_effect` scores `corr_forward = 0.017`, so
it provably knows nothing about the part that had not yet happened, while ridge
scores 0.580.

## County eligibility

Counties selling fewer than **10 homes in a median month** are excluded from
modelling but remain visible in the panel, flagged by `is_included`. Four
counties fall below the floor: Modoc, Alpine, Sierra and Inyo, which sell one to
five homes in a median month — too few to support a meaningful median price.

The floor is a judgement. It was evidenced, published, and deliberately set
*before* any test data was touched. It is not moved after seeing results.

## Features

61 features, every one declared in
[`configs/features.yaml`](../configs/features.yaml). The transforms live in
Python because they are logic, but *which* features exist is a diffable
specification, and the availability table is generated from it.

| Family | Count | What it carries |
|---|---|---|
| momentum | growth and its changes at several horizons | the series' own recent behaviour |
| base_effect | `price_smoothed__log_diff3_o9` | the mechanical component of the target |
| macro | mortgage rate, unemployment | conditions common to all counties |
| price | levels and ratios | where the county sits, not only how it moved |
| supply | inventory, months of supply, new listings | the stock side |
| volume | homes sold, pending sales | market thickness |
| tightness | days on market, sale-to-list, sold above list | how fast and how hot |
| quality | imputation indicators | whether a value was observed or filled |

**Eligibility is a date, not a completeness test.** The original plan restricted
modelling to rows with a complete feature vector. Measured, that rule deleted Del
Norte from the training set entirely and left Colusa with 22 of 117 months,
because Redfin's coverage of small counties is sporadic and one absent month
blanks a twelve-month rolling window — thin-tier training rows fell 35%. Since
the project had already committed to reporting by volume tier, training without a
tier would have made those reports meaningless. Eligibility therefore begins at
**2014-03**, the first month with enough history for the deepest feature.
Residual gaps — 0.41% of feature cells — are imputed inside the model pipeline on
training statistics only.

## The split

Chronological, contiguous, and frozen in
[`configs/split.yaml`](../configs/split.yaml) on 11 August 2026 — before any
model was fitted — and asserted by a guard test, including the row counts.

| Split | Rows | Window | cooling / stable / heating |
|---|---|---|---|
| train | 6,271 | 2014-03 → 2023-11 | 35.3% / 31.5% / 33.2% |
| validation | 486 | 2024-03 → 2024-11 | 37.4% / 33.3% / 29.2% |
| test | 648 | 2025-03 → 2026-02 | 36.7% / 36.4% / 26.9% |

All 54 modelled counties appear in all three splits. Each boundary carries a
three-month embargo — see [LEAKAGE.md](LEAKAGE.md#the-embargo-which-is-not-a-feature-control).

The window is capped by the data: the Redfin bulk file was last refreshed on
2 June 2026, so the panel ends at 2026-05, and after the three-month lead needed
to observe Δg the most recent usable target row is 2026-02. **If a newer Redfin
file shifts the panel, the split must be recomputed before the test set is read
again — never after.**

## What was fitted

Five naive baselines, plus one model per primary metric.

| Model | Kind | What it assumes |
|---|---|---|
| `zero_change` | naive | momentum does not change |
| `base_effect` | naive | forward growth is at its climatological average; subtract the observable base |
| `persistence` | naive | the last observed momentum change continues |
| `mean_reversion` | naive | momentum returns toward its own rolling mean, at a rate `k` fitted on training rows |
| `majority_class` | naive | always predict the most common training class |
| `ridge` | learned | regularized linear regression on the magnitude target |
| `multinomial_logistic` | learned | direct three-class classifier |

Hyperparameters were scanned on **validation only**; the chosen configurations
were refit on train + validation; the test split was scored **once**. Feature
preprocessing is fitted inside an sklearn `Pipeline` so its statistics come only
from training data, asserted by a test.

Directional labels for `ridge` are derived by applying the frozen ±2 pp
thresholds to its magnitude predictions, which is why it appears in both the
magnitude and direction tables.

## Metrics

Fixed in the plan before any result was seen, and named in code
(`metrics.PRIMARY_MAGNITUDE`, `metrics.PRIMARY_DIRECTIONAL`) so a later table
cannot quietly promote whichever number came out best.

- **Magnitude:** MAE in percentage points, primary. RMSE, median AE and **mean
  error** reported beside it — mean error because a model that predicts +3 half
  the time and −3 the other half has the same MAE as one that always predicts +3,
  and those are very different failures.
- **Direction:** macro-F1 across the three classes, primary. Balanced accuracy,
  per-class precision/recall and the 3×3 confusion matrix reported beside it.
- **Forward skill:** `corr_forward`, for the reason given above.
- **Probabilistic:** multiclass log loss and Brier score for the logistic. These
  are *scored* but not *calibrated*; no reliability diagram has been fitted.

Every one of these is reportable within volume tier and within period, and the
project's convention is that they are always reported that way. A pooled score is
dominated by whichever segment is largest or noisiest, and a model can appear to
improve simply by getting better at the counties that matter least.

## Uncertainty

95% intervals come from a **block bootstrap**, resampling whole blocks rather
than rows, because rows within a county and within a month are not independent.
Both blocking schemes are reported:

- **Resampling counties** answers "would a different set of counties give this?"
- **Resampling months** answers "would a different twelve months give this?"

The month interval is the narrower of the two, and that is not reassuring — it is
narrow precisely because twelve months of a single cooling-leaning regime resemble
each other. Reporting only the tighter number would flatter the result.
