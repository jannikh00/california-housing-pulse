# Milestone 0 — Scope and Target Contract

**Status:** Completed on 2 August 2026. Scope frozen for the MVP; the target
threshold remains draft pending the Milestone 2 prevalence check.  
**Date:** 2 August 2026

## One-page problem statement

California housing conditions vary substantially by county and can change faster
than annual summaries reveal. A market analyst who covers many counties needs a
consistent way to identify which markets are gaining or losing price-growth
momentum, estimate the size of that change, and decide where deeper investigation
is warranted. Comparing raw prices is not enough: counties have different price
levels, housing mixes, transaction volumes, and seasonal patterns.

This project will build a monthly county-level forecasting signal. For every
eligible California county and reference month \(t\), it will use only information
available by a documented prediction cutoff to estimate the change in the
county's year-over-year median-sale-price growth between month \(t\) and month
\(t+3\). The primary prediction is a signed numeric value measured in percentage
points. Positive values mean price growth is expected to accelerate; negative
values mean it is expected to decelerate. The same value is translated into one
of three directional labels—**heating**, **stable**, or **cooling**—using thresholds
fixed before model evaluation.

The intended user is a housing-market analyst. The output supports a monthly
county-comparison workflow: rank counties by predicted change, distinguish small
movements from material changes, and select counties for further research. It is
an analytical prioritization aid, not an automated investment, lending, appraisal,
or public-policy decision system.

One modeling row represents one unique `(county_fips, reference_month)` pair.
The reference month is the latest housing observation month used to define the
starting momentum, and the forecast horizon ends three calendar months later.
All county-level housing features, national mortgage-rate features, and optional
county unemployment features must satisfy the prediction cutoff. Future values
may be used only to construct historical targets after those outcomes occur.

The MVP target deliberately measures **change in price-growth momentum**, not the
health of the entire housing market. A county can be labeled cooling even while
prices are still rising if their growth rate slows materially. Conversely, a
county can be labeled heating while prices are falling if the decline becomes
less severe. Inventory, sales, days on market, and sale-to-list ratio are candidate
predictors and contextual indicators, but they do not change the meaning of the
MVP target.

Success means that the pipeline is reproducible and leakage-safe, the learned
model is evaluated against zero-change and persistence baselines on chronological
holdouts, and its magnitude and direction errors are reported honestly. MAE in
percentage points is the primary magnitude metric and macro-F1 is the primary
directional metric. A useful project result does not require a complex model or a
profitable trading signal; it requires evidence that the forecast adds—or does
not add—value beyond simple baselines.

## User and use case

**User:** A housing-market analyst responsible for monitoring and comparing
California counties.

**Use-case statement:** Each month, the analyst compares counties by the expected
near-term direction and magnitude of change in price-growth momentum, ranks the
largest predicted movements, and chooses which counties need deeper investigation.

**Decision supported:** Allocation of the analyst's attention, not a direct
buy/sell, lending, appraisal, or policy decision.

**Output:** For every eligible county:

- a predicted signed change in price-growth momentum, in percentage points;
- a derived label of heating, stable, or cooling;
- later in the project, an uncertainty or empirical error band;
- contextual recent indicators and clear data-as-of metadata.

## Grain and horizon

### County-month grain

The dataset's grain is **one row per California county per reference month**.
The unique key is `(county_fips, reference_month)`. A row is not a home, listing,
sale, ZIP code, or statewide monthly total. County-level values vary across
counties; a national mortgage-rate value will repeat across all counties for the
same month.

Example:

| county_fips | reference_month | meaning |
|---|---|---|
| `06037` | `2026-04` | Los Angeles County's April 2026 modeling row |
| `06073` | `2026-04` | San Diego County's April 2026 modeling row |

Before modeling, the pipeline must verify that this key is unique. Property type
is not part of the MVP grain: use Redfin's aggregate **All Residential** series so
that multiple property-type rows do not accidentally duplicate a county-month.

### Three-month horizon

For reference month \(t\), the outcome month is \(t+3\). For example, an April
2026 row predicts how momentum will have changed by July 2026. “Three months”
means three calendar-month steps in the data, not 90 exact days.

## Continuous target and directional labels

Yes: **continuous target** means the model predicts a numeric value rather than
only a category. More precisely, it can take many signed real-valued outcomes,
such as `-3.7`, `+0.4`, or `+5.1` percentage points.

For county \(c\) and month \(t\), let:

- \(P_{c,t}\) be the monthly median sale price;
- \(S_{c,t}\) be the trailing three-month mean of monthly median sale price;
- \(g_{c,t}\) be the year-over-year growth rate of that smoothed price;
- \(y_{c,t}\) be the continuous three-month-ahead target.

Draft formula:

\[
S_{c,t} = ({P_{c,t} + P_{c,t-1} + P_{c,t-2}})/{3}
\]

\[
g_{c,t} = 100x(((S_{c,t})/(S_{c,t-12})) - 1)
\]

\[
y_{c,t} = \Delta g_{c,t}^{(3)} = g_{c,t+3} - g_{c,t}
\]

Both \(g\) and \(y\) are expressed in **percentage points**, not percent. If
year-over-year growth falls from 5% to 1%, then \(y=-4\) percentage points and
momentum has cooled.

A **directional threshold** is simply a cutoff that maps this numeric prediction
or observed target into a category. With draft threshold \(\tau=2\) percentage
points:

\[
\operatorname{label}(y)=
\begin{cases}
\text{heating}, & y \ge +2 \\
\text{stable}, & -2 < y < +2 \\
\text{cooling}, & y \le -2
\end{cases}
\]

The boundary convention is intentional: exactly `+2.0` is heating and exactly
`-2.0` is cooling.

### Threshold-selection rule

The threshold is a definition of material movement, not a model hyperparameter.
Use this rule once during Milestone 2:

1. Start with the substantively interpretable default \(\tau=2.0\) percentage
   points.
2. Use only the development period—never the final holdout test period—to compute
   class prevalence overall, by year, and by county-volume segment.
3. Retain `2.0` if every class contains at least 15% of development rows overall
   and at least 10% in every full development year.
4. Otherwise, examine the predeclared symmetric grid
   `{1.0, 1.5, 2.0, 2.5, 3.0}` and choose the value closest to `2.0` that meets
   those prevalence floors. Do not choose the threshold that gives the best model
   score or forces perfectly balanced classes.
5. If no candidate qualifies, revisit target stability or county eligibility
   rather than inventing a test-optimized cutoff. Document the evidence and freeze
   the decision before training comparative models.

## Prediction-time (“as-of”) contract

An **as-of rule** answers: “On the day this forecast is claimed to have been made,
could the analyst actually have known this value?” An observation labeled April
is not automatically available on April 30; many April datasets are published or
revised in May.

### Fixed prediction cutoff

For reference month \(t\), define the MVP forecast cutoff as **23:59 Pacific Time
on the 15th calendar day of month \(t+1\)**. If the main Redfin monthly county
release for \(t\) has not occurred by then, move that month's cutoff to the end of
the first day on which it is released and record the exception. Thus “at the end
of month \(t\)” in the high-level project question means “after month \(t\) closes
and its main housing release becomes available,” not literally midnight on the
last day of \(t\).

This cutoff makes the latest main Redfin month consistently usable while forcing
later-published sources to use an older observation. It also gives every row a
concrete `prediction_as_of` timestamp.

### Rule by feature family

| Feature family | Allowed value at the cutoff |
|---|---|
| Redfin main monthly housing metrics | Reference month \(t\), only if the release occurred on or before `prediction_as_of`; otherwise latest previously released month |
| Redfin metrics released separately or later | Latest observation whose actual publication date is on or before `prediction_as_of`; otherwise lag or exclude the feature |
| BLS county unemployment | Latest published county observation, expected to be at least \(t-1\) under the 15th-day cutoff; never assume month \(t\) is known |
| Freddie Mac/FRED mortgage rate | Only weekly observations released on or before `prediction_as_of`; aggregate those observations without using later weeks |
| Calendar variables and county identifiers | May use month, year, season, county FIPS, and fixed geography known by the cutoff |
| Rolling, lagged, or change features | May use only allowed source values above; rolling windows end at the latest allowed observation |
| Imputation, scaling, encoding, feature selection | Fit on training data only, then apply to validation/test data |
| Target and target-derived values | Never features; \(g_{c,t+3}\), \(y_{c,t}\), and the direction label are unavailable until the future outcome is published |

Every staged observation should retain at least `reference_period`, `release_date`
or documented release-lag assumption, `retrieved_at`, `source`, and `raw_snapshot`.
The feature table should retain `prediction_as_of` and the latest source period
used for each feature family.

**Revision limitation:** Current bulk downloads may contain values revised after
their original publication. The MVP will preserve every acquired raw snapshot,
use release-calendar lags, and disclose that historical Redfin vintages cannot be
fully reconstructed unless the source provides them. FRED/ALFRED vintages should
be used where revision history matters. This limitation must not be hidden behind
the simpler rule `observation_month <= t`.

## Source inventory

| Source and MVP role | Access method | Coverage | Native grain | Update frequency / availability | License and usage notes |
|---|---|---|---|---|---|
| [Redfin Housing Market Tracker](https://www.redfin.com/news/data-center/) — target price plus sales, inventory, days on market, and sale-to-list predictors | Download county data from the Data Center; retain the original file and retrieval timestamp. Do not scrape listing pages. | U.S. county data, filtered to California and `All Residential`; metric history and county completeness vary and must be measured from the downloaded files. Redfin's current methodology notes that some newer listing-history metrics begin in 2016. | Monthly county-market metrics; roughly 40 measures across listings, sales, price, and time on market. | Monthly reference periods. The [2026 methodology calendar](https://www.redfin.com/news/data-center/methodology/) generally publishes the main prior-month tracker around the 8th–13th of the next month. Data are subject to revisions and an approximately four-week curing window. | Redfin's [data-use guidance](https://www.redfin.com/news/data-center/) welcomes use with attribution and asks for a citation and link on first reference. Verify terms again before public release and do not imply Redfin endorsement. |
| [Freddie Mac PMMS via FRED, MORTGAGE30US](https://fred.stlouisfed.org/series/MORTGAGE30US) — national financing feature | Reproducible HTTPS CSV download or FRED API. The API requires a registered key; keep it out of version control. ALFRED can provide vintage-aware retrieval. | U.S. national 30-year fixed mortgage average; history since April 1971. | Weekly, ending Thursday, percent, not seasonally adjusted. Convert to cutoff-safe trailing level/change features; a national value repeats for all counties. | Released weekly, normally Thursday. Only releases on or before the row's cutoff are allowed. | Freddie Mac data are copyrighted, provided “as is,” and marked **citation required** on FRED. Include the source's suggested citation. FRED API terms do not override the original data owner's restrictions. |
| [BLS Local Area Unemployment Statistics](https://www.bls.gov/lau/) — optional single county-economic source | Prefer BLS bulk county files for snapshots or the public data API. API v1 is unregistered with lower limits; v2 registration provides larger limits. | Monthly county data are available back to 1990; use California counties. County values are not seasonally adjusted. | County-month estimates of labor force, employment, unemployment, and unemployment rate. | Monthly, with publication lag and annual revisions; use the actual release calendar. Under the 15th-day cutoff, lag to the latest published month rather than using the row's reference month automatically. | BLS publications are public domain except identified third-party images; BLS requests source citation. Do not use the BLS emblem. Estimates are revised, including up to five prior years during annual processing. |
| [U.S. Census Bureau county/ANSI code list](https://www.census.gov/library/reference/code-lists/ansi.html) — geography crosswalk only | Download the official national or California county code text table and retain a snapshot. | All California counties; use state FIPS `06` and the three-digit county code to form a five-digit county FIPS. | One row per county or county equivalent; non-temporal reference data for this project. | Infrequent/as geographic codes change; check for a newer official file when rebuilding. | Publicly disseminated federal reference data; cite the Census Bureau and link the code list. The crosswalk is metadata, not a predictive feature beyond county identity. |

### Source acceptance checks before Milestone 1

- Confirm Redfin exposes a county-level calendar-month file with the required
  price field and enough California history for the chronological split.
- Measure county and metric coverage rather than assuming all 58 counties are
  complete.
- Confirm whether price drops are in the same release as the main tracker; lag or
  exclude them if their release misses the prediction cutoff.
- Record exact source URLs, retrieval timestamps, file hashes, schemas, and terms
  links with each raw snapshot.
- Drop BLS from the MVP at the 5 August gate if its join or release timing delays
  the core housing panel.

## Explicit MVP exclusions

The MVP will not include:

- property-, listing-, ZIP-, city-, or metro-level modeling;
- forecasts beyond California counties or beyond the three-month primary horizon;
- a composite “housing health” target or causal claims about why a market changes;
- direct investment, appraisal, lending, or policy recommendations;
- Zillow, Realtor.com, CoreLogic, ATTOM, HCD, ACS, permit, migration, climate,
  school, crime, demographic, or additional macro datasets;
- more than one national financing source and one optional county-economic source;
- deep learning, large ensembles, extensive hyperparameter searches, or spatial
  econometric models;
- random train/test splits or any feature value published after its forecast cutoff;
- real-time API serving, cloud deployment, scheduled refresh, dashboard, or
  interactive map;
- full historical reconstruction of proprietary-source revisions when vintage
  snapshots are unavailable;
- production SLAs, automated decisions, or claims that the signal is profitable.

## Stretch backlog, in priority order

1. Robust rolling-origin validation and sensitivity to alternative symmetric
   thresholds.
2. Alternative target constructions or supporting signals using inventory, days
   on market, sale-to-list ratio, or a repeat-sales price index.
3. Empirical prediction intervals, classifier calibration, and richer county-level
   error analysis.
4. One bagged-tree and one gradient-boosted model after linear baselines are sound.
5. Additional public economic features such as permits, employment composition,
   migration, ACS demographics, or California HCD data.
6. County adjacency or spatial features, tested without temporal or geographic
   leakage.
7. Interactive California county map or lightweight dashboard.
8. Scheduled data refresh, monitoring, and a simple prediction interface.

Stretch work starts only after the MVP pipeline, temporal evaluation, documentation,
and clean-environment rerun are complete.

## Milestone 0 definition-of-done check

- **One row:** one unique California county and reference month.
- **Prediction:** signed change in year-over-year smoothed-price growth from \(t\)
  to \(t+3\), plus a threshold-derived direction.
- **Prediction time:** a recorded cutoff on or just after the 15th of \(t+1\), with
  source-specific publication lags enforced.
- **User value:** ranks near-term county movements so an analyst can prioritize
  deeper research.
- **Leakage rule:** no feature may have a publication timestamp after its row's
  cutoff; future values are used only to construct matured historical targets.
