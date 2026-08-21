# Changelog

## v0.1.0-mvp — 18 August 2026

First complete, reproducible release: a small finished project rather than a
large unfinished one. `make all` runs fetch → build → features → eda → baselines
→ report → test from a clean checkout, with no upstream download required.

**Headline.** Roughly half the target is knowable in advance by arithmetic. That
was found, disclosed, and measured against: ridge cuts MAE **18.6%** against the
base-effect baseline on a test window scored once, and scores `corr_forward`
**0.580** on the genuinely unknown part where the baseline scores **0.017**.
Multinomial logistic reaches macro-F1 **0.638** against a majority floor of
0.179.

### Milestone 4 — MVP release and story

- `chp report` and the `reporting/` package: renders `reports/results.md` and
  four results figures from saved predictions, without refitting a model.
- Four results figures: baseline comparison, results by volume tier, per-county
  error ranking, and the forward-skill decomposition.
- `viz.py`: figure palette and chrome shared by the exploratory and results
  figures, so the two sets cannot drift apart visually.
- README rewritten story-first, with the method, leakage and data detail moved
  into `docs/METHOD.md`, `docs/LEAKAGE.md` and `docs/DATA.md`.
- A prominent statement of what the project can and cannot claim, with every
  number in it interpolated from the live results rather than transcribed.
- `LICENSE` (MIT) and this changelog.

### Milestone 3 — Leakage-safe features, temporal split, baselines

- 61 features declared in `configs/features.yaml`, enforced by backward-only
  transforms, truncation replay and a publication audit. The audit **found a real
  leak**: BLS publishes county unemployment six to eight weeks after the
  reference month, so a one-month lag fell about a week short of the forecast
  cutoff. Corrected to two months.
- Chronological split frozen in `configs/split.yaml` with a three-month embargo
  at each boundary, protecting model selection rather than features.
- Five naive baselines plus ridge and multinomial logistic; hyperparameters
  selected on validation, refit on train + validation, test scored once.
- **The target's base-effect decomposition**, which changed how every result in
  the project is stated: `base_effect` replaced `zero_change` as the headline
  magnitude bar, `b(t)` became an explicit feature, and `corr_forward` became the
  reported skill measure.
- Block-bootstrap intervals over counties and over months; metrics by volume tier
  and by period.

### Milestone 2 — Target construction and exploratory analysis

- Volatility shown to track market thinness rather than market movement; four
  counties below a median of 10 monthly sales excluded from modelling and kept
  visible in the panel.
- Volume tier retained as a permanent evaluation dimension after `stable` was
  shown to be partly a volume artifact.
- Regime shifts and the absence of residual seasonality established; five
  exploratory figures.

### Milestone 1 — Data foundation

- Hybrid acquisition with full provenance: URL, SHA-256, size, retrieval time and
  server `Last-Modified` per source, plus a documented manual fallback that
  produces identical records.
- Panel built on a complete county × month spine, so coverage gaps are countable
  rather than invisible.
- Column registry with hard and plausible bounds, and a generated data dictionary
  that cannot drift from the built panel.

### Milestone 0 — Scope and target contract

- Target, horizon, directional threshold and county inclusion floor frozen in
  `configs/target.yaml` before any modelling, guarded by a test.
