# Leakage controls

The central claim of this project is that no feature uses information that was
unavailable at the moment the forecast was made. This document is how that claim
is enforced, and what it caught.

Four mechanisms, each catching failures the others cannot.

| Control | Catches | Enforced by |
|---|---|---|
| Backward-only transforms | a mis-signed shift | `_shift` raises on any negative offset |
| Truncation replay | a window straddling the cutoff | rebuild on a panel cut at *t*, assert identical values |
| Publication audit | a lag too short for the real world | each input's publication date ≤ `prediction_as_of` |
| Split embargo | labels resolving into the next split | three unused months at each boundary |

## Every row carries a cutoff

`prediction_as_of` is the 15th of the month after the reference month. A feature
is safe when every input it reads was published on or before that timestamp.

This is deliberately a *timestamp* rather than a row offset. "Lag by one month"
is a statement about the index; "was it published yet" is a statement about the
world, and only the second one is the question that matters.

## Transforms are backward-only

There is no forward shift anywhere in `features/transforms.py`. The shift helper
raises on a negative offset, so the failure cannot be *expressed* rather than
merely being absent from the current code. A future contributor reaching forward
gets an exception, not a plausible-looking number.

## Truncation replay

Every feature is recomputed on a panel cut off at month *t* and must reproduce
the full-panel value exactly. If any transform reached forward, hiding the future
would change the answer, and the assertion fails.

This is the control that catches the subtle case: a rolling window that is
correctly *shifted* but incorrectly *centred* still reads forward, and reading
the transform code will not always reveal it.

## A publication audit, which found a real problem

Each source declares two separate things: the lag **we chose**
(`release_lag_months`) and how long the publisher **actually takes**
(`publication_delay_days`). The audit checks the first against the second.

It found a genuine leak. BLS publishes county Local Area Unemployment Statistics
roughly six to eight weeks after the reference month, but the forecast cutoff
falls on the 15th of the following month. Unemployment for month *t* is therefore
**not knowable** at *t*'s cutoff, and the natural one-month lag is about a week
short. `bls_lau_california` consequently carries a **two-month** release lag, and
`unemployment_rate__diff12` reads *t−14* → *t−2*.

The audit is not a restatement of its own configuration. A test forces the BLS
lag back to one month and asserts that the audit objects — flagging three
features as leaking by up to seven days. Without that test the check could
degrade into comparing a config value with itself and still pass.

All 61 features currently clear the cutoff by at least **8 days**.
[`reports/feature_availability.md`](../reports/feature_availability.md) states the
oldest month every feature reads, generated from
[`configs/features.yaml`](../configs/features.yaml) so the published table cannot
drift from the code.

## The embargo, which is not a feature control

The three controls above protect *features*. The embargo protects **model
selection**, which is a different failure and needs a different mechanism.

The label at month *t* is only observable at *t+3*. With contiguous windows, a
training row at 2024-02 carries an outcome that resolves inside the validation
period — so a model selected on that validation window has, indirectly, been
selected using an outcome it was trained on. Three months at each boundary are
therefore assigned to no split and used by nothing, costing 324 of 8,053
otherwise usable rows.

The embargo was not in the original plan. It was added in Milestone 3 once the
horizon and the contiguous split were both fixed and the interaction became
visible.

## What these controls do not cover

**Vintages.** Redfin revises recent months, and the current bulk file cannot be
used to reconstruct what the data looked like at any past date. The
prediction-time contract therefore rests on *documented release lags* rather than
on true point-in-time snapshots. A feature can be provably lagged correctly and
still have been computed from a number that was later revised. This is disclosed
rather than solved; solving it would require archiving vintages going forward.

**The target's own construction.** Roughly half the target is knowable in advance
for arithmetic reasons — see [METHOD.md](METHOD.md#the-base-effect). That is not
leakage, because the quantity involved is legitimately observable at forecast
time, but it has the same power to inflate a result, and it needs the same kind
of explicit accounting.
