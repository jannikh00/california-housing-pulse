"""The transform vocabulary the feature spec draws on.

Each function takes a panel column and returns a new column of the same length,
computed **within county** on a panel sorted by ``(county_fips, reference_month)``.

Two invariants hold for every transform here, and the leakage tests in
``tests/test_features.py`` check both:

*Backward only.* A value at row *t* is a function of rows ``<= t - lag`` for that
county and of nothing else. There is no ``shift(-k)`` anywhere in this module;
the only forward reach in the project is the target itself.

*Positional shifts are calendar shifts.* Milestone 1 built a complete
county-month spine, so row *t-3* for a county really is three calendar months
earlier. Without that guarantee a positional shift would quietly mean something
different in counties with coverage gaps, which is why the builder re-sorts and
:func:`assert_complete_spine` refuses to run on a panel with holes in it.

The ``lag`` argument on every function is the *effective* lag — the source's
publication lag plus any extra offset the spec declares — and it is applied
before the transform's own window, never after. Applying it afterwards would
compute a rolling mean over months the forecaster could not yet see and then
merely relabel the result, which is the subtle version of the bug these
functions exist to prevent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

GROUP_KEY = "county_fips"


def assert_complete_spine(panel: pd.DataFrame) -> None:
    """Raise unless every county has a gap-free run of consecutive months.

    Positional shifts stand in for calendar shifts throughout this module. That
    substitution is only valid on a complete spine, so it is checked rather than
    assumed.
    """
    months = panel.groupby(GROUP_KEY)["reference_month"]
    spans = months.agg(["min", "max", "count"])
    expected = (
        (spans["max"].dt.year - spans["min"].dt.year) * 12
        + (spans["max"].dt.month - spans["min"].dt.month)
        + 1
    )
    broken = spans.index[expected != spans["count"]]
    if len(broken):
        raise ValueError(
            f"Panel spine is incomplete for {len(broken)} county(ies): "
            f"{', '.join(map(str, broken[:5]))}. Positional shifts would no longer "
            "correspond to calendar months."
        )


def _grouped(panel: pd.DataFrame, values: pd.Series):
    return values.groupby(panel[GROUP_KEY], sort=False)


def _shift(panel: pd.DataFrame, values: pd.Series, periods: int) -> pd.Series:
    if periods < 0:
        raise ValueError(
            f"Refusing a forward shift of {periods}: feature transforms are "
            "backward-looking by construction."
        )
    if periods == 0:
        return values
    return _grouped(panel, values).shift(periods)


def _positive(values: pd.Series) -> pd.Series:
    """Mask non-positive values so a ratio can be logged.

    Counts legitimately hit zero in thin counties — a month with no sales is a
    real observation, but its log growth is not a number. NA is the honest
    answer; imputing a floor would fabricate a finite growth rate.
    """
    return values.where(values > 0)


def lag(panel: pd.DataFrame, values: pd.Series, param: int, *, lag_months: int) -> pd.Series:
    """``x(t - lag - param)`` — the level, as of the cutoff."""
    return _shift(panel, values, lag_months + param)


def diff(panel: pd.DataFrame, values: pd.Series, param: int, *, lag_months: int) -> pd.Series:
    """``x(t - lag) - x(t - lag - param)`` — change over ``param`` months."""
    current = _shift(panel, values, lag_months)
    earlier = _shift(panel, values, lag_months + param)
    return current - earlier


def log_diff(panel: pd.DataFrame, values: pd.Series, param: int, *, lag_months: int) -> pd.Series:
    """``100 * ln( x(t-lag) / x(t-lag-param) )`` — growth in percent.

    Log rather than simple percentage change, for the same reason the target uses
    it: the result is symmetric, so a rise and the fall that undoes it have equal
    and opposite magnitudes.
    """
    positive = _positive(values)
    current = _shift(panel, positive, lag_months)
    earlier = _shift(panel, positive, lag_months + param)
    ratio = (current / earlier).where(lambda r: r > 0)
    return 100.0 * np.log(ratio)


def rollmean(panel: pd.DataFrame, values: pd.Series, param: int, *, lag_months: int) -> pd.Series:
    """Mean over the ``param`` months ending at ``t - lag``."""
    shifted = _shift(panel, values, lag_months)
    return _grouped(panel, shifted).transform(
        lambda s: s.rolling(param, min_periods=param).mean()
    )


def rollstd(panel: pd.DataFrame, values: pd.Series, param: int, *, lag_months: int) -> pd.Series:
    """Sample standard deviation over the ``param`` months ending at ``t - lag``.

    This is the volatility feature the plan asks for. It matters here more than
    in most panels: Milestone 2 found the target's dispersion is largely a
    function of county thinness, so a model needs some way to know how noisy the
    series in front of it has recently been.
    """
    shifted = _shift(panel, values, lag_months)
    return _grouped(panel, shifted).transform(lambda s: s.rolling(param, min_periods=param).std())


def reltrend(panel: pd.DataFrame, values: pd.Series, param: int, *, lag_months: int) -> pd.Series:
    """``x(t-lag) / mean over the last param months - 1`` — unitless deviation.

    The scale-free counterpart to :func:`diff`. Because 54 counties of very
    different size are pooled into one model, "sales are 30% above this county's
    own recent normal" is a comparable statement across rows in a way that "sales
    are up by 400 homes" is not.
    """
    shifted = _shift(panel, values, lag_months)
    mean = _grouped(panel, shifted).transform(lambda s: s.rolling(param, min_periods=param).mean())
    return shifted / mean.where(mean != 0) - 1.0


TransformFn = Callable[..., pd.Series]

TRANSFORMS: dict[str, TransformFn] = {
    "lag": lag,
    "diff": diff,
    "log_diff": log_diff,
    "rollmean": rollmean,
    "rollstd": rollstd,
    "reltrend": reltrend,
}

# How many months of history each transform consumes beyond its effective lag.
# Used to compute the earliest reference month at which a feature can be known,
# which is what the generated availability table reports.
_DEPTH: dict[str, Callable[[int], int]] = {
    "lag": lambda param: param,
    "diff": lambda param: param,
    "log_diff": lambda param: param,
    "rollmean": lambda param: param - 1,
    "rollstd": lambda param: param - 1,
    "reltrend": lambda param: param - 1,
}


def history_depth(transform: str, param: int) -> int:
    """Months of history a transform reaches back, excluding the release lag."""
    try:
        return _DEPTH[transform](param)
    except KeyError:
        raise KeyError(f"Unknown transform '{transform}'.") from None


def apply_transform(
    panel: pd.DataFrame,
    column: str,
    transform: str,
    param: int,
    *,
    lag_months: int,
) -> pd.Series:
    """Dispatch to a named transform, validating the name first."""
    try:
        fn = TRANSFORMS[transform]
    except KeyError:
        known = ", ".join(sorted(TRANSFORMS))
        raise KeyError(f"Unknown transform '{transform}'; known transforms are {known}.") from None
    return fn(panel, panel[column].astype("float64"), param, lag_months=lag_months)
