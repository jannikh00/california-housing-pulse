"""The chronological train / validation / test split.

Like the target, this is a *contract* rather than a computation: the boundaries
live in ``configs/split.yaml``, are frozen at a dated decision, and are asserted
by a guard test. The plan is emphatic about why. A split recomputed on every run
would move silently the moment Redfin published a newer bulk file, and a split
that can move after test results are seen is not a holdout at all.

**The embargo.** The label at month *t* is only observable at *t+3*, so a row in
the last three months of a split carries an outcome drawn from the era of the
next split. Contiguous boundaries would therefore let training labels resolve
inside the validation window — not feature leakage, which the feature audit
already rules out, but leakage into *model selection*. The three months at each
boundary are assigned to ``embargo`` and used by nothing.

    train      ....2023-11 |  embargo 2023-12..2024-02  | validation 2024-03....
                            ^ every train label resolves at or before 2024-02

**Eligibility.** A row may be used when its county clears the Milestone 2 volume
floor, its label is observable, and the panel holds enough history for the
features to exist. That last condition is a *date* rather than a per-row
completeness test, and the distinction matters: requiring every feature to be
non-null silently deletes the thin counties whose Redfin coverage is sporadic,
which are exactly the counties Milestone 2 asked to keep visible. Residual gaps
after the lead-in are a preprocessing concern, handled inside the model pipeline
where the fill can be fitted on training data alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from ..paths import CONFIG_DIR

SPLIT_CONFIG = CONFIG_DIR / "split.yaml"

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
EMBARGO = "embargo"
INELIGIBLE = "ineligible"

# Order matters: reports and figures read it, and it is chronological.
SPLIT_ORDER = (TRAIN, VALIDATION, TEST)


@dataclass(frozen=True)
class SplitWindow:
    """One contiguous span of reference months."""

    name: str
    start: pd.Timestamp | None
    end: pd.Timestamp

    def contains(self, months: pd.Series) -> pd.Series:
        within = months <= self.end
        if self.start is not None:
            within &= months >= self.start
        return within

    def describe(self) -> str:
        start = f"{self.start:%Y-%m}" if self.start is not None else "panel start"
        return f"{start} to {self.end:%Y-%m}"


@dataclass(frozen=True)
class SplitContract:
    """The frozen definition of which rows may be used, and for what."""

    windows: tuple[SplitWindow, ...]
    embargo_months: int
    lead_in_first_month: pd.Timestamp
    require_complete_features: bool
    expected_rows: dict[str, int]
    frozen_status: str
    frozen_date: str

    def window(self, name: str) -> SplitWindow:
        for window in self.windows:
            if window.name == name:
                return window
        raise KeyError(f"No split window named '{name}'.")

    def describe(self) -> str:
        spans = " · ".join(f"{w.name} {w.describe()}" for w in self.windows)
        return f"{spans} · {self.embargo_months}-month embargo at each boundary"


@lru_cache(maxsize=1)
def load_split_contract(config_path: Path | None = None) -> SplitContract:
    """Parse ``configs/split.yaml`` into the frozen :class:`SplitContract`."""
    path = Path(config_path) if config_path else SPLIT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Split contract not found at {path}")

    raw = yaml.safe_load(path.read_text())
    boundaries = raw["boundaries"]
    eligibility = raw["eligibility"]
    frozen = raw["frozen"]

    windows = tuple(
        SplitWindow(
            name=name,
            start=pd.Timestamp(boundaries[name]["start"]) if boundaries[name]["start"] else None,
            end=pd.Timestamp(boundaries[name]["end"]),
        )
        for name in SPLIT_ORDER
    )

    return SplitContract(
        windows=windows,
        embargo_months=int(raw["embargo"]["months"]),
        lead_in_first_month=pd.Timestamp(eligibility["lead_in"]["first_month"]),
        require_complete_features=bool(eligibility["feature_gaps"]["require_complete"]),
        expected_rows={str(k): int(v) for k, v in raw["expected_rows"].items()},
        frozen_status=frozen["status"],
        frozen_date=str(frozen["date"]),
    )


def eligible(
    features: pd.DataFrame,
    contract: SplitContract | None = None,
    feature_columns: list[str] | None = None,
) -> pd.Series:
    """Boolean mask of rows the project may model on at all."""
    contract = contract or load_split_contract()
    mask = (
        features["is_included"]
        & features["has_target"]
        & (features["reference_month"] >= contract.lead_in_first_month)
    )
    if contract.require_complete_features:
        if feature_columns is None:
            from ..features.build import feature_names

            feature_columns = feature_names()
        mask &= features[feature_columns].notna().all(axis=1)
    return mask


def assign_split(
    features: pd.DataFrame,
    contract: SplitContract | None = None,
    feature_columns: list[str] | None = None,
) -> pd.Series:
    """Label every row ``train``/``validation``/``test``/``embargo``/``ineligible``.

    Returned as a column rather than three frames on purpose: the assignment
    stays attached to the data, so an evaluation report can always show what a
    row was used for, and a row can never silently appear in two places.
    """
    contract = contract or load_split_contract()
    months = features["reference_month"]
    usable = eligible(features, contract, feature_columns)

    split = pd.Series(INELIGIBLE, index=features.index, dtype="object")
    assigned = pd.Series(False, index=features.index)
    for window in contract.windows:
        in_window = usable & window.contains(months) & ~assigned
        split[in_window] = window.name
        assigned |= in_window

    # Anything eligible and inside the overall span but not claimed by a window
    # is embargoed: it fell in a gap the boundaries deliberately left open.
    first = contract.windows[0]
    last = contract.windows[-1]
    inside = months <= last.end
    if first.start is not None:
        inside &= months >= first.start
    split[usable & inside & ~assigned] = EMBARGO

    return pd.Series(split, index=features.index, dtype="string")


@dataclass
class SplitReport:
    """Row accounting and regime description for the frozen split."""

    counts: dict[str, int]
    months: dict[str, int]
    counties: dict[str, int]
    spans: dict[str, str]
    prevalence: dict[str, dict[str, float]]
    embargo_rows: int = 0
    ineligible_rows: int = 0

    def summary(self) -> str:
        lines = ["split:"]
        for name in SPLIT_ORDER:
            shares = self.prevalence.get(name, {})
            mix = " / ".join(f"{shares.get(k, 0.0):.1%}" for k in ("cooling", "stable", "heating"))
            lines.append(
                f"  {name:<11} {self.counts.get(name, 0):>5,} rows  "
                f"{self.spans.get(name, ''):<22} "
                f"{self.months.get(name, 0):>3} months  "
                f"{self.counties.get(name, 0):>2} counties  {mix}"
            )
        lines.append(
            f"  embargoed {self.embargo_rows:,} rows at the boundaries; "
            f"{self.ineligible_rows:,} rows ineligible"
        )
        return "\n".join(lines)


def summarize_split(features: pd.DataFrame, split: pd.Series) -> SplitReport:
    """Measure what the split produced, including the class mix per period."""
    frame = features.assign(split=split)
    counts, months, counties, spans, prevalence = {}, {}, {}, {}, {}

    for name in SPLIT_ORDER:
        part = frame.loc[frame["split"] == name]
        counts[name] = len(part)
        months[name] = int(part["reference_month"].nunique())
        counties[name] = int(part["county_fips"].nunique())
        spans[name] = (
            f"{part['reference_month'].min():%Y-%m} to {part['reference_month'].max():%Y-%m}"
            if len(part)
            else ""
        )
        if len(part):
            shares = part["target_label"].value_counts(normalize=True)
            prevalence[name] = {
                k: float(shares.get(k, 0.0)) for k in ("cooling", "stable", "heating")
            }

    return SplitReport(
        counts=counts,
        months=months,
        counties=counties,
        spans=spans,
        prevalence=prevalence,
        embargo_rows=int((frame["split"] == EMBARGO).sum()),
        ineligible_rows=int((frame["split"] == INELIGIBLE).sum()),
    )


def check_no_label_crosses_a_boundary(
    features: pd.DataFrame,
    split: pd.Series,
    horizon_months: int,
) -> list[str]:
    """Verify the embargo actually did its job.

    A row at month *t* resolves at *t + horizon*. For the split to be honest,
    every training label must resolve at or before the first validation month,
    and every validation label at or before the first test month. This recomputes
    that from the assigned data rather than trusting the configured dates.
    """
    frame = features.assign(split=split)
    problems = []
    for earlier, later in ((TRAIN, VALIDATION), (VALIDATION, TEST)):
        before = frame.loc[frame["split"] == earlier, "reference_month"]
        after = frame.loc[frame["split"] == later, "reference_month"]
        if before.empty or after.empty:
            continue
        resolves = before.max() + pd.DateOffset(months=horizon_months)
        if resolves > after.min():
            problems.append(
                f"{earlier} labels resolve at {resolves:%Y-%m}, after {later} "
                f"begins at {after.min():%Y-%m}"
            )
    return problems
