"""Typed Parquet read/write helpers.

Parquet preserves dtypes, which matters here for one specific reason: county FIPS
codes are zero-padded strings (``06037``). Any round-trip that silently converts
them to integers destroys the join key. These helpers make the string contract
explicit and enforce it on read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import relative

# Columns that must remain zero-padded strings everywhere in the pipeline.
STRING_KEY_COLUMNS = ("county_fips", "state_fips", "county_code")


def write_parquet(frame: pd.DataFrame, path: Path, *, quiet: bool = False) -> Path:
    """Write a DataFrame to Parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_string_keys(frame, context=f"write {relative(path)}")
    frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    if not quiet:
        print(f"  wrote {relative(path)}  ({len(frame):,} rows x {frame.shape[1]} cols)")
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file and re-assert the string-key contract."""
    frame = pd.read_parquet(Path(path), engine="pyarrow")
    _assert_string_keys(frame, context=f"read {relative(Path(path))}")
    return frame


def _assert_string_keys(frame: pd.DataFrame, *, context: str) -> None:
    for column in STRING_KEY_COLUMNS:
        if column in frame.columns and not pd.api.types.is_string_dtype(frame[column]):
            raise TypeError(
                f"{context}: column '{column}' must be a string dtype to preserve "
                f"leading zeros, got {frame[column].dtype}."
            )
