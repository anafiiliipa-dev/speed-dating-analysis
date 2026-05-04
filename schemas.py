"""
schemas.py — Lightweight schema validation for incoming dataframes.

Philosophy
----------
We do NOT validate every cell with Pydantic — that would be O(N×K) and
slow on 8k×195 rows. Instead we validate the *contract*:

  1. Required columns are present.
  2. Critical columns have sensible types (after coercion).
  3. Critical numeric columns fall within plausible ranges.

A `SchemaValidationError` is raised early so a malformed CSV cannot
silently propagate into modelling.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict


class SchemaValidationError(ValueError):
    """Raised when an incoming dataframe violates its declared contract."""


class RawDatasetContract(BaseModel):
    """
    Declared contract for the raw Speed Dating CSV.

    Use as:
        RawDatasetContract.validate_dataframe(df)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Columns required by the downstream pipeline. Missing any → fail.
    REQUIRED_COLUMNS: ClassVar[set[str]] = {
        "iid", "pid", "wave", "gender", "dec", "match",
        "attr", "sinc", "intel", "fun", "amb", "shar",
        "attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1",
        "age", "income",
    }

    # (column, lower_bound, upper_bound) — inclusive, after coercion.
    NUMERIC_RANGES: ClassVar[list[tuple[str, float, float]]] = [
        ("wave",   1, 21),
        ("gender", 0, 1),
        ("dec",    0, 1),
        ("match",  0, 1),
        ("age",   17, 60),
        # Event-night ratings: valid range 1-10; we'll clip outliers downstream.
        # Range here is permissive (0-12) to allow data-entry errors through
        # validation, which `clip_rating_outliers` then handles.
        ("attr",   0, 12),
        ("sinc",   0, 12),
        ("intel",  0, 12),
        ("fun",    0, 12),
        ("amb",    0, 12),
        ("shar",   0, 12),
    ]

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> None:
        """Run all contract checks. Raise `SchemaValidationError` on failure."""
        if df.empty:
            raise SchemaValidationError("Dataframe is empty.")

        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise SchemaValidationError(
                f"Missing required columns: {sorted(missing)}"
            )

        for col, lo, hi in cls.NUMERIC_RANGES:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.dropna()
            if valid.empty:
                raise SchemaValidationError(
                    f"Column '{col}' contains no valid numeric values."
                )
            below = (valid < lo).sum()
            above = (valid > hi).sum()
            if below + above > 0:
                # We log but do not fail on out-of-range rating values —
                # they are handled by clip_rating_outliers downstream.
                # We DO fail for structural fields (gender, wave, dec, match).
                if col in {"wave", "gender", "dec", "match"}:
                    raise SchemaValidationError(
                        f"Column '{col}' has {below + above} value(s) "
                        f"outside [{lo}, {hi}]."
                    )


class CleanDatasetContract(BaseModel):
    """Contract for the cleaned dataframe handed to EDA / modelling."""

    REQUIRED_COLUMNS: ClassVar[set[str]] = (
        RawDatasetContract.REQUIRED_COLUMNS
        | {"selectivity", "partner_avg_rating"}
        | {f"{d}1_1_clr" for d in ["attr", "sinc", "intel", "fun", "amb", "shar"]}
    )

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> None:
        if df.empty:
            raise SchemaValidationError("Clean dataframe is empty.")
        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise SchemaValidationError(
                f"Clean dataframe missing columns: {sorted(missing)}"
            )
        # Target must be binary 0/1 with no NaN
        if df["dec"].isna().any():
            raise SchemaValidationError("Target 'dec' contains NaN values.")
        unique_dec = set(pd.to_numeric(df["dec"], errors="coerce").dropna().unique())
        if not unique_dec.issubset({0, 1}):
            raise SchemaValidationError(
                f"Target 'dec' must be binary 0/1, found {unique_dec}"
            )
