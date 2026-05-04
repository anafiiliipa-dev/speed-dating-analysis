"""
01_data_engineering.py — Stage 1: Data Engineering & Cleaning.

Pipeline (sequential, each step is a pure function):

    load_raw  →  validate  →  flag_structural_missings
              →  harmonise_pref_scales  →  clr_transform
              →  impute_random_missings  →  clip_rating_outliers
              →  build_derived_features  →  build_model_ready_dataset

Key scientific decisions (preserved from v1):

  • Missing values classified as STRUCTURAL (not collected) vs RANDOM
    (skipped). Only RANDOM NaNs imputed.
  • Waves 6-9 stated-pref scale ×10 + row-renormalised to sum 100
    (Fisman et al. 2006).
  • CLR transform on compositional preference vectors (Aitchison 1986)
    to remove unit-sum constraint before regression.
  • T2/T3 columns excluded from model-ready dataset (anti-leakage).

Engineering improvements over v1:

  • Loguru structured logging (DEBUG/INFO/SUCCESS).
  • Pydantic schema validation (fail-fast on raw & clean dataframes).
  • Custom exception hierarchy (`DataEngineeringError`).
  • `PipelineResult` dataclass — typed return contract.
  • All paths via validated `settings` (no hardcoded paths).
  • Pure functions — every step does df.copy() and returns new frame.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    ATTR_DIMS, LEAKAGE_COLS, LIFESTYLE_COLS, LIKERT_WAVES,
    MODEL_FEATURES, MODEL_TARGET, PARTNER_PREF_COLS,
    RATINGS_GIVEN, RATINGS_RECV, STATED_PREF_CLR, STATED_PREF_COLS,
    settings,
)
from logging_config import get_logger
from schemas import CleanDatasetContract, RawDatasetContract, SchemaValidationError

warnings.filterwarnings("ignore", category=FutureWarning)
log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────
class DataEngineeringError(RuntimeError):
    """Base class for errors raised in this stage."""


class RawFileNotFoundError(DataEngineeringError):
    """Raised when the raw CSV cannot be located."""


class ImputationError(DataEngineeringError):
    """Raised when imputation cannot complete."""


# ─────────────────────────────────────────────────────────────────────
# Result contract
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PipelineResult:
    clean_df: pd.DataFrame
    model_df: pd.DataFrame
    n_rows_raw: int
    n_rows_clean: int
    n_rows_model: int
    target_positive_rate: float


# ─────────────────────────────────────────────────────────────────────
# 1. Raw loader
# ─────────────────────────────────────────────────────────────────────
def load_raw(path=None) -> pd.DataFrame:
    csv_path = settings.data_raw if path is None else path

    if not csv_path.exists():
        msg = (
            f"Raw CSV not found at: {csv_path}\n"
            f"  → Place 'Speed+Dating+Data.csv' in {csv_path.parent} "
            f"or override SD_DATA_RAW in your .env file."
        )
        log.error(msg)
        raise RawFileNotFoundError(msg)

    log.info("Loading raw CSV from {path}", path=csv_path)
    try:
        df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError) as e:
        raise DataEngineeringError(f"Failed to parse {csv_path}: {e}") from e

    if "income" in df.columns:
        df["income"] = (
            df["income"].astype(str)
            .str.replace(",", "", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )

    log.success("Raw loaded: {n_rows} rows × {n_cols} cols",
                n_rows=df.shape[0], n_cols=df.shape[1])
    return df


# ─────────────────────────────────────────────────────────────────────
# 2. Schema validation
# ─────────────────────────────────────────────────────────────────────
def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    try:
        RawDatasetContract.validate_dataframe(df)
    except SchemaValidationError as e:
        log.error("Raw schema validation FAILED: {err}", err=str(e))
        raise DataEngineeringError(f"Raw schema invalid: {e}") from e
    log.info("Raw schema validated ✓")
    return df


# ─────────────────────────────────────────────────────────────────────
# 3. Structural vs random missings
# ─────────────────────────────────────────────────────────────────────
def flag_structural_missings(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags for STRUCTURAL missings (T2/T3 not collected)."""
    df = df.copy()

    leak_present = [c for c in LEAKAGE_COLS if c in df.columns]
    if leak_present:
        flags = df[leak_present].isna().astype(np.int8)
        flags.columns = [f"_STRUCTURAL_{c}" for c in flags.columns]
        df = pd.concat([df, flags], axis=1)
        log.debug("Added {n} structural-missing flags", n=len(flags.columns))

    if "income" in df.columns:
        df["income_missing"] = df["income"].isna().astype(np.int8)

    return df


# ─────────────────────────────────────────────────────────────────────
# 4. Stated-preference scale harmonisation
# ─────────────────────────────────────────────────────────────────────
def harmonise_stated_preference_scales(df: pd.DataFrame) -> pd.DataFrame:
    """Bring all waves onto a common 100-pt allocation, then renormalise rows."""
    df = df.copy()
    available = [c for c in STATED_PREF_COLS if c in df.columns]
    if len(available) < len(STATED_PREF_COLS):
        log.warning("Some stated-pref columns missing: {miss}",
                    miss=set(STATED_PREF_COLS) - set(available))

    mask_likert = df["wave"].isin(LIKERT_WAVES)
    if mask_likert.any():
        df.loc[mask_likert, available] = df.loc[mask_likert, available] * 10
        log.debug("Scaled {n} Likert-wave rows ×10", n=int(mask_likert.sum()))

    row_sums = df[available].sum(axis=1).replace(0, np.nan)
    df[available] = df[available].div(row_sums, axis=0) * 100

    log.info("Preference scales harmonised (waves 6-9 ×10 + row-renorm)")
    return df


# ─────────────────────────────────────────────────────────────────────
# 5. CLR (Centered Log-Ratio) transform
# ─────────────────────────────────────────────────────────────────────
def clr_transform(
    df: pd.DataFrame,
    source_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    epsilon: float = 0.5,
) -> pd.DataFrame:
    """
    CLR transform — maps Aitchison simplex isometrically to ℝᴰ.

        CLR(xᵢ) = log(xᵢ + ε) − (1/D) · Σⱼ log(xⱼ + ε)
    """
    source_cols = source_cols or STATED_PREF_COLS
    target_cols = target_cols or STATED_PREF_CLR

    df = df.copy()
    available = [c for c in source_cols if c in df.columns]
    if not available:
        log.warning("No source columns for CLR — skipping")
        return df

    X = df[available].to_numpy(dtype=float, na_value=np.nan)
    X_eps = X + epsilon
    log_X = np.log(X_eps)
    log_gm = log_X.mean(axis=1, keepdims=True)
    X_clr = log_X - log_gm

    clr_block = pd.DataFrame(X_clr, columns=target_cols, index=df.index)
    df = pd.concat([df, clr_block], axis=1)

    log.info("CLR transform applied → {n} columns", n=len(target_cols))
    return df


# ─────────────────────────────────────────────────────────────────────
# 6. Random-missing imputation
# ─────────────────────────────────────────────────────────────────────
def impute_random_missings(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation within wave × gender strata for RANDOM missings."""
    df = df.copy()

    impute_cols = (
        RATINGS_GIVEN + RATINGS_RECV + PARTNER_PREF_COLS + LIFESTYLE_COLS
        + ["age", "age_o", "imprace", "imprelig", "income"]
    )
    impute_cols = [c for c in impute_cols if c in df.columns]

    n_imputed = 0
    for col in impute_cols:
        is_null = df[col].isna()
        if not is_null.any():
            continue

        strat_median = df.groupby(["wave", "gender"], observed=True)[col].transform("median")
        global_median = df[col].median()

        if pd.isna(global_median):
            raise ImputationError(
                f"Column '{col}' has no valid values to compute global median."
            )

        df[col] = df[col].fillna(strat_median).fillna(global_median)
        n_imputed += int(is_null.sum())

    log.info("Imputed {n} random missings via stratified median", n=n_imputed)
    return df


# ─────────────────────────────────────────────────────────────────────
# 7. Outlier clipping
# ─────────────────────────────────────────────────────────────────────
def clip_rating_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rating_cols = [c for c in RATINGS_GIVEN + RATINGS_RECV if c in df.columns]
    if rating_cols:
        before = df[rating_cols].apply(lambda s: ((s < 1) | (s > 10)).sum()).sum()
        df[rating_cols] = df[rating_cols].clip(lower=1, upper=10)
        if before:
            log.debug("Clipped {n} rating values outside [1,10]", n=int(before))
    return df


# ─────────────────────────────────────────────────────────────────────
# 8. Derived features
# ─────────────────────────────────────────────────────────────────────
def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for dim in ATTR_DIMS:
        stated_col, rating_col = f"{dim}1_1", dim
        if stated_col in df.columns and rating_col in df.columns:
            df[f"{dim}_gap"] = df[rating_col] - (df[stated_col] / 10)

    recv_cols = [f"{d}_o" for d in ATTR_DIMS if f"{d}_o" in df.columns]
    if recv_cols:
        df["partner_avg_rating"] = df[recv_cols].mean(axis=1)

    df["selectivity"] = df.groupby("iid")["dec"].transform("mean")

    if "dec_o" in df.columns:
        df["dec_o_numeric"] = pd.to_numeric(df["dec_o"], errors="coerce")

    log.info("Derived features built (gap, selectivity, partner_avg_rating)")
    return df


# ─────────────────────────────────────────────────────────────────────
# 9. Anti-leakage final filter → model-ready dataset
# ─────────────────────────────────────────────────────────────────────
def build_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in MODEL_FEATURES if c in df.columns]
    gap_cols = [c for c in df.columns if c.endswith("_gap")]
    extras = ["selectivity", "partner_avg_rating", "iid", "wave", MODEL_TARGET]

    keep = list(dict.fromkeys(available + gap_cols + extras))
    keep = [c for c in keep if c in df.columns]

    model_df = df[keep].dropna(subset=[MODEL_TARGET]).copy()
    model_df[MODEL_TARGET] = model_df[MODEL_TARGET].astype(int)

    log.info(
        "Model-ready dataset: {shape} | target positive rate {rate:.2%}",
        shape=model_df.shape, rate=model_df[MODEL_TARGET].mean(),
    )
    return model_df


# ─────────────────────────────────────────────────────────────────────
# 10. Orchestrator
# ─────────────────────────────────────────────────────────────────────
def run_data_engineering() -> PipelineResult:
    """Execute Stage 1 end-to-end. Returns a typed `PipelineResult`."""
    log.info("─" * 60)
    log.info("STAGE 1 | Data Engineering & Cleaning")
    log.info("─" * 60)

    settings.out_data.mkdir(parents=True, exist_ok=True)

    try:
        df = load_raw()
        n_raw = len(df)

        df = validate_raw(df)
        df = flag_structural_missings(df)
        df = harmonise_stated_preference_scales(df)
        df = clr_transform(df)
        df = impute_random_missings(df)
        df = clip_rating_outliers(df)
        df = build_derived_features(df)

        try:
            CleanDatasetContract.validate_dataframe(df)
        except SchemaValidationError as e:
            raise DataEngineeringError(f"Clean schema invalid: {e}") from e

        df.to_parquet(settings.clean_parquet, index=False)
        log.success("Clean parquet → {p}", p=settings.clean_parquet)

        model_df = build_model_ready_dataset(df)
        model_df.to_parquet(settings.model_parquet, index=False)
        log.success("Model parquet → {p}", p=settings.model_parquet)

        result = PipelineResult(
            clean_df=df,
            model_df=model_df,
            n_rows_raw=n_raw,
            n_rows_clean=len(df),
            n_rows_model=len(model_df),
            target_positive_rate=float(model_df[MODEL_TARGET].mean()),
        )
        log.success("Stage 1 complete ✓")
        return result

    except DataEngineeringError:
        raise
    except Exception as e:
        log.exception("Unexpected failure in data-engineering stage")
        raise DataEngineeringError(f"Stage 1 failed: {e}") from e


if __name__ == "__main__":
    run_data_engineering()
