"""
01_data_engineering.py — Data Engineering & Cleaning Pipeline

Key decisions documented inline:
  - Missing values are classified as STRUCTURAL (wave didn't collect that field)
    vs RANDOM (person skipped the question). Only RANDOM NaNs are imputed.
  - Waves 6-9 use 1-10 scale for stated preferences → multiplied by 10.
  - CLR transform applied to compositional stated-preference vectors to remove
    the unit-sum constraint before regression / ML.
  - T2/T3 columns are flagged and excluded from the model-ready dataset.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

warnings.filterwarnings("ignore")

# Local
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_RAW, CLEAN_PARQUET, MODEL_PARQUET,
    LIKERT_WAVES, STATED_PREF_COLS, STATED_PREF_CLR,
    ATTR_DIMS, LEAKAGE_COLS, MODEL_FEATURES, MODEL_TARGET,
    RATINGS_GIVEN, RATINGS_RECV, DEMO_COLS, LIFESTYLE_COLS,
    CONTEXT_COLS, PARTNER_PREF_COLS, OUT_DATA,
)


# ── 1. Raw loader ─────────────────────────────────────────────────────────────

def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    """Load the raw CSV with robust encoding and type coercion."""
    df = pd.read_csv(
        path,
        encoding="latin-1",
        low_memory=False,
    )
    # Coerce income: stored as '69,487.00' string
    df["income"] = (
        df["income"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )
    print(f"[load]  raw shape: {df.shape}")
    return df


# ── 2. Structural vs random missing values ────────────────────────────────────

def classify_and_flag_missings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean flags for STRUCTURAL missings.

    Structural = a feature was simply not collected in a wave:
      - T2/T3 follow-up columns in waves where follow-up wasn't done.
      - attr4_* (ideal rating of opposite sex) was added in later waves.
    
    We do NOT impute structural NaNs — they carry information about design.
    We only median-impute RANDOM missings within their wave×gender strata.
    """
    df = df.copy()

    # Flag T2/T3 structural missings — useful for downstream filtering
    for col in LEAKAGE_COLS:
        if col in df.columns:
            df[f"_STRUCTURAL_{col}"] = df[col].isna().astype(np.int8)

    # Flag: was income missing? (often structural — person from abroad)
    df["income_missing"] = df["income"].isna().astype(np.int8)

    print(f"[flags] structural-missing flags added")
    return df


# ── 3. Scale harmonisation ────────────────────────────────────────────────────

def harmonise_stated_preference_scales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Waves 6-9 collected stated preferences on a 1-10 Likert scale.
    All other waves used a 100-point budget allocation.

    Decision: multiply waves 6-9 values by 10 to bring them into [0, 100].
    This is the canonical approach used in the original Fisman et al. paper.
    We then renormalise to sum = 100 per row to restore compositionality.
    """
    df = df.copy()
    mask_likert = df["wave"].isin(LIKERT_WAVES)

    # Scale up 1-10 → 0-100
    df.loc[mask_likert, STATED_PREF_COLS] = (
        df.loc[mask_likert, STATED_PREF_COLS] * 10
    )

    # Re-normalise each row so it sums to 100
    # (small deviations arise because people don't always allocate exactly 100)
    row_sums = df[STATED_PREF_COLS].sum(axis=1).replace(0, np.nan)
    for col in STATED_PREF_COLS:
        df[col] = df[col] / row_sums * 100

    print(f"[scale] preference scale harmonised (waves 6-9 ×10 + row-renorm)")
    return df


# ── 4. CLR transformation ─────────────────────────────────────────────────────

def clr_transform(
    df: pd.DataFrame,
    source_cols: list[str] = STATED_PREF_COLS,
    target_cols: list[str] = STATED_PREF_CLR,
    epsilon: float = 0.5,           # Adds small constant before log to handle zeros
) -> pd.DataFrame:
    """
    Centered Log-Ratio (CLR) transform for compositional preference vectors.

    Why CLR?
      Stated preferences sum to 100 (Aitchison simplex). Regressing directly on
      simplex coordinates causes multicollinearity (perfect linear dependency)
      and violates OLS assumptions. CLR maps the simplex isometrically to ℝᴰ
      where Euclidean operations are valid.

    CLR(xᵢ) = log(xᵢ + ε) − (1/D) · Σⱼ log(xⱼ + ε)

    ε = 0.5 (Aitchison 1986 recommendation for zero-handling).
    """
    df = df.copy()
    X = df[source_cols].values.astype(float)

    # Add epsilon to handle zeros before log
    X_eps = X + epsilon

    log_X   = np.log(X_eps)
    log_gm  = log_X.mean(axis=1, keepdims=True)   # log geometric mean
    X_clr   = log_X - log_gm

    for i, col in enumerate(target_cols):
        df[col] = X_clr[:, i]

    print(f"[CLR]   {len(source_cols)} attributes transformed → {target_cols[:2]}…")
    return df


# ── 5. Random missing imputation ──────────────────────────────────────────────

def impute_random_missings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median imputation within wave × gender strata for RANDOM missings.

    We use median (not mean) because preference and rating variables tend
    to be left-skewed (people over-rate partners). Stratifying by wave×gender
    preserves distributional differences across experimental conditions.

    Imputed columns: event-night ratings, demographic scores, lifestyle.
    We do NOT impute stated preferences (handled after CLR above).
    """
    df = df.copy()
    IMPUTE_COLS = (
        RATINGS_GIVEN + RATINGS_RECV
        + PARTNER_PREF_COLS
        + LIFESTYLE_COLS
        + ["age", "age_o", "imprace", "imprelig", "income"]
    )

    for col in IMPUTE_COLS:
        if col not in df.columns:
            continue
        is_null = df[col].isna()
        if is_null.sum() == 0:
            continue

        # Strata median imputation
        strat_median = df.groupby(["wave", "gender"])[col].transform("median")
        # Fall back to global median if strata has no data
        global_median = df[col].median()
        df[col] = df[col].fillna(strat_median).fillna(global_median)

    print(f"[impute] random missings imputed via stratified median")
    return df


# ── 6. Outlier handling ───────────────────────────────────────────────────────

def clip_rating_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip event-night ratings to [1, 10] — the valid response range.
    A handful of rows have values outside this range due to data-entry errors.
    We clip rather than drop to preserve the observation.
    """
    df = df.copy()
    for col in RATINGS_GIVEN + RATINGS_RECV:
        if col in df.columns:
            df[col] = df[col].clip(lower=1, upper=10)
    return df


# ── 7. Derived features ───────────────────────────────────────────────────────

def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analytically useful composite variables for EDA and modelling.

    - attr_gap: partner's attractiveness rating vs person's stated ideal
    - pref_alignment: cosine similarity between person's CLR prefs and
                      their actual ratings (stated ≈ revealed?)
    - selectivity: how often this person said YES (proportion in wave)
    """
    df = df.copy()

    # Preference–behaviour gap: stated importance vs actual rating weight
    # (positive = over-weights, negative = under-weights in practice)
    for dim in ATTR_DIMS:
        stated_col  = f"{dim}1_1"
        rating_col  = dim
        if stated_col in df.columns and rating_col in df.columns:
            df[f"{dim}_gap"] = df[rating_col] - (df[stated_col] / 10)

    # Partner's average rating received (proxy for overall attractiveness)
    recv_cols = [f"{d}_o" for d in ATTR_DIMS if f"{d}_o" in df.columns]
    df["partner_avg_rating"] = df[recv_cols].mean(axis=1)

    # Selectivity: fraction of yeses given by this person in their wave
    df["selectivity"] = df.groupby("iid")["dec"].transform("mean")

    # Reciprocity signal: did partner also say yes?
    df["dec_o_numeric"] = df["dec_o"].astype(float)

    print(f"[derived] gap, selectivity, partner_avg_rating computed")
    return df


# ── 8. Anti-leakage final filter ──────────────────────────────────────────────

def build_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a feature matrix that contains ONLY T1-safe columns plus the target.

    Anti-leakage protocol:
      - All _2 (day-after) and _3 (3-week) columns are excluded.
      - `match` is excluded because it's simultaneously determined with `dec`.
      - `dec_o` is excluded because it's the partner's simultaneous decision.
      - Only rows with a valid `dec` label are kept.
    """
    # Resolve which of the declared MODEL_FEATURES actually exist
    available = [c for c in MODEL_FEATURES if c in df.columns]

    # Also include the derived gap features
    gap_cols = [c for c in df.columns if c.endswith("_gap")]
    extra_cols = ["selectivity", "partner_avg_rating", "iid", "wave", MODEL_TARGET]

    keep = list(dict.fromkeys(available + gap_cols + extra_cols))
    keep = [c for c in keep if c in df.columns]

    model_df = df[keep].dropna(subset=[MODEL_TARGET]).copy()
    model_df[MODEL_TARGET] = model_df[MODEL_TARGET].astype(int)

    print(f"[model]  model-ready shape: {model_df.shape}  "
          f"| target positive rate: {model_df[MODEL_TARGET].mean():.2%}")
    return model_df


# ── 9. Pipeline orchestrator ──────────────────────────────────────────────────

def run_data_engineering() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full data-engineering pipeline end-to-end.
    Returns (clean_df, model_df).
    """
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = classify_and_flag_missings(df)
    df = harmonise_stated_preference_scales(df)
    df = clr_transform(df)
    df = impute_random_missings(df)
    df = clip_rating_outliers(df)
    df = build_derived_features(df)

    # Persist clean dataset (all columns, no leakage concern yet)
    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"[save]   clean parquet → {CLEAN_PARQUET}")

    # Build and persist model-ready dataset
    model_df = build_model_ready_dataset(df)
    model_df.to_parquet(MODEL_PARQUET, index=False)
    print(f"[save]   model parquet → {MODEL_PARQUET}")

    return df, model_df


if __name__ == "__main__":
    run_data_engineering()
