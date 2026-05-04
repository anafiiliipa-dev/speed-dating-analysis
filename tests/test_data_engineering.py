"""Smoke tests for Stage 1 — Data Engineering.

Contract tests, not unit tests of every function.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

# pyproject sets pythonpath=["."] so config/schemas resolve from project root.
# 01_data_engineering.py can't be imported normally because module names
# can't start with a digit — load it via importlib.
_PIPELINE_PATH = Path(__file__).resolve().parents[1] / "01_data_engineering.py"
_spec = importlib.util.spec_from_file_location("step01_data_engineering", _PIPELINE_PATH)
step01 = importlib.util.module_from_spec(_spec)
sys.modules["step01_data_engineering"] = step01
_spec.loader.exec_module(step01)

from config import settings
from schemas import CleanDatasetContract, RawDatasetContract, SchemaValidationError


@pytest.fixture(scope="module")
def raw_df():
    if not settings.data_raw.exists():
        pytest.skip(f"Raw CSV not present at {settings.data_raw}")
    return step01.load_raw()


def test_raw_loader_returns_nonempty(raw_df):
    assert len(raw_df) > 1000
    assert "iid" in raw_df.columns
    assert "dec" in raw_df.columns


def test_raw_schema_validates(raw_df):
    RawDatasetContract.validate_dataframe(raw_df)


def test_clr_columns_have_zero_row_mean(raw_df):
    """Defining property of CLR: each transformed row sums to 0."""
    df = step01.harmonise_stated_preference_scales(raw_df)
    df = step01.clr_transform(df)
    clr_cols = [c for c in df.columns if c.endswith("_clr")]
    row_means = df[clr_cols].mean(axis=1)
    assert row_means.abs().max() < 1e-9


def test_full_pipeline_runs(raw_df):
    result = step01.run_data_engineering()
    assert result.n_rows_clean == result.n_rows_raw
    assert result.n_rows_model > 0
    assert 0 < result.target_positive_rate < 1
    CleanDatasetContract.validate_dataframe(result.clean_df)


def test_missing_file_raises():
    with pytest.raises(step01.RawFileNotFoundError):
        step01.load_raw(path=Path("/this/path/does/not/exist.csv"))


def test_validate_rejects_empty_frame():
    with pytest.raises(SchemaValidationError):
        RawDatasetContract.validate_dataframe(pd.DataFrame())