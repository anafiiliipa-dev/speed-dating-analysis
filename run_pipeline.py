"""
run_pipeline.py - Master Orchestrator for the Speed Dating Analysis Pipeline.

Runs the four stages sequentially:
    Stage 1 - Data Engineering & Cleaning
    Stage 2 - EDA & Statistical Inference
    Stage 3 - XGBoost + SHAP Explainability
    Stage 4 - Bipartite Network Analysis

Each stage is loaded dynamically from its numbered script.
A typed PipelineResult is returned by Stage 1; later stages return either
DataFrames, fitted models, or NetworkX graphs - we just keep references.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

from logging_config import get_logger

log = get_logger(__name__)


def load_module(name: str, filepath: Path):
    """Dynamically load a module whose filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def banner(title: str) -> None:
    w = 64
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def run_pipeline() -> None:
    root = Path(__file__).parent
    t0 = time.time()

    # Stage 1
    banner("STAGE 1 | Data Engineering & Cleaning")
    de = load_module("data_engineering", root / "01_data_engineering.py")
    result = de.run_data_engineering()
    log.info(
        "Stage 1 result: raw={raw}, clean={clean}, model={model}, yes_rate={rate:.2%}",
        raw=result.n_rows_raw,
        clean=result.n_rows_clean,
        model=result.n_rows_model,
        rate=result.target_positive_rate,
    )

    # Stage 2
    banner("STAGE 2 | EDA & Statistical Inference")
    eda = load_module("eda_statistics", root / "02_eda_statistics.py")
    eda.run_eda_statistics()

    # Stage 3
    banner("STAGE 3 | XGBoost + SHAP Explainability")
    mod = load_module("modeling", root / "03_modeling.py")
    mod.run_modeling()

    # Stage 4
    banner("STAGE 4 | Bipartite Network Analysis")
    net = load_module("network_analysis", root / "04_network_analysis.py")
    net.run_network_analysis()

    # Summary
    elapsed = time.time() - t0
    from config import OUT_FIGS, OUT_REPORTS
    figs = sorted(OUT_FIGS.glob("*.png"))
    reports = sorted(OUT_REPORTS.glob("*.csv"))

    banner(f"PIPELINE COMPLETE  |  {elapsed:.1f}s")
    print(f"\n  Figures   : {len(figs)}")
    for f in figs:
        print(f"    - {f.name}")
    print(f"\n  Reports   : {len(reports)}")
    for r in reports:
        print(f"    - {r.name}")
    print()


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        log.exception("Pipeline failed: {err}", err=str(e))
        sys.exit(1)
