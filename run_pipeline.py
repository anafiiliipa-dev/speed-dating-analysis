"""
run_pipeline.py — Master Orchestrator for the Speed Dating Analysis Pipeline
"""

import time
import sys
import importlib.util
from pathlib import Path


def load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
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
    t0   = time.time()

    banner("STAGE 1 | Data Engineering & Cleaning")
    de = load_module("data_engineering", root / "01_data_engineering.py")
    clean_df, model_df = de.run_data_engineering()

    banner("STAGE 2 | EDA & Statistical Inference")
    eda = load_module("eda_statistics", root / "02_eda_statistics.py")
    stats_df = eda.run_eda_statistics()

    banner("STAGE 3 | XGBoost + SHAP Explainability")
    mod = load_module("modeling", root / "03_modeling.py")
    model = mod.run_modeling()

    banner("STAGE 4 | Bipartite Network Analysis")
    net = load_module("network_analysis", root / "04_network_analysis.py")
    G = net.run_network_analysis()

    elapsed = time.time() - t0
    from config import OUT_FIGS, OUT_REPORTS
    figs    = list(OUT_FIGS.glob("*.png"))
    reports = list(OUT_REPORTS.glob("*.csv"))

    banner(f"PIPELINE COMPLETE  |  {elapsed:.1f}s")
    print(f"\n  Figures   : {len(figs)}")
    for f in sorted(figs):
        print(f"    - {f.name}")
    print(f"\n  Reports   : {len(reports)}")
    for r in sorted(reports):
        print(f"    - {r.name}")


if __name__ == "__main__":
    run_pipeline()
