# Speed Dating Analysis

End-to-end data science pipeline on the Columbia Speed Dating dataset (Fisman et al. 2006), exploring the gap between **stated** romantic preferences and **revealed** behaviour during 4-minute speed-dates.

> `CV AUC 0.918 ± 0.008` ·  `13 figures`  ·  `4 reports`  ·  `8378 dyadic interactions`

---

## Project Overview

When asked beforehand what they look for in a partner, people allocate 100 points across six attributes (attractiveness, sincerity, intelligence, fun, ambition, shared interests). Four minutes later, after meeting 10-20 strangers, what *actually* drives their YES decision?

This project answers that question with a four-stage pipeline:

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | `01_data_engineering.py` | Schema validation, CLR transform, anti-leakage cleaning |
| 2 | `02_eda_statistics.py`   | Stated vs revealed preferences, gender effects, hypothesis tests |
| 3 | `03_modeling.py`         | XGBoost with GroupKFold + SHAP explainability |
| 4 | `04_network_analysis.py` | Bipartite YES-graph, HITS authority/hub scores |

## Key Results

- **Predictive performance:** XGBoost reaches **AUC 0.918 ± 0.008** with leak-safe `GroupKFold` on participant id (no participant in both train and validation).
- **Stated vs revealed gap:** Both genders state preferences that diverge sharply from what predicts a YES decision (quantified via SHAP global importance).
- **Network structure:** The 551-node bipartite YES-graph reveals desirability hubs and selectivity asymmetries detected via HITS.
- **CLR transform:** Applied to the compositional preference vectors (Aitchison 1986) so they can enter regression without unit-sum collinearity.

## Tech Stack

**Configuration & validation:** `pydantic-settings` for `.env`-driven config, `pydantic` schema contracts validated at pipeline entry.
**Logging:** `loguru` with rotated daily log files in `outputs/logs/`.
**Modelling:** `xgboost` (pinned `<2.0` for SHAP compatibility), `shap` for global / dependence / interaction explanations.
**Data:** `pandas` 2.x, `numpy`, `pyarrow` (Parquet I/O).
**Networks:** `networkx` 3.x.
**Visualisation:** `matplotlib` + `seaborn` (dark scientific palette); `plotly` planned for interactive dashboard.
**Testing:** `pytest` with 6 contract tests including the CLR row-mean = 0 mathematical property.

## How to Run

### 1. Clone & install dependencies

\\\ash
git clone https://github.com/anafiiliipa-dev/speed-dating-analysis.git
cd speed-dating-analysis
pip install -r requirements.txt
\\\

### 2. Place the raw dataset

Download `Speed Dating Data.csv` from the [Kaggle mirror](https://www.kaggle.com/datasets/whenamancodes/speed-dating) of the Fisman et al. (2006) study and place it at:

\\\
data/Speed+Dating+Data.csv
\\\

### 3. Configure environment

\\\ash
cp .env.example .env
\\\

Defaults work for most users; edit `.env` only if you want to change the log level, random seed, or data path.

### 4. Run the pipeline

\\\ash
# Run all four stages end-to-end (~16s on a modern laptop)
python run_pipeline.py

# Or run only Stage 1 (data engineering)
python 01_data_engineering.py
\\\

Outputs land in:
- `outputs/data/` — clean & model-ready Parquet files
- `outputs/figures/` — 13 PNG figures
- `outputs/reports/` — 4 CSV summaries (CV metrics, statistical tests, network metrics)
- `outputs/logs/` — rotated daily logs

### 5. Run the test suite

\\\ash
python -m pytest tests/ -v
\\\

Expected: **6 passed**. Tests skip gracefully if the raw CSV is not present.

## Repository Structure

\\\
speed-dating-analysis/
├── .env.example              # template for runtime config
├── .gitignore
├── pyproject.toml            # ruff + pytest configuration
├── requirements.txt          # pinned dependencies
├── README.md
├── run_pipeline.py           # master orchestrator (4 stages)
│
├── config.py                 # pydantic-settings + domain constants
├── logging_config.py         # loguru setup
├── schemas.py                # pydantic dataframe contracts
│
├── 01_data_engineering.py    # Stage 1
├── 02_eda_statistics.py      # Stage 2
├── 03_modeling.py            # Stage 3
├── 04_network_analysis.py    # Stage 4
│
├── data/                     # raw CSV (gitignored)
├── outputs/                  # all generated artefacts (gitignored)
└── tests/
    └── test_data_engineering.py
\\\

## Engineering Decisions

- **Anti-leakage protocol:** All `_2` (next-day) and `_3` (3-week) survey columns are excluded from the model matrix. `GroupKFold` ensures no participant appears in both train and validation folds.
- **Structural vs random missings:** Missing values are flagged as STRUCTURAL (a wave didn't collect that column) versus RANDOM (participant skipped). Only random missings are imputed (stratified median by wave × gender).
- **Scale harmonisation:** Waves 6-9 used a 1-10 Likert scale for stated preferences while other waves used a 100-point budget allocation. We multiply Likert waves ×10 and row-renormalise to sum=100 per Fisman et al.
- **CLR (Centered Log-Ratio) transform:** Stated preferences live on the Aitchison simplex. CLR maps them isometrically to ℝᴰ where Euclidean operations are valid. The defining property `mean(CLR row) = 0` is verified in the test suite.

## Roadmap

- [x] Stage 1 senior refactor (pydantic-settings, loguru, schema validation, contract tests)
- [ ] Stage 2-4 senior refactor (same patterns)
- [ ] Streamlit interactive dashboard
- [ ] Architecture diagram & badges
- [ ] GitHub Actions CI

## License

MIT
