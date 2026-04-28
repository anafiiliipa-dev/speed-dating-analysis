# Speed Dating — Stated vs. Revealed Preferences
### An End-to-End Data Science Pipeline · Columbia University Dataset

> *"People say they want sincerity. The model says otherwise."*

---

## Overview

This project is a full end-to-end data science pipeline built on the **Columbia University Speed Dating dataset** (Fisman et al., 2006) — 8,378 speed dates across 21 experimental waves, involving 551 participants.

The central question: **Is there a measurable gap between what people say they want in a partner and what actually drives their decisions?**

We answer this using a combination of statistical inference, gradient boosting with SHAP explainability, and bipartite network analysis — with every design decision documented and justified in the code.

---

## Key Findings

| Finding | Detail |
|---|---|
| **Model AUC** | 0.919 ± 0.006 (GroupKFold, groups = individual) |
| **Stated ≠ Revealed** | Welch T-test significant (p < 0.0001) for all 6 attributes, both genders |
| **Top revealed driver** | Attractiveness rating + "Overall Like" dominate SHAP importance |
| **Race effect** | Chi-square not significant (p = 0.245) when other factors are controlled |
| **Network density** | 0.012 — sparse, highly selective decision graph |
| **Hub archetype** | Top hubs are women with hub_score = 1.0 and selectivity > 0.80 |

The gap between **stated importance** (what people allocate budget to in a survey) and **revealed preference** (what the XGBoost model actually uses to predict YES) is statistically and practically significant across every attribute dimension.

---

## Project Structure

```
speed_dating/
│
├── config.py                  # Single source of truth: paths, column groups,
│                              # wave metadata, visual constants
│
├── 01_data_engineering.py     # Raw loading → clean + model-ready parquets
│                              # Scale harmonisation, CLR transform, imputation
│
├── 02_eda_statistics.py       # 5 publication-quality figures
│                              # T-tests, Chi-Square, ANOVA, clustered-SE OLS
│
├── 03_modeling.py             # XGBoost + GroupKFold CV
│                              # SHAP global, dependence, interaction, gender split
│
├── 04_network_analysis.py     # Bipartite YES-graph · HITS · hub/selectivity scores
│
├── run_pipeline.py            # Single-command orchestrator
│
├── requirements.txt
│
└── outputs/
    ├── figures/               # 13 PNG figures, presentation-ready
    ├── data/                  # Parquet files (clean + model-ready)
    └── reports/               # CSV reports (stats, CV metrics, node metrics)
```

---

## Architecture Decisions

### 1. Data Engineering

**Structural vs. random missings** — The dataset has two types of NaN: fields that were never collected in a given wave (structural) and questions that participants skipped (random). We flag structural missings rather than imputing them, preserving experimental design information. Only random missings are imputed via stratified median (wave × gender).

**Scale harmonisation** — Waves 6–9 used a 1–10 Likert scale for stated attribute preferences; all other waves used a 100-point budget allocation. We multiply waves 6–9 values by 10 and renormalise rows to sum = 100, following the canonical approach from Fisman et al.

**CLR transformation** — Stated preferences live on the Aitchison simplex (sum-constrained). Regressing directly on simplex coordinates induces perfect multicollinearity. Centered Log-Ratio (CLR) maps the composition isometrically to ℝ⁶, where Euclidean operations are valid and coefficients are interpretable.

```
CLR(xᵢ) = log(xᵢ + ε) − (1/D) · Σⱼ log(xⱼ + ε)     ε = 0.5
```

### 2. Anti-Leakage Protocol

All T2 (next-day survey) and T3 (3-week follow-up) columns are identified in `config.py` and excluded from the feature matrix before any modelling step. The target `match` is also excluded — it is simultaneously determined with `dec` and would constitute direct leakage. The pipeline raises no exceptions on this because the exclusion is enforced at the config level, not as an afterthought.

### 3. GroupKFold by `iid`

Standard K-Fold would allow the same individual to appear in both train and validation splits, creating an optimistic AUC. `GroupKFold(groups=iid)` ensures each fold tests the model on **entirely new people** — the deployment scenario that matters.

### 4. SHAP Explainability

Three levels of explanation are produced:
- **Global importance** — mean |SHAP value| per feature, top 20
- **Dependence plots** — how the top-3 features relate to their SHAP contribution, coloured by the highest-covariance interaction feature
- **Interaction heatmap** — SHAP vector correlations across top-10 features, revealing which signals fire together

### 5. Network Analysis

A directed bipartite graph (Men → Women and Women → Men, edge = YES decision) is constructed via NetworkX. Node metrics computed: hub score (desirability), selectivity (pickiness), reciprocity, and HITS authority/hub scores. A quadrant scatter — Hub Score × Selectivity — reveals four strategic archetypes in the dating market.

---

## Outputs: 13 Figures

| # | Filename | Description |
|---|---|---|
| 01 | `01_stated_vs_revealed_heatmap` | Core finding: stated allocation vs. Pearson r with dec |
| 02 | `02_match_rate_analysis` | YES rate by gender, match rate by race, attr distribution |
| 03 | `03_pref_behaviour_gap` | Violin: actual rating weight − stated importance |
| 04 | `04_correlation_matrix` | Stated (CLR) ↔ revealed ratings full correlation matrix |
| 05 | `05_gender_preference_gap` | Stated vs revealed lift, men vs women side-by-side |
| 06 | `06_model_diagnostics` | ROC, Precision-Recall, Calibration (all OOF) |
| 07 | `07_shap_global_importance` | Top-20 SHAP features bar chart |
| 08 | `08_shap_dependence` | Dependence plots for top-3 features |
| 09 | `09_shap_interaction_heatmap` | SHAP vector correlation matrix, top-10 features |
| 10 | `10_shap_by_gender` | Gender-stratified SHAP importance comparison |
| 11 | `11_network_wave*` | Bipartite YES-graph for a sampled wave |
| 12 | `12_hub_vs_selectivity` | Scatter: desirability × pickiness, 4 archetypes |
| 13 | `13_hits_distribution` | KDE of HITS authority and hub scores by gender |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the raw CSV at the path defined in config.py (DATA_RAW)

# 3. Run the full pipeline
python run_pipeline.py
```

To run individual stages:

```bash
python 01_data_engineering.py   # produces outputs/data/*.parquet
python 02_eda_statistics.py     # produces figures 01–05 + statistical_tests.csv
python 03_modeling.py           # produces figures 06–10 + cv_metrics.csv
python 04_network_analysis.py   # produces figures 11–13 + network CSVs
```

---

## Dataset

Fisman, R., Iyengar, S. S., Kamenica, E., & Simonson, I. (2006).  
*Gender Differences in Mate Selection: Evidence from a Speed Dating Experiment.*  
Quarterly Journal of Economics, 121(2), 673–697.

Data available at: [Columbia Business School — Speed Dating Experiment](http://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/)

---

## Tech Stack

`pandas` · `numpy` · `scipy` · `scikit-learn` · `xgboost` · `shap` · `networkx` · `matplotlib` · `seaborn` · `plotly`
