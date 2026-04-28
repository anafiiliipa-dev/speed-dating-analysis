"""
03_modeling.py — ML Modeling & SHAP Explainability

Pipeline:
  1. XGBoost classifier with GroupKFold cross-validation (groups = iid).
     GroupKFold ensures the model is evaluated on UNSEEN individuals,
     not just unseen dates — which is the real generalisation challenge.
  2. Calibration check (Brier score, log-loss).
  3. SHAP Global importance bar chart.
  4. SHAP Dependence plots for the top-3 features.
  5. SHAP Interaction heatmap (top features × top features).
  6. Gender-stratified SHAP comparison.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc, brier_score_loss, log_loss,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODEL_PARQUET, OUT_FIGS, OUT_REPORTS, OUT_DATA,
    MODEL_TARGET, ATTR_LABELS, GENDER_LABELS,
    SEABORN_STYLE, BG_COLOR, PANEL_COLOR, TEXT_COLOR,
    ACCENT_1, ACCENT_2, ACCENT_3, ACCENT_4,
)

warnings.filterwarnings("ignore")
plt.rcParams.update(SEABORN_STYLE)
plt.rcParams["font.family"] = "DejaVu Sans"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> Path:
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    path = OUT_FIGS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [fig] saved → {path.name}")
    return path


FEATURE_DISPLAY = {
    **{f"{d}1_1_clr": f"{ATTR_LABELS[d]} (stated CLR)" for d in ATTR_LABELS},
    **{d: f"{ATTR_LABELS[d]} rating" for d in ATTR_LABELS},
    **{f"{d}_o": f"{ATTR_LABELS.get(d, d)} (rcvd)" for d in ["attr","sinc","intel","fun","amb","shar"]},
    "like": "Overall Like", "prob": "Perceived Match Prob.",
    "like_o": "Partner Liked (rcvd)", "prob_o": "Partner Match Prob (rcvd)",
    "samerace": "Same Race", "int_corr": "Interest Correlation",
    "age": "Age", "age_o": "Partner Age",
    "selectivity": "Personal Selectivity",
    "partner_avg_rating": "Partner Avg Rating",
}


# ── 1. Prepare data ───────────────────────────────────────────────────────────

def prepare_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract X, y, groups from the model-ready dataset.
    Drops any column that is the target, iid, or wave (used for grouping only).
    Fills residual NaNs with column median (safety net — should be minimal).
    """
    forbidden = {MODEL_TARGET, "iid", "wave", "match", "dec_o",
                 "dec_o_numeric", "id", "idg", "pid", "partner"}
    feature_cols = [c for c in df.columns if c not in forbidden
                    and not c.startswith("_STRUCTURAL_")]

    X = df[feature_cols].copy()
    y = df[MODEL_TARGET].astype(int)
    groups = df["iid"]

    # Safety-net median fill (minimal, most imputation done in step 01)
    X = X.fillna(X.median(numeric_only=True))

    print(f"[model] X: {X.shape}  |  class balance: {y.mean():.2%} positive")
    return X, y, groups


# ── 2. XGBoost with GroupKFold ────────────────────────────────────────────────

XGB_PARAMS = {
    "n_estimators":      500,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,       # prevents overfitting on small groups
    "scale_pos_weight":  1,       # roughly balanced; adjust if needed
    "use_label_encoder": False,
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
    "tree_method":       "hist",  # fast exact approx
}


def train_with_group_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> tuple[xgb.XGBClassifier, dict, np.ndarray]:
    """
    GroupKFold CV ensuring that the same individual (iid) never appears
    in both train and validation — mimics deployment on NEW people.

    Returns:
      - final model trained on full data
      - cv_metrics dictionary
      - oof_probs  (out-of-fold predicted probabilities for calibration)
    """
    gkf = GroupKFold(n_splits=n_splits)
    cv_aucs, cv_briers, cv_logloss = [], [], []
    oof_probs = np.zeros(len(y))

    X_arr = X.values
    y_arr = y.values
    g_arr = groups.values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_arr, y_arr, g_arr)):
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        prob = model.predict_proba(X_va)[:, 1]
        oof_probs[va_idx] = prob

        cv_aucs.append(roc_auc_score(y_va, prob))
        cv_briers.append(brier_score_loss(y_va, prob))
        cv_logloss.append(log_loss(y_va, prob))
        print(f"  Fold {fold+1}/{n_splits}  AUC={cv_aucs[-1]:.3f}  "
              f"Brier={cv_briers[-1]:.3f}  LogLoss={cv_logloss[-1]:.3f}")

    cv_metrics = {
        "auc_mean":     np.mean(cv_aucs),
        "auc_std":      np.std(cv_aucs),
        "brier_mean":   np.mean(cv_briers),
        "logloss_mean": np.mean(cv_logloss),
    }
    print(f"\n  CV AUC: {cv_metrics['auc_mean']:.3f} ± {cv_metrics['auc_std']:.3f}")

    # Final model on full data
    final_model = xgb.XGBClassifier(**XGB_PARAMS)
    final_model.fit(X_arr, y_arr, verbose=False)
    return final_model, cv_metrics, oof_probs


# ── 3. ROC / PR / Calibration plots ──────────────────────────────────────────

def plot_model_diagnostics(
    y: pd.Series,
    oof_probs: np.ndarray,
    cv_metrics: dict,
) -> None:
    """Three-panel: ROC curve, PR curve, and calibration plot."""
    y_arr = y.values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG_COLOR)

    # ROC
    fpr, tpr, _ = roc_curve(y_arr, oof_probs)
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=ACCENT_1, linewidth=2,
                 label=f"AUC = {roc_auc_val:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=TEXT_COLOR, alpha=0.4)
    axes[0].set_title("ROC Curve (OOF)", color=TEXT_COLOR)
    axes[0].set_xlabel("False Positive Rate", color=TEXT_COLOR)
    axes[0].set_ylabel("True Positive Rate", color=TEXT_COLOR)
    axes[0].legend(facecolor=PANEL_COLOR, edgecolor="#2E3347",
                   labelcolor=TEXT_COLOR)
    axes[0].set_facecolor(PANEL_COLOR)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_arr, oof_probs)
    pr_auc = auc(rec, prec)
    axes[1].plot(rec, prec, color=ACCENT_2, linewidth=2,
                 label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].axhline(y_arr.mean(), color=TEXT_COLOR, linestyle="--",
                    alpha=0.4, label="Baseline (prevalence)")
    axes[1].set_title("Precision-Recall Curve (OOF)", color=TEXT_COLOR)
    axes[1].set_xlabel("Recall", color=TEXT_COLOR)
    axes[1].set_ylabel("Precision", color=TEXT_COLOR)
    axes[1].legend(facecolor=PANEL_COLOR, edgecolor="#2E3347",
                   labelcolor=TEXT_COLOR)
    axes[1].set_facecolor(PANEL_COLOR)

    # Calibration
    prob_true, prob_pred = calibration_curve(y_arr, oof_probs, n_bins=10)
    axes[2].plot(prob_pred, prob_true, "o-", color=ACCENT_3, linewidth=2,
                 label=f"Brier={cv_metrics['brier_mean']:.3f}")
    axes[2].plot([0, 1], [0, 1], "--", color=TEXT_COLOR, alpha=0.4,
                 label="Perfect calibration")
    axes[2].set_title("Calibration Curve (OOF)", color=TEXT_COLOR)
    axes[2].set_xlabel("Mean Predicted Probability", color=TEXT_COLOR)
    axes[2].set_ylabel("Fraction of Positives", color=TEXT_COLOR)
    axes[2].legend(facecolor=PANEL_COLOR, edgecolor="#2E3347",
                   labelcolor=TEXT_COLOR)
    axes[2].set_facecolor(PANEL_COLOR)

    fig.suptitle(
        "XGBoost  |  GroupKFold CV  (groups = individual)",
        fontsize=14, color=TEXT_COLOR, fontweight="bold",
    )
    fig.tight_layout(pad=2)
    _save(fig, "06_model_diagnostics")


# ── 4. SHAP Global importance ─────────────────────────────────────────────────

def plot_shap_global(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    sample_n: int = 2000,
) -> shap.Explanation:
    """
    Global SHAP importance (mean |SHAP value|) — bar chart.
    Uses TreeExplainer for speed; samples rows for tractability.
    """
    X_sample = X.sample(min(sample_n, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_sample)

    # Rename for display
    feat_names = [FEATURE_DISPLAY.get(c, c) for c in X.columns]

    # Mean absolute SHAP
    mean_abs = np.abs(shap_vals.values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:20]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(order)))
    y_pos = np.arange(len(order))
    ax.barh(y_pos, mean_abs[order], color=colors, edgecolor="#2E3347")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_names[i] for i in order], color=TEXT_COLOR)
    ax.set_xlabel("Mean |SHAP value|", color=TEXT_COLOR)
    ax.set_title(
        "Global Feature Importance  |  XGBoost SHAP\n"
        "→ Revealed preferences, not stated ones, drive the model",
        color=TEXT_COLOR, fontsize=13, fontweight="bold",
    )

    _save(fig, "07_shap_global_importance")
    return shap_vals, X_sample


# ── 5. SHAP Dependence plots ──────────────────────────────────────────────────

def plot_shap_dependence(
    shap_vals: shap.Explanation,
    X_sample: pd.DataFrame,
    top_n: int = 3,
) -> None:
    """
    SHAP dependence plots for the top-N features.
    Each point is a date; colour shows the interaction feature (auto-selected).
    """
    feat_names = list(X_sample.columns)
    mean_abs   = np.abs(shap_vals.values).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]

    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 5),
                             facecolor=BG_COLOR)
    if top_n == 1:
        axes = [axes]

    for ax, idx in zip(axes, top_idx):
        feat      = feat_names[idx]
        feat_disp = FEATURE_DISPLAY.get(feat, feat)

        # Interaction feature: highest |SHAP × feature| covariance (auto)
        covar = np.abs(
            (shap_vals.values - shap_vals.values.mean(0))
            * (X_sample.values - X_sample.values.mean(0))
        ).mean(0)
        covar[idx] = -1
        inter_idx  = np.argmax(covar)
        inter_feat = feat_names[inter_idx]
        inter_disp = FEATURE_DISPLAY.get(inter_feat, inter_feat)

        sc = ax.scatter(
            X_sample.iloc[:, idx],
            shap_vals.values[:, idx],
            c=X_sample.iloc[:, inter_idx],
            cmap="viridis", alpha=0.5, s=8, edgecolors="none",
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(inter_disp, color=TEXT_COLOR, fontsize=8)
        cb.ax.tick_params(colors=TEXT_COLOR)

        ax.axhline(0, color=TEXT_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(feat_disp, color=TEXT_COLOR, fontsize=11)
        ax.set_xlabel("Feature value", color=TEXT_COLOR)
        ax.set_ylabel("SHAP value", color=TEXT_COLOR)
        ax.set_facecolor(PANEL_COLOR)

    fig.suptitle(
        "SHAP Dependence Plots — Top Features",
        fontsize=14, color=TEXT_COLOR, fontweight="bold",
    )
    fig.tight_layout(pad=2)
    _save(fig, "08_shap_dependence")


# ── 6. SHAP Interaction heatmap ───────────────────────────────────────────────

def plot_shap_interaction_heatmap(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame,
    top_n: int = 10,
) -> None:
    import seaborn as sns

    feat_names = list(X_sample.columns)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer(X_sample).values

    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:top_n]
    top_shap  = shap_vals[:, top_idx]
    top_labels = [FEATURE_DISPLAY.get(feat_names[i], feat_names[i])
                  for i in top_idx]

    inter_matrix = np.corrcoef(top_shap.T)

    fig, ax = plt.subplots(figsize=(13, 10), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    sns.heatmap(
        inter_matrix, ax=ax,
        xticklabels=top_labels, yticklabels=top_labels,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        linewidths=0.3, linecolor="#2E3347",
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(
        "SHAP Feature Interaction Heatmap\n"
        "(Correlation between SHAP vectors — positive = features fire together)",
        color=TEXT_COLOR, fontsize=13, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45, colors=TEXT_COLOR)
    ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)
    fig.tight_layout()
    _save(fig, "09_shap_interaction_heatmap")


# ── 7. Gender-stratified SHAP ─────────────────────────────────────────────────

def plot_shap_by_gender(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    df_full: pd.DataFrame,
    top_n: int = 8,
) -> None:
    """
    Compare mean |SHAP value| per feature between men and women.
    Reveals whether the model uses different signals for each gender.
    """
    import seaborn as sns

    if "gender" not in df_full.columns:
        print("  [skip] gender column not in df")
        return

    explainer = shap.TreeExplainer(model)
    records = []

    for gender_id, gender_name in GENDER_LABELS.items():
        mask = (df_full["gender"] == gender_id).values[:len(X)]
        X_g  = X[mask].sample(min(1000, mask.sum()), random_state=42)
        sv   = explainer(X_g).values
        mean_abs = np.abs(sv).mean(axis=0)
        for i, col in enumerate(X.columns):
            records.append({
                "Gender":    gender_name,
                "Feature":   FEATURE_DISPLAY.get(col, col),
                "MeanAbsSHAP": mean_abs[i],
            })

    rdf = pd.DataFrame(records)

    # Top N features globally
    global_top = (
        rdf.groupby("Feature")["MeanAbsSHAP"].mean()
        .nlargest(top_n).index.tolist()
    )
    rdf_top = rdf[rdf["Feature"].isin(global_top)]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    sns.barplot(
        data=rdf_top, y="Feature", x="MeanAbsSHAP", hue="Gender",
        palette={"Women": ACCENT_2, "Men": ACCENT_1},
        ax=ax, edgecolor="#2E3347", linewidth=0.8,
    )
    ax.set_title(
        "Gender-Stratified SHAP Importance  |  Top Features\n"
        "Do men and women respond to different attraction signals?",
        color=TEXT_COLOR, fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Mean |SHAP value|", color=TEXT_COLOR)
    ax.set_ylabel("", color=TEXT_COLOR)
    ax.legend(facecolor=PANEL_COLOR, edgecolor="#2E3347",
              labelcolor=TEXT_COLOR)

    _save(fig, "10_shap_by_gender")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_modeling() -> xgb.XGBClassifier:
    """Full modeling pipeline — train, evaluate, explain."""
    df = pd.read_parquet(MODEL_PARQUET)
    df_full = pd.read_parquet(MODEL_PARQUET)   # keep gender col for stratification

    X, y, groups = prepare_model_data(df)

    print("\n→ Training XGBoost with GroupKFold …")
    model, cv_metrics, oof_probs = train_with_group_kfold(X, y, groups)

    # Save metrics
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([cv_metrics]).to_csv(
        OUT_REPORTS / "cv_metrics.csv", index=False
    )

    print("\n→ Model Diagnostics …")
    plot_model_diagnostics(y, oof_probs, cv_metrics)

    print("\n→ SHAP Global Importance …")
    shap_vals, X_sample = plot_shap_global(model, X)

    print("→ SHAP Dependence Plots …")
    plot_shap_dependence(shap_vals, X_sample)

    print("→ SHAP Interaction Heatmap …")
    plot_shap_interaction_heatmap(model, X_sample)

    print("→ SHAP by Gender …")
    plot_shap_by_gender(model, X, df_full)

    print("\n[modeling] All plots saved.")
    return model


if __name__ == "__main__":
    run_modeling()
