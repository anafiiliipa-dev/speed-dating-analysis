"""
02_eda_statistics.py — Exploratory Data Analysis & Statistical Inference

Analyses:
  1. Stated vs Revealed preference heatmap (by gender)
  2. Match-rate breakdown by attribute ratings, gender, race
  3. T-tests: stated vs revealed preference weights (clustered SE by iid)
  4. Chi-square: match rate by same-race / condition
  5. ANOVA: attribute ratings by field of study
  6. Preference–behaviour gap distribution plots
  7. Correlation matrix: stated prefs vs event-night ratings
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CLEAN_PARQUET, OUT_FIGS, OUT_REPORTS,
    ATTR_DIMS, ATTR_LABELS, GENDER_LABELS,
    STATED_PREF_COLS, RATINGS_GIVEN,
    SEABORN_STYLE, BG_COLOR, PANEL_COLOR, TEXT_COLOR,
    ACCENT_1, ACCENT_2, ACCENT_3, ACCENT_4, PALETTE_DIV, PALETTE_SEQ,
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


def clustered_se(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    cluster_col: str = "iid",
) -> dict:
    """
    OLS with clustered standard errors (by individual).

    Why clustered SE?
      Each individual appears in multiple rows (one per speed date).
      Observations within the same individual are correlated, so naive SE
      underestimates uncertainty. Clustering corrects for within-group
      correlation without assuming independence.
    """
    from numpy.linalg import lstsq, inv

    valid = df[[y_col, x_col, cluster_col]].dropna()
    y = valid[y_col].values
    X = np.column_stack([np.ones(len(y)), valid[x_col].values])
    clusters = valid[cluster_col].values

    # OLS estimates
    beta, _, _, _ = lstsq(X, y, rcond=None)
    residuals = y - X @ beta

    # Sandwich estimator: V = (X'X)⁻¹ · B · (X'X)⁻¹
    XtX_inv = inv(X.T @ X)
    B = np.zeros((2, 2))
    for cid in np.unique(clusters):
        mask = clusters == cid
        Xc   = X[mask]
        ec   = residuals[mask]
        B   += Xc.T @ np.outer(ec, ec) @ Xc

    V_cluster = XtX_inv @ B @ XtX_inv
    se_cluster = np.sqrt(np.diag(V_cluster))

    n, k = len(y), 2
    n_clusters = len(np.unique(clusters))
    # Small-sample correction G/(G-1) · (N-1)/(N-k)
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    V_cluster  *= correction
    se_cluster  = np.sqrt(np.diag(V_cluster))

    t_stat = beta[1] / se_cluster[1]
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_clusters - 1))

    return {
        "beta": beta[1],
        "se":   se_cluster[1],
        "t":    t_stat,
        "p":    p_val,
        "n":    n,
        "n_clusters": n_clusters,
    }


# ── 1. Stated vs Revealed preference heatmap ─────────────────────────────────

def plot_stated_vs_revealed_heatmap(df: pd.DataFrame) -> None:
    """
    Compare what men/women SAY matters (attr1_1…) vs what actually
    correlates with saying YES (point-biserial correlation of attr…dec).

    Left half  = stated importance (mean % allocation, normalised to 0-1)
    Right half = revealed preference (correlation with dec, absolute)
    """
    results = []
    for gender_id, gender_name in GENDER_LABELS.items():
        sub = df[df["gender"] == gender_id]
        for dim in ATTR_DIMS:
            stated_col  = f"{dim}1_1"
            rating_col  = dim
            if stated_col not in sub.columns or rating_col not in sub.columns:
                continue

            # Stated: mean importance (% allocation → 0-1)
            stated_mean = sub[stated_col].mean() / 100

            # Revealed: Pearson correlation between rating and decision
            valid = sub[[rating_col, "dec"]].dropna()
            corr, _ = stats.pearsonr(valid[rating_col], valid["dec"])

            results.append({
                "Gender":   gender_name,
                "Attribute": ATTR_LABELS.get(dim, dim),
                "Stated":   stated_mean,
                "Revealed": corr,
            })

    res_df = pd.DataFrame(results)

    # Pivot for heatmap
    stated_pivot   = res_df.pivot(index="Attribute", columns="Gender", values="Stated")
    revealed_pivot = res_df.pivot(index="Attribute", columns="Gender", values="Revealed")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
    fig.suptitle(
        "Stated vs Revealed Preferences  |  Women & Men",
        fontsize=16, color=TEXT_COLOR, fontweight="bold", y=1.02,
    )

    kw = dict(annot=True, fmt=".2f", linewidths=0.5,
              linecolor="#2E3347", cbar_kws={"shrink": 0.8})

    sns.heatmap(stated_pivot, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=0.35,
                **kw)
    axes[0].set_title("Stated Importance\n(mean % allocation ÷ 100)",
                      color=TEXT_COLOR, fontsize=12)

    sns.heatmap(revealed_pivot, ax=axes[1], cmap=PALETTE_DIV, vmin=-0.1, vmax=0.6,
                **kw)
    axes[1].set_title("Revealed Preference\n(Pearson r with dec=1)",
                      color=TEXT_COLOR, fontsize=12)

    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.set_xlabel("", color=TEXT_COLOR)
        ax.set_ylabel("", color=TEXT_COLOR)

    _save(fig, "01_stated_vs_revealed_heatmap")


# ── 2. Match rate & ratings distribution ─────────────────────────────────────

def plot_match_rate_analysis(df: pd.DataFrame) -> None:
    """Bar charts: match rate & YES rate by gender, wave, same-race."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG_COLOR)

    # (a) Decision rate by gender
    dec_by_gender = df.groupby("gender")["dec"].mean().reset_index()
    dec_by_gender["label"] = dec_by_gender["gender"].map(GENDER_LABELS)
    colors = [ACCENT_2, ACCENT_1]
    bars = axes[0].bar(dec_by_gender["label"], dec_by_gender["dec"],
                       color=colors, edgecolor="#2E3347", linewidth=1.2, width=0.5)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{bar.get_height():.1%}",
                     ha="center", va="bottom", color=TEXT_COLOR, fontsize=11)
    axes[0].set_title("Decision Rate (YES) by Gender", color=TEXT_COLOR)
    axes[0].set_ylabel("Proportion saying YES", color=TEXT_COLOR)
    axes[0].set_facecolor(PANEL_COLOR)

    # (b) Match rate by same-race
    match_race = df.groupby("samerace")["match"].mean().reset_index()
    match_race["label"] = match_race["samerace"].map({0: "Different Race", 1: "Same Race"})
    c2 = [ACCENT_4, ACCENT_3]
    bars2 = axes[1].bar(match_race["label"], match_race["match"],
                        color=c2, edgecolor="#2E3347", linewidth=1.2, width=0.5)
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.003,
                     f"{bar.get_height():.1%}",
                     ha="center", va="bottom", color=TEXT_COLOR, fontsize=11)
    axes[1].set_title("Match Rate by Race Pairing", color=TEXT_COLOR)
    axes[1].set_ylabel("Match Rate", color=TEXT_COLOR)
    axes[1].set_facecolor(PANEL_COLOR)

    # (c) Distribution of attractiveness ratings by outcome
    for outcome, col, label in [(0, ACCENT_4, "No Decision"), (1, ACCENT_3, "Yes Decision")]:
        sub = df[df["dec"] == outcome]["attr"].dropna()
        axes[2].hist(sub, bins=10, alpha=0.65, color=col, label=label,
                     edgecolor="#2E3347", range=(1, 10))
    axes[2].set_title("Attractiveness Rating Distribution by Decision", color=TEXT_COLOR)
    axes[2].set_xlabel("Attractiveness Rating (1-10)", color=TEXT_COLOR)
    axes[2].set_ylabel("Count", color=TEXT_COLOR)
    axes[2].legend(facecolor=PANEL_COLOR, edgecolor="#2E3347", labelcolor=TEXT_COLOR)
    axes[2].set_facecolor(PANEL_COLOR)

    fig.tight_layout(pad=2)
    _save(fig, "02_match_rate_analysis")


# ── 3. Preference–behaviour gap violin plots ──────────────────────────────────

def plot_pref_behaviour_gap(df: pd.DataFrame) -> None:
    """
    Violin plots of (gap = actual rating − stated importance/10) per attribute.
    A positive gap means the person ACTUALLY weighs that trait more than stated.
    """
    gap_cols = [f"{d}_gap" for d in ATTR_DIMS if f"{d}_gap" in df.columns]
    if not gap_cols:
        print("  [skip] gap columns not found")
        return

    gap_df = df[gap_cols + ["gender"]].copy()
    gap_df = gap_df.melt(id_vars="gender", value_vars=gap_cols,
                          var_name="dimension", value_name="gap")
    gap_df["dimension"] = gap_df["dimension"].str.replace("_gap", "")
    gap_df["dimension"] = gap_df["dimension"].map(ATTR_LABELS)
    gap_df["Gender"] = gap_df["gender"].map(GENDER_LABELS)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    sns.violinplot(
        data=gap_df, x="dimension", y="gap", hue="Gender",
        split=True, inner="quart", palette={"Women": ACCENT_2, "Men": ACCENT_1},
        ax=ax, linewidth=0.8,
    )
    ax.axhline(0, color=TEXT_COLOR, linestyle="--", linewidth=1, alpha=0.6,
               label="No gap (stated = revealed)")
    ax.set_title(
        "Preference–Behaviour Gap  |  Actual Weight − Stated Importance\n"
        "(positive = trait matters MORE in practice than stated)",
        color=TEXT_COLOR, fontsize=13,
    )
    ax.set_xlabel("Attribute", color=TEXT_COLOR)
    ax.set_ylabel("Gap Score", color=TEXT_COLOR)
    ax.legend(facecolor=PANEL_COLOR, edgecolor="#2E3347", labelcolor=TEXT_COLOR)

    _save(fig, "03_pref_behaviour_gap")


# ── 4. Correlation matrix ─────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Correlation matrix between stated preferences (CLR) and revealed ratings.
    Diagonal should be high if people's stated prefs predict their behaviour.
    """
    from config import STATED_PREF_CLR as CLR_COLS

    clr_present    = [c for c in CLR_COLS if c in df.columns]
    rating_present = [c for c in RATINGS_GIVEN if c in df.columns]

    if not clr_present or not rating_present:
        print("  [skip] CLR or rating columns missing")
        return

    combined = clr_present + rating_present
    corr = df[combined].corr()

    # Rename for display
    rename = {c: c.replace("1_1_clr", " (stated)").replace("_", " ").title()
              for c in clr_present}
    rename.update({c: ATTR_LABELS.get(c, c) + " (revealed)"
                   for c in rating_present})
    corr = corr.rename(index=rename, columns=rename)

    fig, ax = plt.subplots(figsize=(14, 11), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True   # show lower triangle

    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap=PALETTE_DIV, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", linewidths=0.4, linecolor="#2E3347",
        square=True, cbar_kws={"shrink": 0.7},
    )
    ax.set_title(
        "Stated Preferences (CLR) ↔ Revealed Ratings\nCorrelation Matrix",
        color=TEXT_COLOR, fontsize=13,
    )
    ax.tick_params(axis="x", rotation=45, colors=TEXT_COLOR)
    ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)

    _save(fig, "04_correlation_matrix")


# ── 5. Statistical tests ──────────────────────────────────────────────────────

def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run and tabulate:
      (a) T-test: stated vs revealed attractiveness weight (by gender)
      (b) Chi-square: match rate × same-race
      (c) ANOVA: attractiveness rating across top 5 field_cd groups
      (d) Clustered-SE regression: attr rating → dec (by gender)

    Returns a summary DataFrame saved as CSV.
    """
    rows = []

    # (a) T-tests: stated importance vs revelation  ───────────────────────────
    for gender_id, gender_name in GENDER_LABELS.items():
        sub = df[df["gender"] == gender_id]
        for dim in ATTR_DIMS:
            sc = f"{dim}1_1"
            rc = dim
            if sc not in sub.columns or rc not in sub.columns:
                continue
            stated   = sub[sc].dropna() / 10      # rescale to 0-10
            revealed = sub[rc].dropna()
            t, p = ttest_ind(stated, revealed, equal_var=False)
            rows.append({
                "Test":      "Welch T-test",
                "Gender":    gender_name,
                "Dimension": ATTR_LABELS.get(dim, dim),
                "Stat":      round(t, 3),
                "p-value":   round(p, 4),
                "Significant": "✓" if p < 0.05 else "",
                "Note":      "Stated (rescaled) vs Revealed rating",
            })

    # (b) Chi-square: match × same-race  ──────────────────────────────────────
    ct = pd.crosstab(df["match"], df["samerace"])
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    rows.append({
        "Test": "Chi-Square",
        "Gender": "Both",
        "Dimension": "Match × Same-Race",
        "Stat": round(chi2, 3),
        "p-value": round(p_chi, 4),
        "Significant": "✓" if p_chi < 0.05 else "",
        "Note": f"df={dof}",
    })

    # (c) ANOVA: attractiveness rating by field_cd  ───────────────────────────
    top_fields = df["field_cd"].value_counts().nlargest(5).index.tolist()
    groups = [
        df.loc[df["field_cd"] == fid, "attr"].dropna().values
        for fid in top_fields
    ]
    if all(len(g) > 1 for g in groups):
        F, p_anova = f_oneway(*groups)
        rows.append({
            "Test": "One-way ANOVA",
            "Gender": "Both",
            "Dimension": "Attr Rating by Field of Study",
            "Stat": round(F, 3),
            "p-value": round(p_anova, 4),
            "Significant": "✓" if p_anova < 0.05 else "",
            "Note": f"Top-{len(top_fields)} fields",
        })

    # (d) Clustered-SE OLS: attr → dec by gender  ─────────────────────────────
    for gender_id, gender_name in GENDER_LABELS.items():
        sub = df[df["gender"] == gender_id]
        res = clustered_se(sub, y_col="dec", x_col="attr", cluster_col="iid")
        rows.append({
            "Test":      "Clustered-SE OLS",
            "Gender":    gender_name,
            "Dimension": "attr → dec",
            "Stat":      round(res["t"], 3),
            "p-value":   round(res["p"], 4),
            "Significant": "✓" if res["p"] < 0.05 else "",
            "Note":      f"β={res['beta']:.3f}, SE={res['se']:.3f}, N_clusters={res['n_clusters']}",
        })

    summary = pd.DataFrame(rows)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = OUT_REPORTS / "statistical_tests.csv"
    summary.to_csv(out_path, index=False)
    print(f"  [csv] statistical tests → {out_path.name}")
    return summary


# ── 6. Gender gap heatmap ─────────────────────────────────────────────────────

def plot_gender_preference_gap(df: pd.DataFrame) -> None:
    """
    Side-by-side bar chart comparing stated vs revealed importance for
    each attribute, split by gender. This is the 'money plot' that
    visualises the stated-revealed discrepancy hypothesis.
    """
    records = []
    for gender_id, gender_name in GENDER_LABELS.items():
        sub = df[df["gender"] == gender_id]
        for dim in ATTR_DIMS:
            sc = f"{dim}1_1"
            rc = dim
            if sc not in sub.columns or rc not in sub.columns:
                continue
            stated_w   = sub[sc].mean() / 10         # rescale to 0-10 space
            revealed_w = sub[[rc, "dec"]].dropna()
            # Use logistic coefficient proxy: mean rating when dec=1 vs dec=0
            mu1 = revealed_w[revealed_w["dec"] == 1][rc].mean()
            mu0 = revealed_w[revealed_w["dec"] == 0][rc].mean()
            effect = mu1 - mu0   # "lift" in rating for yes vs no decision
            records.append({
                "Gender": gender_name,
                "Attribute": ATTR_LABELS.get(dim, dim),
                "Stated (0-10)":    stated_w,
                "Revealed lift":    effect,
            })

    rdf = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_COLOR,
                             sharey=True)

    for ax, gender_name, color in zip(axes,
                                       GENDER_LABELS.values(),
                                       [ACCENT_2, ACCENT_1]):
        sub = rdf[rdf["Gender"] == gender_name]
        x = np.arange(len(sub))
        width = 0.35

        b1 = ax.bar(x - width/2, sub["Stated (0-10)"],   width,
                    label="Stated",   color=color, alpha=0.75,
                    edgecolor="#2E3347")
        b2 = ax.bar(x + width/2, sub["Revealed lift"],   width,
                    label="Revealed lift", color=ACCENT_3, alpha=0.85,
                    edgecolor="#2E3347")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["Attribute"], rotation=25, ha="right",
                           color=TEXT_COLOR)
        ax.set_facecolor(PANEL_COLOR)
        ax.set_title(f"{gender_name} — Stated vs Revealed",
                     color=TEXT_COLOR, fontsize=13, fontweight="bold")
        ax.legend(facecolor=PANEL_COLOR, edgecolor="#2E3347",
                  labelcolor=TEXT_COLOR)
        ax.set_ylabel("Score / Lift", color=TEXT_COLOR)

    fig.suptitle(
        "Stated Importance vs Revealed Lift (YES vs NO rating difference)",
        fontsize=14, color=TEXT_COLOR, fontweight="bold",
    )
    fig.tight_layout(pad=2)
    _save(fig, "05_gender_preference_gap")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_eda_statistics() -> pd.DataFrame:
    """Load clean data and run all EDA / stats steps."""
    df = pd.read_parquet(CLEAN_PARQUET)
    print(f"[EDA] loaded {df.shape} from clean parquet")

    print("\n→ Stated vs Revealed Heatmap …")
    plot_stated_vs_revealed_heatmap(df)

    print("→ Match Rate Analysis …")
    plot_match_rate_analysis(df)

    print("→ Preference–Behaviour Gap …")
    plot_pref_behaviour_gap(df)

    print("→ Correlation Matrix …")
    plot_correlation_matrix(df)

    print("→ Statistical Tests …")
    stats_df = run_statistical_tests(df)

    print("→ Gender Preference Gap …")
    plot_gender_preference_gap(df)

    print("\n[EDA] All figures saved.")
    return stats_df


if __name__ == "__main__":
    run_eda_statistics()
