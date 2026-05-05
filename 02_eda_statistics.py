"""
02_eda_statistics.py — Stage 2: Exploratory Data Analysis & Statistical Inference.

Pipeline (sequential, each step is a pure function):

    load_clean  →  validate_eda_columns
                →  plot_stated_vs_revealed_heatmap
                →  plot_match_rate_analysis
                →  plot_pref_behaviour_gap
                →  plot_correlation_matrix
                →  run_statistical_tests
                →  plot_gender_preference_gap

Key analyses:

  • Stated vs revealed preference heatmap (by gender)
  • Match-rate breakdown (gender, same-race, attractiveness)
  • Welch t-tests: stated vs revealed weights
  • Chi-square: match × same-race
  • One-way ANOVA: attractiveness rating across top fields of study
  • Clustered-SE OLS: attr → dec (clusters by individual `iid`)
  • Preference–behaviour gap distributions

Engineering improvements over v1:

  • Loguru structured logging (DEBUG/INFO/SUCCESS/WARNING).
  • Pydantic schema validation via existing `CleanDatasetContract`.
  • Custom exception hierarchy (`EDAStatisticsError`).
  • `EDAResult` and `ClusteredSEResult` dataclasses — typed return contracts.
  • Plot functions return `Path | None` — None means skipped (missing cols).
  • Targeted warning filter (FutureWarning only) — no global silence.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import inv, lstsq
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind

from config import (
    ACCENT_1, ACCENT_2, ACCENT_3, ACCENT_4,
    ATTR_DIMS, ATTR_LABELS, BG_COLOR, CLEAN_PARQUET,
    GENDER_LABELS, OUT_FIGS, OUT_REPORTS, PALETTE_DIV,
    PANEL_COLOR, RATINGS_GIVEN, SEABORN_STYLE, STATED_PREF_CLR,
    TEXT_COLOR,
)
from logging_config import get_logger
from schemas import CleanDatasetContract, SchemaValidationError

warnings.filterwarnings("ignore", category=FutureWarning)
log = get_logger(__name__)

plt.rcParams.update(SEABORN_STYLE)
plt.rcParams["font.family"] = "DejaVu Sans"


# ──────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────────────────────────────────────
class EDAStatisticsError(RuntimeError):
    """Base class for errors raised in the EDA & statistics stage."""


class MissingColumnsError(EDAStatisticsError):
    """Raised when required columns are missing from the input dataframe."""


class StatisticalTestError(EDAStatisticsError):
    """Raised when a statistical test cannot be computed."""


# ──────────────────────────────────────────────────────────────────────────────
# Result contracts
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ClusteredSEResult:
    """Output of OLS regression with cluster-robust standard errors."""
    beta: float
    se: float
    t: float
    p: float
    n: int
    n_clusters: int


@dataclass(frozen=True)
class EDAResult:
    """End-to-end output of Stage 2."""
    stats_df: pd.DataFrame
    stats_csv: Path
    figures: list[Path]
    n_rows: int
    n_tests: int
    n_significant: int


# Required columns for the EDA stage (subset of the clean schema actually used)
_REQUIRED_EDA_COLUMNS: tuple[str, ...] = (
    "iid", "gender", "dec", "match", "attr", "samerace", "field_cd",
)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────
def _validate_eda_columns(df: pd.DataFrame) -> None:
    """Fail fast if the dataframe is missing the columns Stage 2 depends on."""
    missing = [c for c in _REQUIRED_EDA_COLUMNS if c not in df.columns]
    if missing:
        raise MissingColumnsError(
            f"EDA-required columns not found: {missing}. "
            f"Re-run Stage 1 (01_data_engineering.py) before Stage 2."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str) -> Path:
    """Save a matplotlib figure to OUT_FIGS and return its path."""
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    path = OUT_FIGS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    log.success("Figure saved → {name}", name=path.name)
    return path


def clustered_se(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    cluster_col: str = "iid",
) -> ClusteredSEResult:
    """
    OLS with cluster-robust standard errors (sandwich estimator).

    Why clustered SE?
      Each individual appears in multiple rows (one per speed date).
      Observations within the same individual are correlated, so naive SE
      underestimates uncertainty. Clustering corrects for within-group
      correlation without assuming independence.

        V_cluster = (X'X)^-1 . B . (X'X)^-1
        with small-sample correction G/(G-1) . (N-1)/(N-k).
    """
    valid = df[[y_col, x_col, cluster_col]].dropna()
    if valid.empty:
        raise StatisticalTestError(
            f"No valid rows for clustered SE regression "
            f"({y_col} ~ {x_col}, cluster={cluster_col})."
        )

    y = valid[y_col].values
    X = np.column_stack([np.ones(len(y)), valid[x_col].values])
    clusters = valid[cluster_col].values

    beta, _, _, _ = lstsq(X, y, rcond=None)
    residuals = y - X @ beta

    XtX_inv = inv(X.T @ X)
    B = np.zeros((2, 2))
    for cid in np.unique(clusters):
        mask = clusters == cid
        Xc = X[mask]
        ec = residuals[mask]
        B += Xc.T @ np.outer(ec, ec) @ Xc

    n, k = len(y), 2
    n_clusters = int(len(np.unique(clusters)))
    if n_clusters < 2:
        raise StatisticalTestError(
            f"Need at least 2 clusters for clustered SE (got {n_clusters})."
        )

    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    V_cluster = (XtX_inv @ B @ XtX_inv) * correction
    se_cluster = np.sqrt(np.diag(V_cluster))

    t_stat = float(beta[1] / se_cluster[1])
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n_clusters - 1)))

    return ClusteredSEResult(
        beta=float(beta[1]),
        se=float(se_cluster[1]),
        t=t_stat,
        p=p_val,
        n=int(n),
        n_clusters=n_clusters,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Stated vs Revealed preference heatmap
# ──────────────────────────────────────────────────────────────────────────────
def plot_stated_vs_revealed_heatmap(df: pd.DataFrame) -> Path:
    """
    Compare what men/women SAY matters (attr1_1...) vs what actually
    correlates with saying YES (Pearson r of attr...dec).
    """
    log.info("Building stated-vs-revealed heatmap ...")
    results = []
    for gender_id, gender_name in GENDER_LABELS.items():
        sub = df[df["gender"] == gender_id]
        for dim in ATTR_DIMS:
            stated_col = f"{dim}1_1"
            rating_col = dim
            if stated_col not in sub.columns or rating_col not in sub.columns:
                continue

            stated_mean = sub[stated_col].mean() / 100
            valid = sub[[rating_col, "dec"]].dropna()
            corr, _ = stats.pearsonr(valid[rating_col], valid["dec"])

            results.append({
                "Gender": gender_name,
                "Attribute": ATTR_LABELS.get(dim, dim),
                "Stated": stated_mean,
                "Revealed": corr,
            })

    res_df = pd.DataFrame(results)
    stated_pivot = res_df.pivot(index="Attribute", columns="Gender", values="Stated")
    revealed_pivot = res_df.pivot(index="Attribute", columns="Gender", values="Revealed")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
    fig.suptitle(
        "Stated vs Revealed Preferences  |  Women & Men",
        fontsize=16, color=TEXT_COLOR, fontweight="bold", y=1.02,
    )

    kw = dict(annot=True, fmt=".2f", linewidths=0.5,
              linecolor="#2E3347", cbar_kws={"shrink": 0.8})

    sns.heatmap(stated_pivot, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=0.35, **kw)
    axes[0].set_title("Stated Importance\n(mean % allocation / 100)",
                      color=TEXT_COLOR, fontsize=12)

    sns.heatmap(revealed_pivot, ax=axes[1], cmap=PALETTE_DIV, vmin=-0.1, vmax=0.6, **kw)
    axes[1].set_title("Revealed Preference\n(Pearson r with dec=1)",
                      color=TEXT_COLOR, fontsize=12)

    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.set_xlabel("", color=TEXT_COLOR)
        ax.set_ylabel("", color=TEXT_COLOR)

    return _save(fig, "01_stated_vs_revealed_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Match rate & ratings distribution
# ──────────────────────────────────────────────────────────────────────────────
def plot_match_rate_analysis(df: pd.DataFrame) -> Path:
    """Bar charts: decision rate by gender, match rate by same-race, attr distribution."""
    log.info("Building match-rate analysis ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG_COLOR)

    dec_by_gender = df.groupby("gender")["dec"].mean().reset_index()
    dec_by_gender["label"] = dec_by_gender["gender"].map(GENDER_LABELS)
    bars = axes[0].bar(dec_by_gender["label"], dec_by_gender["dec"],
                       color=[ACCENT_2, ACCENT_1], edgecolor="#2E3347",
                       linewidth=1.2, width=0.5)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{bar.get_height():.1%}",
                     ha="center", va="bottom", color=TEXT_COLOR, fontsize=11)
    axes[0].set_title("Decision Rate (YES) by Gender", color=TEXT_COLOR)
    axes[0].set_ylabel("Proportion saying YES", color=TEXT_COLOR)
    axes[0].set_facecolor(PANEL_COLOR)

    match_race = df.groupby("samerace")["match"].mean().reset_index()
    match_race["label"] = match_race["samerace"].map({0: "Different Race", 1: "Same Race"})
    bars2 = axes[1].bar(match_race["label"], match_race["match"],
                        color=[ACCENT_4, ACCENT_3], edgecolor="#2E3347",
                        linewidth=1.2, width=0.5)
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.003,
                     f"{bar.get_height():.1%}",
                     ha="center", va="bottom", color=TEXT_COLOR, fontsize=11)
    axes[1].set_title("Match Rate by Race Pairing", color=TEXT_COLOR)
    axes[1].set_ylabel("Match Rate", color=TEXT_COLOR)
    axes[1].set_facecolor(PANEL_COLOR)

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
    return _save(fig, "02_match_rate_analysis")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Preference-behaviour gap violin plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_pref_behaviour_gap(df: pd.DataFrame) -> Path | None:
    """
    Violin plots of (gap = actual rating - stated importance/10) per attribute.
    Returns None (skipped) if no `*_gap` columns are present.
    """
    log.info("Building preference-behaviour gap plot ...")
    gap_cols = [f"{d}_gap" for d in ATTR_DIMS if f"{d}_gap" in df.columns]
    if not gap_cols:
        log.warning("Gap columns not found - skipping pref-behaviour gap plot")
        return None

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
        split=True, inner="quart",
        palette={"Women": ACCENT_2, "Men": ACCENT_1},
        ax=ax, linewidth=0.8,
    )
    ax.axhline(0, color=TEXT_COLOR, linestyle="--", linewidth=1, alpha=0.6,
               label="No gap (stated = revealed)")
    ax.set_title(
        "Preference-Behaviour Gap  |  Actual Weight - Stated Importance\n"
        "(positive = trait matters MORE in practice than stated)",
        color=TEXT_COLOR, fontsize=13,
    )
    ax.set_xlabel("Attribute", color=TEXT_COLOR)
    ax.set_ylabel("Gap Score", color=TEXT_COLOR)
    ax.legend(facecolor=PANEL_COLOR, edgecolor="#2E3347", labelcolor=TEXT_COLOR)

    return _save(fig, "03_pref_behaviour_gap")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Correlation matrix
# ──────────────────────────────────────────────────────────────────────────────
def plot_correlation_matrix(df: pd.DataFrame) -> Path | None:
    """Correlation matrix between stated preferences (CLR) and revealed ratings."""
    log.info("Building correlation matrix ...")
    clr_present = [c for c in STATED_PREF_CLR if c in df.columns]
    rating_present = [c for c in RATINGS_GIVEN if c in df.columns]

    if not clr_present or not rating_present:
        log.warning("CLR or rating columns missing - skipping correlation matrix")
        return None

    combined = clr_present + rating_present
    corr = df[combined].corr()

    rename = {c: c.replace("1_1_clr", " (stated)").replace("_", " ").title()
              for c in clr_present}
    rename.update({c: ATTR_LABELS.get(c, c) + " (revealed)"
                   for c in rating_present})
    corr = corr.rename(index=rename, columns=rename)

    fig, ax = plt.subplots(figsize=(14, 11), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap=PALETTE_DIV, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", linewidths=0.4, linecolor="#2E3347",
        square=True, cbar_kws={"shrink": 0.7},
    )
    ax.set_title(
        "Stated Preferences (CLR) vs Revealed Ratings\nCorrelation Matrix",
        color=TEXT_COLOR, fontsize=13,
    )
    ax.tick_params(axis="x", rotation=45, colors=TEXT_COLOR)
    ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)

    return _save(fig, "04_correlation_matrix")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Statistical tests
# ──────────────────────────────────────────────────────────────────────────────
def run_statistical_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    """
    Run and tabulate:
      (a) Welch t-test: stated vs revealed attractiveness weight (by gender)
      (b) Chi-square: match rate x same-race
      (c) ANOVA: attractiveness rating across top-5 field_cd groups
      (d) Clustered-SE OLS: attr -> dec (by gender)
    """
    log.info("Running statistical tests ...")
    rows: list[dict] = []

    for gender_id, gender_name in GENDER_LABELS.items():
        sub = df[df["gender"] == gender_id]
        for dim in ATTR_DIMS:
            sc, rc = f"{dim}1_1", dim
            if sc not in sub.columns or rc not in sub.columns:
                continue
            stated = sub[sc].dropna() / 10
            revealed = sub[rc].dropna()
            t, p = ttest_ind(stated, revealed, equal_var=False)
            rows.append({
                "Test": "Welch T-test",
                "Gender": gender_name,
                "Dimension": ATTR_LABELS.get(dim, dim),
                "Stat": round(float(t), 3),
                "p-value": round(float(p), 4),
                "Significant": "yes" if p < 0.05 else "",
                "Note": "Stated (rescaled) vs Revealed rating",
            })

    ct = pd.crosstab(df["match"], df["samerace"])
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    rows.append({
        "Test": "Chi-Square",
        "Gender": "Both",
        "Dimension": "Match x Same-Race",
        "Stat": round(float(chi2), 3),
        "p-value": round(float(p_chi), 4),
        "Significant": "yes" if p_chi < 0.05 else "",
        "Note": f"df={dof}",
    })

    top_fields = df["field_cd"].value_counts().nlargest(5).index.tolist()
    groups = [df.loc[df["field_cd"] == fid, "attr"].dropna().values
              for fid in top_fields]
    if all(len(g) > 1 for g in groups):
        F, p_anova = f_oneway(*groups)
        rows.append({
            "Test": "One-way ANOVA",
            "Gender": "Both",
            "Dimension": "Attr Rating by Field of Study",
            "Stat": round(float(F), 3),
            "p-value": round(float(p_anova), 4),
            "Significant": "yes" if p_anova < 0.05 else "",
            "Note": f"Top-{len(top_fields)} fields",
        })
    else:
        log.warning("ANOVA skipped - at least one field group has <2 obs")

    for gender_id, gender_name in GENDER_LABELS.items():
        sub = df[df["gender"] == gender_id]
        try:
            res = clustered_se(sub, y_col="dec", x_col="attr", cluster_col="iid")
        except StatisticalTestError as e:
            log.warning("Clustered-SE skipped for {g}: {err}",
                        g=gender_name, err=str(e))
            continue
        rows.append({
            "Test": "Clustered-SE OLS",
            "Gender": gender_name,
            "Dimension": "attr -> dec",
            "Stat": round(res.t, 3),
            "p-value": round(res.p, 4),
            "Significant": "yes" if res.p < 0.05 else "",
            "Note": f"beta={res.beta:.3f}, SE={res.se:.3f}, N_clusters={res.n_clusters}",
        })

    summary = pd.DataFrame(rows)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = OUT_REPORTS / "statistical_tests.csv"
    summary.to_csv(out_path, index=False)
    log.success("Statistical tests CSV -> {p}", p=out_path.name)
    return summary, out_path


# ──────────────────────────────────────────────────────────────────────────────
# 6. Gender preference gap
# ──────────────────────────────────────────────────────────────────────────────
def plot_gender_preference_gap(df: pd.DataFrame) -> Path:
    """
    Side-by-side bar chart comparing stated vs revealed importance for
    each attribute, split by gender. Visualises the stated-revealed
    discrepancy hypothesis (the 'money plot').
    """
    log.info("Building gender preference gap plot ...")
    records = []
    for gender_id, gender_name in GENDER_LABELS.items():
        sub = df[df["gender"] == gender_id]
        for dim in ATTR_DIMS:
            sc, rc = f"{dim}1_1", dim
            if sc not in sub.columns or rc not in sub.columns:
                continue
            stated_w = sub[sc].mean() / 10
            revealed_w = sub[[rc, "dec"]].dropna()
            mu1 = revealed_w[revealed_w["dec"] == 1][rc].mean()
            mu0 = revealed_w[revealed_w["dec"] == 0][rc].mean()
            effect = mu1 - mu0
            records.append({
                "Gender": gender_name,
                "Attribute": ATTR_LABELS.get(dim, dim),
                "Stated (0-10)": stated_w,
                "Revealed lift": effect,
            })

    rdf = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_COLOR, sharey=True)

    for ax, gender_name, color in zip(axes, GENDER_LABELS.values(),
                                      [ACCENT_2, ACCENT_1]):
        sub = rdf[rdf["Gender"] == gender_name]
        x = np.arange(len(sub))
        width = 0.35

        ax.bar(x - width / 2, sub["Stated (0-10)"], width,
               label="Stated", color=color, alpha=0.75, edgecolor="#2E3347")
        ax.bar(x + width / 2, sub["Revealed lift"], width,
               label="Revealed lift", color=ACCENT_3, alpha=0.85, edgecolor="#2E3347")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["Attribute"], rotation=25, ha="right",
                           color=TEXT_COLOR)
        ax.set_facecolor(PANEL_COLOR)
        ax.set_title(f"{gender_name} - Stated vs Revealed",
                     color=TEXT_COLOR, fontsize=13, fontweight="bold")
        ax.legend(facecolor=PANEL_COLOR, edgecolor="#2E3347", labelcolor=TEXT_COLOR)
        ax.set_ylabel("Score / Lift", color=TEXT_COLOR)

    fig.suptitle(
        "Stated Importance vs Revealed Lift (YES vs NO rating difference)",
        fontsize=14, color=TEXT_COLOR, fontweight="bold",
    )
    fig.tight_layout(pad=2)
    return _save(fig, "05_gender_preference_gap")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def run_eda_statistics() -> EDAResult:
    """Execute Stage 2 end-to-end. Returns a typed `EDAResult`."""
    log.info("─" * 60)
    log.info("STAGE 2 | EDA & Statistical Inference")
    log.info("─" * 60)

    try:
        if not CLEAN_PARQUET.exists():
            raise EDAStatisticsError(
                f"Clean parquet not found at {CLEAN_PARQUET}. "
                f"Run Stage 1 (01_data_engineering.py) first."
            )

        df = pd.read_parquet(CLEAN_PARQUET)
        log.success("Clean parquet loaded: {n_rows} rows x {n_cols} cols",
                    n_rows=df.shape[0], n_cols=df.shape[1])

        try:
            CleanDatasetContract.validate_dataframe(df)
        except SchemaValidationError as e:
            raise EDAStatisticsError(f"Clean schema invalid: {e}") from e
        log.info("Clean schema validated ✓")

        _validate_eda_columns(df)
        log.info("EDA-required columns present ✓")

        figures: list[Path] = []
        figures.append(plot_stated_vs_revealed_heatmap(df))
        figures.append(plot_match_rate_analysis(df))

        gap_fig = plot_pref_behaviour_gap(df)
        if gap_fig is not None:
            figures.append(gap_fig)

        corr_fig = plot_correlation_matrix(df)
        if corr_fig is not None:
            figures.append(corr_fig)

        stats_df, stats_csv = run_statistical_tests(df)

        figures.append(plot_gender_preference_gap(df))

        n_significant = int((stats_df["Significant"] == "yes").sum())
        result = EDAResult(
            stats_df=stats_df,
            stats_csv=stats_csv,
            figures=figures,
            n_rows=int(df.shape[0]),
            n_tests=int(len(stats_df)),
            n_significant=n_significant,
        )
        log.success(
            "Stage 2 complete ✓ | {n_figs} figures, {n_tests} tests "
            "({n_sig} significant)",
            n_figs=len(figures), n_tests=result.n_tests, n_sig=n_significant,
        )
        return result

    except EDAStatisticsError:
        raise
    except Exception as e:
        log.exception("Unexpected failure in EDA & statistics stage")
        raise EDAStatisticsError(f"Stage 2 failed: {e}") from e


if __name__ == "__main__":
    run_eda_statistics()
