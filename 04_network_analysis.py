"""
04_network_analysis.py — Bipartite Network Analysis

Builds a directed bipartite graph:
  - Left nodes  = Men   (gender = 1)
  - Right nodes = Women (gender = 0)
  - Edge (m → w) exists if man said YES to woman (dec = 1)
  - Edge (w → m) exists if woman said YES to man (dec = 1)

Metrics:
  - Hub score (in-degree / total dates): how many people wanted YOU
  - Selectivity (out-degree inverted):   how often YOU said YES
  - Reciprocity:                         mutual YES pairs / total YES pairs
  - HITS algorithm for authority / hub scores
  - Community structure via projection
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CLEAN_PARQUET, OUT_FIGS, OUT_REPORTS, OUT_DATA,
    GENDER_LABELS,
    SEABORN_STYLE, BG_COLOR, PANEL_COLOR, TEXT_COLOR,
    ACCENT_1, ACCENT_2, ACCENT_3, ACCENT_4, PALETTE_SEQ,
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


# ── 1. Build bipartite graph ──────────────────────────────────────────────────

def build_bipartite_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Directed bipartite graph: node = iid, edge = dec=1 directed from
    the person who said YES toward their date.

    Node attributes: gender, wave, iid.
    Edge attributes: like, attr (attractiveness rating).
    """
    G = nx.DiGraph()

    # Add all nodes with gender attribute
    iid_gender = df[["iid", "gender"]].drop_duplicates().set_index("iid")["gender"]
    for iid, gender in iid_gender.items():
        G.add_node(
            int(iid),
            gender=int(gender),
            bipartite=int(gender),
            label=f"{'M' if gender == 1 else 'W'}{int(iid)}",
        )

    # Add directed edges for YES decisions
    yes_edges = df[df["dec"] == 1][["iid", "pid", "attr", "like"]].dropna(subset=["iid", "pid"])
    for _, row in yes_edges.iterrows():
        src = int(row["iid"])
        dst = int(row["pid"])
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(
                src, dst,
                attr_rating=float(row["attr"]) if pd.notna(row["attr"]) else 5.0,
                like_rating=float(row["like"]) if pd.notna(row["like"]) else 5.0,
            )

    print(f"[graph] nodes={G.number_of_nodes()}  "
          f"edges={G.number_of_edges()}  "
          f"density={nx.density(G):.4f}")
    return G


# ── 2. Node metrics ───────────────────────────────────────────────────────────

def compute_node_metrics(
    G: nx.DiGraph,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-node metrics:
      - in_degree:    number of YES received
      - out_degree:   number of YES given
      - hub_score:    in_degree / total_dates (desirability)
      - selectivity:  1 − out_degree / total_dates (how picky)
      - reciprocity:  proportion of mutual matches
      - hits_auth:    HITS authority score (being chosen by hubs)
      - hits_hub:     HITS hub score (choosing authorities)
    """
    # Total dates per person
    total_dates = df.groupby("iid")["dec"].count().to_dict()

    # HITS algorithm
    hits_hub, hits_auth = nx.hits(G, max_iter=200, normalized=True)

    records = []
    for node in G.nodes():
        n_dates = total_dates.get(node, 1)
        in_d    = G.in_degree(node)
        out_d   = G.out_degree(node)

        # Reciprocity: symmetric YES edges / in_degree
        mutual = sum(
            1 for nbr in G.predecessors(node)
            if G.has_edge(node, nbr)
        )
        recip_rate = mutual / max(in_d, 1)

        records.append({
            "iid":         node,
            "gender":      G.nodes[node]["gender"],
            "gender_label": GENDER_LABELS[G.nodes[node]["gender"]],
            "in_degree":   in_d,
            "out_degree":  out_d,
            "total_dates": n_dates,
            "hub_score":   in_d / n_dates,          # desirability
            "selectivity": 1 - out_d / n_dates,     # pickiness
            "reciprocity": recip_rate,
            "hits_authority": hits_auth.get(node, 0),
            "hits_hub":       hits_hub.get(node, 0),
        })

    metrics_df = pd.DataFrame(records).sort_values("hub_score", ascending=False)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_REPORTS / "network_node_metrics.csv", index=False)
    print(f"  [csv] node metrics → network_node_metrics.csv")
    return metrics_df


# ── 3. Visualise network (wave-sampled subset) ────────────────────────────────

def plot_network_sample(
    G: nx.DiGraph,
    metrics_df: pd.DataFrame,
    wave: int = 11,
    df: pd.DataFrame = None,
) -> None:
    """
    Plot the bipartite YES-graph for a single wave.
    Node size  = hub_score (desirability)
    Node color = gender  (blue = men, pink = women)
    Edge alpha = attr_rating (more attractive → more opaque)
    """
    if df is None:
        return

    wave_iids = set(df[df["wave"] == wave]["iid"].dropna().astype(int))
    sub = G.subgraph([n for n in G.nodes() if n in wave_iids])

    if sub.number_of_nodes() < 3:
        print(f"  [skip] wave {wave} too small for plot")
        return

    pos = nx.bipartite_layout(
        sub,
        nodes=[n for n in sub.nodes() if G.nodes[n]["gender"] == 1],
    )

    hub_lookup = metrics_df.set_index("iid")["hub_score"].to_dict()
    node_sizes  = [max(50, hub_lookup.get(n, 0.1) * 800) for n in sub.nodes()]
    node_colors = [ACCENT_1 if G.nodes[n]["gender"] == 1 else ACCENT_2
                   for n in sub.nodes()]

    # Edge alphas by attr_rating
    edge_alphas = [
        min(1.0, G[u][v].get("attr_rating", 5) / 10)
        for u, v in sub.edges()
    ]

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    nx.draw_networkx_edges(
        sub, pos, ax=ax,
        arrows=True, arrowsize=8,
        edge_color=[f"#{int(255*a):02x}{int(140*a):02x}{int(200*a):02x}"
                    for a in edge_alphas],
        width=0.8, alpha=0.6,
    )
    nx.draw_networkx_nodes(
        sub, pos, ax=ax,
        node_size=node_sizes, node_color=node_colors, alpha=0.9,
    )

    # Gender legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=ACCENT_1, label="Men"),
        Patch(facecolor=ACCENT_2, label="Women"),
    ]
    ax.legend(handles=legend_els, facecolor=PANEL_COLOR,
              edgecolor="#2E3347", labelcolor=TEXT_COLOR, loc="upper left")

    ax.set_title(
        f"Bipartite YES-Graph  |  Wave {wave}\n"
        "Node size = Hub Score (desirability)  |  Arrow = YES decision",
        color=TEXT_COLOR, fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    _save(fig, f"11_network_wave{wave}")


# ── 4. Hub Score vs Selectivity scatter ──────────────────────────────────────

def plot_hub_vs_selectivity(metrics_df: pd.DataFrame) -> None:
    """
    Scatter: hub_score (x) vs selectivity (y), coloured by gender.
    Quadrants reveal strategic archetypes:
      High hub + high sel  = Highly desired AND picky  ('Elite')
      High hub + low sel   = Desired but not picky     ('Approachable')
      Low hub  + high sel  = Not desired AND picky     ('Gatekeeping')
      Low hub  + low sel   = Not desired, says yes anyway ('Eager')
    """
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    for gender_id, gender_name, color in [
        (1, "Men",   ACCENT_1),
        (0, "Women", ACCENT_2),
    ]:
        sub = metrics_df[metrics_df["gender"] == gender_id]
        ax.scatter(
            sub["hub_score"], sub["selectivity"],
            alpha=0.55, s=40, c=color, label=gender_name,
            edgecolors="none",
        )

    # Quadrant lines at medians
    m_hub = metrics_df["hub_score"].median()
    m_sel = metrics_df["selectivity"].median()
    ax.axvline(m_hub, color=TEXT_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(m_sel, color=TEXT_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)

    # Quadrant labels
    for (xi, yi, label) in [
        (m_hub + 0.05, m_sel + 0.05, "Elite"),
        (m_hub + 0.05, m_sel - 0.08, "Approachable"),
        (m_hub - 0.12, m_sel + 0.05, "Gatekeeping"),
        (m_hub - 0.12, m_sel - 0.08, "Eager"),
    ]:
        if 0 <= xi <= 1 and 0 <= yi <= 1:
            ax.text(xi, yi, label, color=ACCENT_4, fontsize=10,
                    fontweight="bold", alpha=0.8)

    ax.set_xlabel("Hub Score (desirability — fraction of YES received)", color=TEXT_COLOR)
    ax.set_ylabel("Selectivity (pickiness — fraction of YES NOT given)", color=TEXT_COLOR)
    ax.set_title(
        "Attractiveness Hubs & Selectivity Landscape\n"
        "Every point = one speed-dating participant",
        color=TEXT_COLOR, fontsize=13, fontweight="bold",
    )
    ax.legend(facecolor=PANEL_COLOR, edgecolor="#2E3347", labelcolor=TEXT_COLOR)
    _save(fig, "12_hub_vs_selectivity")


# ── 5. HITS authority / hub distribution ─────────────────────────────────────

def plot_hits_distribution(metrics_df: pd.DataFrame) -> None:
    """KDE plots of HITS authority and hub scores by gender."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)

    for ax, metric, title in [
        (axes[0], "hits_authority", "HITS Authority Score\n(chosen by desirable people)"),
        (axes[1], "hits_hub",       "HITS Hub Score\n(chooses desirable people)"),
    ]:
        ax.set_facecolor(PANEL_COLOR)
        for gender_id, gender_name, color in [
            (1, "Men",   ACCENT_1),
            (0, "Women", ACCENT_2),
        ]:
            sub = metrics_df[metrics_df["gender"] == gender_id][metric].dropna()
            sns.kdeplot(sub, ax=ax, color=color, fill=True, alpha=0.3,
                        label=gender_name, linewidth=2)
        ax.set_title(title, color=TEXT_COLOR)
        ax.set_xlabel(metric.replace("_", " ").title(), color=TEXT_COLOR)
        ax.set_ylabel("Density", color=TEXT_COLOR)
        ax.legend(facecolor=PANEL_COLOR, edgecolor="#2E3347",
                  labelcolor=TEXT_COLOR)

    fig.suptitle("HITS Scores — Who Are the Hubs and Authorities?",
                 fontsize=14, color=TEXT_COLOR, fontweight="bold")
    fig.tight_layout(pad=2)
    _save(fig, "13_hits_distribution")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_network_analysis() -> nx.DiGraph:
    """Full network-analysis pipeline."""
    df = pd.read_parquet(CLEAN_PARQUET)

    print("\n→ Building bipartite YES-graph …")
    G = build_bipartite_graph(df)

    print("→ Computing node metrics …")
    metrics_df = compute_node_metrics(G, df)

    print("→ Network visualisation (wave sample) …")
    # Pick a medium-sized wave for clarity
    wave_sizes = df.groupby("wave")["iid"].nunique()
    target_wave = wave_sizes[(wave_sizes >= 15) & (wave_sizes <= 30)].index
    chosen_wave = int(target_wave[0]) if len(target_wave) > 0 else 11
    plot_network_sample(G, metrics_df, wave=chosen_wave, df=df)

    print("→ Hub vs Selectivity scatter …")
    plot_hub_vs_selectivity(metrics_df)

    print("→ HITS score distributions …")
    plot_hits_distribution(metrics_df)

    # Summary statistics
    summary = metrics_df.groupby("gender_label")[
        ["hub_score", "selectivity", "reciprocity",
         "hits_authority", "hits_hub"]
    ].describe().round(3)
    summary.to_csv(OUT_REPORTS / "network_summary.csv")
    print(f"  [csv] network summary saved")
    print("\n[network] All plots saved.")
    return G


if __name__ == "__main__":
    run_network_analysis()
