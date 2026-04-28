"""
config.py — Central configuration for the Speed Dating Pipeline.

All paths, wave metadata, column groups, and visual constants live here.
Importing this module from any step guarantees consistency across the pipeline.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_RAW = Path(r"C:\Users\anafi\Desktop\speed_dating\data\Speed+Dating+Data.csv")
OUT_DATA    = ROOT / "outputs" / "data"
OUT_FIGS    = ROOT / "outputs" / "figures"
OUT_REPORTS = ROOT / "outputs" / "reports"

CLEAN_PARQUET = OUT_DATA / "speed_dating_clean.parquet"
MODEL_PARQUET = OUT_DATA / "speed_dating_model_ready.parquet"

# ── Wave metadata ─────────────────────────────────────────────────────────────
# Waves 6-9 used a 1-10 Likert scale for stated attribute importance (attr1_1 …).
# All other waves used a 100-point budget allocation.
# We normalise waves 6-9 by ×10 so every wave lives on a [0, 100] space.
LIKERT_WAVES   = {6, 7, 8, 9}        # 1-10 scale → multiply by 10
ALLOC_WAVES    = set(range(1, 22)) - LIKERT_WAVES  # 100-pt allocation

# ── Attribute dimensions ──────────────────────────────────────────────────────
ATTR_DIMS = ["attr", "sinc", "intel", "fun", "amb", "shar"]

# ── Column groups ─────────────────────────────────────────────────────────────

# (a) Stated preference vectors — filled at signup BEFORE the event (T1)
#     Compositional: values sum ~100 per person.
STATED_PREF_COLS = [f"{d}1_1" for d in ATTR_DIMS]   # attr1_1 … shar1_1
STATED_PREF_CLR  = [f"{d}1_1_clr" for d in ATTR_DIMS]

# (b) Guess how opposite sex rates YOU (self-perception) — T1
SELF_PERC_COLS = [f"{d}3_1" for d in ["attr", "sinc", "fun", "intel", "amb"]]

# (c) Partner's stated preferences (mirrored from partner's row)
PARTNER_PREF_COLS = ["pf_o_att", "pf_o_sin", "pf_o_int",
                     "pf_o_fun", "pf_o_amb", "pf_o_sha"]

# (d) Ratings GIVEN to partner during the event (T1 — night of event)
RATINGS_GIVEN = [f"{d}" for d in ATTR_DIMS] + ["like", "prob"]

# (e) Ratings RECEIVED from partner
RATINGS_RECV  = [f"{d}_o" for d in ATTR_DIMS] + ["like_o", "prob_o"]

# (f) T2 columns (next-day survey) — must NEVER enter the ML feature matrix
T2_COLS = [
    "satis_2", "length", "numdat_2", "you_call", "them_cal",
    *[f"{d}7_2" for d in ATTR_DIMS],
    *[f"{d}1_2" for d in ATTR_DIMS],
    *[f"{d}4_2" for d in ATTR_DIMS],
    *[f"{d}2_2" for d in ATTR_DIMS],
    *[f"{d}3_2" for d in ["attr", "sinc", "intel", "fun", "amb"]],
    *[f"{d}5_2" for d in ["attr", "sinc", "intel", "fun", "amb"]],
]

# (g) T3 columns (3-week follow-up) — same leakage risk
T3_COLS = [
    "date_3", "numdat_3", "num_in_3",
    *[f"{d}1_3" for d in ATTR_DIMS],
    *[f"{d}7_3" for d in ATTR_DIMS],
    *[f"{d}4_3" for d in ATTR_DIMS],
    *[f"{d}2_3" for d in ATTR_DIMS],
    *[f"{d}3_3" for d in ["attr", "sinc", "intel", "fun", "amb"]],
    *[f"{d}5_3" for d in ["attr", "sinc", "intel", "fun", "amb"]],
]

LEAKAGE_COLS = T2_COLS + T3_COLS

# (h) Lifestyle / interests (1-10 self-ratings at signup)
LIFESTYLE_COLS = [
    "sports", "tvsports", "exercise", "dining", "museums", "art",
    "hiking", "gaming", "clubbing", "reading", "tv", "theater",
    "movies", "concerts", "music", "shopping", "yoga",
]

# (i) Person-level demographics at signup
DEMO_COLS = ["age", "gender", "race", "field_cd", "career_c",
             "imprace", "imprelig", "goal", "date", "go_out"]

# (j) Interaction context
CONTEXT_COLS = ["samerace", "age_o", "race_o", "int_corr", "wave",
                "condtn", "order", "round"]

# ── Final feature set for modelling (T1-only, anti-leakage) ──────────────────
MODEL_FEATURES = (
    STATED_PREF_CLR        # compositional preferences (CLR-transformed)
    + RATINGS_GIVEN        # how person rated partner on the night
    + RATINGS_RECV         # how partner rated person on the night
    + PARTNER_PREF_COLS    # partner's stated preferences
    + CONTEXT_COLS
    + DEMO_COLS
    + LIFESTYLE_COLS
)

MODEL_TARGET = "dec"       # binary: did THIS person say yes?

# ── Visualisation constants ───────────────────────────────────────────────────
# "Scientific Clean / Executive Dark" palette
BG_COLOR     = "#0F1117"
PANEL_COLOR  = "#1A1D27"
TEXT_COLOR   = "#E8EAF0"
ACCENT_1     = "#7EB8F7"   # cool blue — men
ACCENT_2     = "#F77EB8"   # warm pink — women
ACCENT_3     = "#7EF7B8"   # teal — match
ACCENT_4     = "#F7C97E"   # amber — highlight
PALETTE_DIV  = "RdBu_r"
PALETTE_SEQ  = "viridis"

PLOTLY_TEMPLATE = "plotly_dark"

SEABORN_STYLE = {
    "axes.facecolor":    PANEL_COLOR,
    "figure.facecolor":  BG_COLOR,
    "axes.edgecolor":    "#2E3347",
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        "#2E3347",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
}

ATTR_LABELS = {
    "attr":  "Attractiveness",
    "sinc":  "Sincerity",
    "intel": "Intelligence",
    "fun":   "Fun",
    "amb":   "Ambition",
    "shar":  "Shared Interests",
}

GENDER_LABELS = {0: "Women", 1: "Men"}
