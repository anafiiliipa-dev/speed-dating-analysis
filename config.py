"""config.py - Speed Dating Pipeline configuration."""

from __future__ import annotations
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="SD_",
        extra="ignore",
    )
    data_raw: Path = Field(default=Path("data/Speed+Dating+Data.csv"))
    log_level: str = Field(default="INFO")
    random_seed: int = Field(default=42, ge=0)
    n_jobs: int = Field(default=-1)

    @field_validator("data_raw", mode="after")
    @classmethod
    def _resolve_data_raw(cls, v):
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @field_validator("log_level", mode="after")
    @classmethod
    def _validate_log_level(cls, v):
        v = v.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return v

    @property
    def out_data(self): return PROJECT_ROOT / "outputs" / "data"
    @property
    def out_figs(self): return PROJECT_ROOT / "outputs" / "figures"
    @property
    def out_reports(self): return PROJECT_ROOT / "outputs" / "reports"
    @property
    def out_logs(self): return PROJECT_ROOT / "outputs" / "logs"
    @property
    def clean_parquet(self): return self.out_data / "speed_dating_clean.parquet"
    @property
    def model_parquet(self): return self.out_data / "speed_dating_model_ready.parquet"


settings = Settings()

ROOT          = PROJECT_ROOT
DATA_RAW      = settings.data_raw
OUT_DATA      = settings.out_data
OUT_FIGS      = settings.out_figs
OUT_REPORTS   = settings.out_reports
CLEAN_PARQUET = settings.clean_parquet
MODEL_PARQUET = settings.model_parquet

LIKERT_WAVES = frozenset({6, 7, 8, 9})
ALLOC_WAVES  = frozenset(set(range(1, 22)) - LIKERT_WAVES)

ATTR_DIMS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
STATED_PREF_COLS = [f"{d}1_1" for d in ATTR_DIMS]
STATED_PREF_CLR  = [f"{d}1_1_clr" for d in ATTR_DIMS]
SELF_PERC_COLS = [f"{d}3_1" for d in ["attr", "sinc", "fun", "intel", "amb"]]
PARTNER_PREF_COLS = ["pf_o_att", "pf_o_sin", "pf_o_int", "pf_o_fun", "pf_o_amb", "pf_o_sha"]
RATINGS_GIVEN = list(ATTR_DIMS) + ["like", "prob"]
RATINGS_RECV  = [f"{d}_o" for d in ATTR_DIMS] + ["like_o", "prob_o"]

T2_COLS = (["satis_2", "length", "numdat_2", "you_call", "them_cal"]
    + [f"{d}7_2" for d in ATTR_DIMS]
    + [f"{d}1_2" for d in ATTR_DIMS]
    + [f"{d}4_2" for d in ATTR_DIMS]
    + [f"{d}2_2" for d in ATTR_DIMS]
    + [f"{d}3_2" for d in ["attr", "sinc", "intel", "fun", "amb"]]
    + [f"{d}5_2" for d in ["attr", "sinc", "intel", "fun", "amb"]])

T3_COLS = (["date_3", "numdat_3", "num_in_3"]
    + [f"{d}1_3" for d in ATTR_DIMS]
    + [f"{d}7_3" for d in ATTR_DIMS]
    + [f"{d}4_3" for d in ATTR_DIMS]
    + [f"{d}2_3" for d in ATTR_DIMS]
    + [f"{d}3_3" for d in ["attr", "sinc", "intel", "fun", "amb"]]
    + [f"{d}5_3" for d in ["attr", "sinc", "intel", "fun", "amb"]])

LEAKAGE_COLS = T2_COLS + T3_COLS

LIFESTYLE_COLS = ["sports", "tvsports", "exercise", "dining", "museums", "art",
    "hiking", "gaming", "clubbing", "reading", "tv", "theater",
    "movies", "concerts", "music", "shopping", "yoga"]
DEMO_COLS = ["age", "gender", "race", "field_cd", "career_c",
    "imprace", "imprelig", "goal", "date", "go_out"]
CONTEXT_COLS = ["samerace", "age_o", "race_o", "int_corr", "wave",
    "condtn", "order", "round"]

MODEL_FEATURES = (STATED_PREF_CLR + RATINGS_GIVEN + RATINGS_RECV
    + PARTNER_PREF_COLS + CONTEXT_COLS + DEMO_COLS + LIFESTYLE_COLS)
MODEL_TARGET = "dec"

BG_COLOR    = "#0F1117"
PANEL_COLOR = "#1A1D27"
TEXT_COLOR  = "#E8EAF0"
ACCENT_1    = "#7EB8F7"
ACCENT_2    = "#F77EB8"
ACCENT_3    = "#7EF7B8"
ACCENT_4    = "#F7C97E"
PALETTE_DIV = "RdBu_r"
PALETTE_SEQ = "viridis"
PLOTLY_TEMPLATE = "plotly_dark"

SEABORN_STYLE = {
    "axes.facecolor":   PANEL_COLOR,
    "figure.facecolor": BG_COLOR,
    "axes.edgecolor":   "#2E3347",
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "grid.color":       "#2E3347",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
}

ATTR_LABELS = {"attr": "Attractiveness", "sinc": "Sincerity", "intel": "Intelligence",
    "fun": "Fun", "amb": "Ambition", "shar": "Shared Interests"}
GENDER_LABELS = {0: "Women", 1: "Men"}
