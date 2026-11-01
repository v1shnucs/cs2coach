"""Paths and global settings. Override the data root with CS2COACH_DATA."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("CS2COACH_DATA", Path(__file__).resolve().parents[2] / "data"))
RAW_DEMOS = DATA_DIR / "demos" / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
MODELS_DIR = DATA_DIR / "models"
FINDINGS_DIR = DATA_DIR / "findings"
DB_PATH = DATA_DIR / "coach.duckdb"

# Snapshot rate for the ticks table. CS2 demos record at 64 Hz; 8 Hz keeps
# a full match at roughly 200k rows while still resolving peeks and rotations.
SAMPLE_HZ = 8

TABLES = ("matches", "rounds", "ticks", "kills", "damages", "grenades", "bomb")

LLM_MODEL = os.environ.get("CS2COACH_LLM", "qwen3.5:9b")


def ensure_dirs() -> None:
    for p in (RAW_DEMOS, PARQUET_DIR, MODELS_DIR, FINDINGS_DIR):
        p.mkdir(parents=True, exist_ok=True)
