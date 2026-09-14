"""
DataStore — lightweight in-memory / on-disk data access service.

On startup, checks for existing parquet files in data/synthetic/.
If not found, generates them automatically.

This module is the single source of truth for all DataFrames consumed
by the API layer.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "src/data/synthetic")
_REQUIRED_FILES = ["weather", "grid", "assets", "bess"]


def _parquet_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.parquet")


def _ensure_data() -> None:
    """Generate synthetic data if any required parquet file is missing."""
    missing = [n for n in _REQUIRED_FILES if not Path(_parquet_path(n)).exists()]
    if missing:
        print(f"[DataStore] Missing datasets: {missing} — generating synthetic data …")
        from src.data.synthetic_generator import generate_all
        generate_all(output_dir=DATA_DIR)


def load_weather() -> pd.DataFrame:
    _ensure_data()
    df = pd.read_parquet(_parquet_path("weather"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_grid() -> pd.DataFrame:
    _ensure_data()
    df = pd.read_parquet(_parquet_path("grid"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_assets() -> pd.DataFrame:
    _ensure_data()
    df = pd.read_parquet(_parquet_path("assets"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_bess() -> pd.DataFrame:
    _ensure_data()
    df = pd.read_parquet(_parquet_path("bess"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Cached model singleton
# ---------------------------------------------------------------------------

_MODEL = None
_MODEL_PATH = os.getenv("MODEL_PATH", "src/ml/saved_models/load_forecast_xgb.json")
_MODEL_METRICS: dict = {}


def get_or_train_model(force_retrain: bool = False):
    """
    Load the pre-trained XGBoost model from disk, or train a new one.

    On first call, this may take 30–60 seconds to train.
    """
    global _MODEL, _MODEL_METRICS
    if _MODEL is not None and not force_retrain:
        return _MODEL, _MODEL_METRICS

    model_path = Path(_MODEL_PATH)
    if model_path.exists() and not force_retrain:
        from src.ml.load_forecasting import load_model
        _MODEL = load_model(str(model_path))
        print(f"[DataStore] Loaded pre-trained model from {model_path}")
        return _MODEL, _MODEL_METRICS

    # Train fresh
    print("[DataStore] Training XGBoost load forecast model …")
    from src.ml.load_forecasting import train
    grid_df = load_grid()
    weather_df = load_weather()
    _MODEL, _MODEL_METRICS, _ = train(grid_df, weather_df, model_path=str(model_path))
    return _MODEL, _MODEL_METRICS
