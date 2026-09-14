"""
XGBoost-based load forecasting pipeline.

Pipeline stages:
  1. Feature engineering (temporal + weather features)
  2. Chronological train / validation / test split
  3. XGBoost training with early stopping on validation set
  4. Evaluation: MAE, RMSE, R²
  5. 24-hour ahead forecast

LEAKAGE PREVENTION:
  - Target (grid_load_mw) is never used as an input feature.
  - All lag features are computed only on past observations.
  - The split is strictly chronological — no shuffling.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "grid_load_mw"
TEMPORAL_FEATURES = ["hour", "day_of_week", "month", "is_weekend", "day_of_year"]
WEATHER_FEATURES = ["temperature", "humidity"]
LAG_HOURS = [1, 2, 3, 24, 48, 168]          # 1h,2h,3h,1d,2d,1w lags
ROLLING_WINDOWS = [3, 6, 24]                  # rolling mean windows

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST = remaining 15%

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(grid_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge grid and weather, then engineer features.
    Returns a DataFrame with feature columns and the target column.
    Target column is `grid_load_mw`.
    """
    df = grid_df[["timestamp", TARGET]].copy()
    df = df.merge(
        weather_df[["timestamp"] + WEATHER_FEATURES],
        on="timestamp",
        how="left",
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    # Lag features — use ONLY past values (no leakage)
    for lag in LAG_HOURS:
        df[f"load_lag_{lag}h"] = df[TARGET].shift(lag)

    # Rolling mean features — also strictly past
    for window in ROLLING_WINDOWS:
        df[f"load_rolling_mean_{window}h"] = (
            df[TARGET].shift(1).rolling(window=window, min_periods=1).mean()
        )

    # Drop rows where lags create NaN (first max(LAG_HOURS) rows)
    df = df.dropna().reset_index(drop=True)
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (excludes timestamp and target)."""
    return [c for c in df.columns if c not in ("timestamp", TARGET)]


# ---------------------------------------------------------------------------
# Train / val / test split (chronological)
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame in chronological order.
    Returns (train, val, test).
    """
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    grid_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    model_path: Optional[str] = None,
) -> Tuple[object, dict, pd.DataFrame]:
    """
    Build features, split chronologically, train XGBoost, evaluate.

    Returns:
      (model, metrics_dict, test_predictions_df)
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is required for load forecasting. Run: pip install xgboost")

    df = build_features(grid_df, weather_df)
    train_df, val_df, test_df = chronological_split(df)

    feature_cols = _feature_columns(df)

    X_train, y_train = train_df[feature_cols].values, train_df[TARGET].values
    X_val,   y_val   = val_df[feature_cols].values,   val_df[TARGET].values
    X_test,  y_test  = test_df[feature_cols].values,  test_df[TARGET].values

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate on all splits
    metrics = {}
    for name, X, y in [("train", X_train, y_train),
                        ("val",   X_val,   y_val),
                        ("test",  X_test,  y_test)]:
        preds = model.predict(X)
        metrics[name] = {
            "mae":  float(mean_absolute_error(y, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "r2":   float(r2_score(y, preds)),
            "n":    int(len(y)),
        }

    test_preds_df = test_df[["timestamp", TARGET]].copy()
    test_preds_df["predicted_load_mw"] = model.predict(X_test)

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        print(f"  Model saved -> {model_path}")

    print("\nLoad Forecasting -- Evaluation Metrics")
    print("-" * 45)
    for split, m in metrics.items():
        print(f"  {split:5s}  MAE={m['mae']:.2f} MW  RMSE={m['rmse']:.2f} MW  R2={m['r2']:.4f}  (n={m['n']})")

    return model, metrics, test_preds_df


# ---------------------------------------------------------------------------
# 24-hour ahead forecast
# ---------------------------------------------------------------------------

def forecast_24h(
    model,
    grid_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    forecast_start: Optional[datetime] = None,
    weather_forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Produce a 24-hour load forecast starting from `forecast_start`.

    If weather_forecast_df is not provided, the last available weather
    values are repeated (persistence forecast for weather).

    Returns a DataFrame with columns: [timestamp, forecast_load_mw].
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is required")

    df = build_features(grid_df, weather_df)
    feature_cols = _feature_columns(df)

    if forecast_start is None:
        forecast_start = df["timestamp"].max() + timedelta(hours=1)

    # Build future timestamps
    future_ts = [forecast_start + timedelta(hours=h) for h in range(24)]

    # Use last available weather or provided forecast weather
    if weather_forecast_df is not None:
        weather_future = weather_forecast_df.copy()
    else:
        last_weather = weather_df.sort_values("timestamp").tail(24).copy()
        # Cycle last-day weather values to fill 24-hour forecast window
        weather_future = last_weather.copy()
        weather_future["timestamp"] = future_ts[: len(weather_future)]

    # Seed from last historical observations
    historical_loads = df.set_index("timestamp")[TARGET]

    predictions = []
    running_loads = historical_loads.copy()

    for ts in future_ts:
        hour = ts.hour
        dow = ts.weekday()
        month = ts.month
        is_weekend = int(dow >= 5)
        doy = ts.timetuple().tm_yday

        # Weather at this timestamp
        w_row = weather_future[weather_future["timestamp"] <= ts]
        if w_row.empty:
            w_row = weather_df.sort_values("timestamp").tail(1)
        else:
            w_row = w_row.tail(1)
        temperature = float(w_row["temperature"].iloc[0]) if "temperature" in w_row.columns else 20.0
        humidity = float(w_row["humidity"].iloc[0]) if "humidity" in w_row.columns else 55.0

        # Lag features using running predictions
        lag_vals = {}
        for lag in LAG_HOURS:
            lag_ts = ts - timedelta(hours=lag)
            if lag_ts in running_loads.index:
                lag_vals[f"load_lag_{lag}h"] = running_loads[lag_ts]
            else:
                # Fall back to same-hour average
                same_hour = running_loads[running_loads.index.hour == hour]
                lag_vals[f"load_lag_{lag}h"] = float(same_hour.mean()) if not same_hour.empty else 1000.0

        rolling_vals = {}
        for window in ROLLING_WINDOWS:
            recent = running_loads.tail(window)
            rolling_vals[f"load_rolling_mean_{window}h"] = float(recent.mean()) if not recent.empty else 1000.0

        row_dict = {
            "hour": hour, "day_of_week": dow, "month": month,
            "is_weekend": is_weekend, "day_of_year": doy,
            "temperature": temperature, "humidity": humidity,
            **lag_vals, **rolling_vals,
        }
        X = np.array([[row_dict[c] for c in feature_cols]])
        pred = float(model.predict(X)[0])
        predictions.append({"timestamp": ts, "forecast_load_mw": round(pred, 2)})
        running_loads[ts] = pred  # update rolling state

    return pd.DataFrame(predictions)


# ---------------------------------------------------------------------------
# Load a saved model
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load a previously saved XGBoost model."""
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is required")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model
