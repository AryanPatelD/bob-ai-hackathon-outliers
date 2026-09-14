"""
Anomaly detection engine for renewable asset performance.

Compares actual_generation_mw vs expected_generation_mw and classifies
anomalies using thresholds on:
  - percentage_deviation  (actual - expected) / expected
  - performance_ratio      actual / expected
  - absolute_deviation_mw  |actual - expected|

Classification rules
---------------------
CRITICAL : performance_ratio < CRITICAL_PERF_RATIO  AND  abs_deviation > CRITICAL_ABS_MW
WARNING  : performance_ratio < WARNING_PERF_RATIO   AND  abs_deviation > WARNING_ABS_MW
NORMAL   : otherwise

IMPORTANT: Poor weather alone does NOT constitute an anomaly.
  If expected_generation_mw is very small (< MIN_EXPECTED_GENERATION_MW),
  the asset is classified NORMAL regardless of actual generation,
  because weather conditions already explain low output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnomalyConfig:
    # Performance ratio thresholds (actual / expected)
    critical_perf_ratio: float = float(os.getenv("ANOMALY_CRITICAL_PERF_RATIO", "0.50"))
    warning_perf_ratio:  float = float(os.getenv("ANOMALY_WARNING_PERF_RATIO",  "0.75"))

    # Absolute deviation thresholds (MW) — guards against spurious alerts on tiny plants
    critical_abs_mw: float = float(os.getenv("ANOMALY_CRITICAL_ABS_MW", "5.0"))
    warning_abs_mw:  float = float(os.getenv("ANOMALY_WARNING_ABS_MW",  "2.0"))

    # Minimum expected generation (MW) below which we do NOT flag anomalies
    # (prevents false positives due to nighttime, calm weather, etc.)
    min_expected_mw: float = float(os.getenv("ANOMALY_MIN_EXPECTED_MW", "2.0"))


DEFAULT_ANOMALY_CONFIG = AnomalyConfig()


# ---------------------------------------------------------------------------
# Single-asset anomaly calculation
# ---------------------------------------------------------------------------

def _compute_anomaly(
    actual_mw: float,
    expected_mw: float,
    cfg: AnomalyConfig,
) -> dict:
    """
    Compute anomaly metrics and severity for a single observation.

    Returns
    -------
    dict with keys:
      absolute_deviation_mw, percentage_deviation, performance_ratio, severity
    """
    abs_dev = actual_mw - expected_mw               # negative = under-performing
    abs_dev_mag = abs(abs_dev)

    # If expected is nearly zero (bad weather/nighttime), suppress the alert
    if expected_mw < cfg.min_expected_mw:
        return {
            "absolute_deviation_mw": round(abs_dev, 4),
            "percentage_deviation": 0.0,
            "performance_ratio": 1.0,
            "severity": "NORMAL",
        }

    pct_dev = abs_dev / expected_mw                 # negative = under-performing
    perf_ratio = actual_mw / expected_mw if expected_mw > 0 else 1.0

    # Classification
    if perf_ratio < cfg.critical_perf_ratio and abs_dev_mag > cfg.critical_abs_mw:
        severity = "CRITICAL"
    elif perf_ratio < cfg.warning_perf_ratio and abs_dev_mag > cfg.warning_abs_mw:
        severity = "WARNING"
    else:
        severity = "NORMAL"

    return {
        "absolute_deviation_mw": round(abs_dev, 4),
        "percentage_deviation": round(pct_dev, 4),
        "performance_ratio": round(perf_ratio, 4),
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# Batch anomaly detection on asset DataFrame
# ---------------------------------------------------------------------------

def detect_anomalies(
    assets_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    cfg: Optional[AnomalyConfig] = None,
) -> pd.DataFrame:
    """
    Detect anomalies across all renewable assets.

    Parameters
    ----------
    assets_df : DataFrame with columns:
        timestamp, asset_id, asset_name, asset_type,
        actual_generation_mw, expected_generation_mw, capacity_mw, [status]
    weather_df : Optional weather DataFrame (unused in calculations here,
                 but may be attached to output for transparency).
    cfg : AnomalyConfig

    Returns
    -------
    DataFrame with all input columns plus:
      absolute_deviation_mw, percentage_deviation, performance_ratio, severity
    """
    if cfg is None:
        cfg = DEFAULT_ANOMALY_CONFIG

    df = assets_df.copy()

    metrics = df.apply(
        lambda r: _compute_anomaly(
            r["actual_generation_mw"],
            r["expected_generation_mw"],
            cfg,
        ),
        axis=1,
        result_type="expand",
    )

    df = pd.concat([df, metrics], axis=1)
    return df


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def anomaly_summary(anomaly_df: pd.DataFrame) -> dict:
    """Return a summary dict for the anomaly analysis."""
    total = anomaly_df["asset_id"].nunique()
    in_warning  = anomaly_df[anomaly_df["severity"] == "WARNING"]["asset_id"].nunique()
    in_critical = anomaly_df[anomaly_df["severity"] == "CRITICAL"]["asset_id"].nunique()
    return {
        "total_assets": total,
        "assets_in_warning": in_warning,
        "assets_in_critical": in_critical,
        "total_anomaly_records": int((anomaly_df["severity"] != "NORMAL").sum()),
    }
