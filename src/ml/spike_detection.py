"""
Demand spike detection built on top of load forecasts.

Deterministic, threshold-based classification.  All thresholds are
configurable via the SpikeDetectorConfig dataclass or environment variables.

Risk classification logic:
  CRITICAL  — reserve_margin < CRITICAL_RESERVE_MARGIN
  HIGH      — reserve_margin < HIGH_RESERVE_MARGIN
  MEDIUM    — reserve_margin < MEDIUM_RESERVE_MARGIN  OR  forecast > SPIKE_ABS_THRESHOLD_MW
  LOW       — otherwise

Spike risk score is a continuous [0, 1] value computed from the reserve margin.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpikeDetectorConfig:
    """All thresholds are configurable.  Defaults reflect conservative grid ops."""
    # Reserve-margin thresholds (fraction of available capacity)
    critical_reserve_margin: float = float(os.getenv("SPIKE_CRITICAL_RESERVE", "0.05"))  # <5%
    high_reserve_margin: float     = float(os.getenv("SPIKE_HIGH_RESERVE",     "0.10"))  # <10%
    medium_reserve_margin: float   = float(os.getenv("SPIKE_MEDIUM_RESERVE",   "0.20"))  # <20%

    # Absolute spike threshold (MW) — load > this triggers at least MEDIUM
    spike_abs_threshold_mw: float  = float(os.getenv("SPIKE_ABS_THRESHOLD_MW", "1700"))

    # Capacity fallback when available_capacity_mw is missing
    default_capacity_mw: float = float(os.getenv("DEFAULT_CAPACITY_MW", "2600"))


DEFAULT_CONFIG = SpikeDetectorConfig()


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def _compute_risk_score(reserve_margin: float) -> float:
    """
    Maps reserve_margin → risk_score in [0, 1].
    Linear between 0% margin (score=1.0) and 30% margin (score=0.0).
    Values outside that range are clipped.
    """
    score = 1.0 - (reserve_margin / 0.30)
    return float(max(0.0, min(1.0, score)))


def _classify_risk(
    reserve_margin: float,
    forecast_load_mw: float,
    cfg: SpikeDetectorConfig,
) -> str:
    if reserve_margin < cfg.critical_reserve_margin:
        return "CRITICAL"
    if reserve_margin < cfg.high_reserve_margin:
        return "HIGH"
    if reserve_margin < cfg.medium_reserve_margin or forecast_load_mw > cfg.spike_abs_threshold_mw:
        return "MEDIUM"
    return "LOW"


def detect_spikes(
    forecast_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    cfg: Optional[SpikeDetectorConfig] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    forecast_df : DataFrame with columns [timestamp, forecast_load_mw]
    capacity_df : DataFrame with columns [timestamp, available_capacity_mw].
                  If None or missing timestamps, default capacity is used.
    cfg         : SpikeDetectorConfig (defaults to DEFAULT_CONFIG)

    Returns
    -------
    DataFrame with columns:
      timestamp, forecast_load_mw, available_capacity_mw,
      reserve_margin, spike_risk_score, risk_level
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    df = forecast_df[["timestamp", "forecast_load_mw"]].copy()

    if capacity_df is not None and "available_capacity_mw" in capacity_df.columns:
        cap = capacity_df[["timestamp", "available_capacity_mw"]].copy()
        df = df.merge(cap, on="timestamp", how="left")
        df["available_capacity_mw"] = df["available_capacity_mw"].fillna(cfg.default_capacity_mw)
    else:
        df["available_capacity_mw"] = cfg.default_capacity_mw

    df["reserve_margin"] = (
        (df["available_capacity_mw"] - df["forecast_load_mw"]) / df["available_capacity_mw"]
    ).clip(lower=0.0)

    df["spike_risk_score"] = df["reserve_margin"].apply(_compute_risk_score)

    df["risk_level"] = df.apply(
        lambda r: _classify_risk(r["reserve_margin"], r["forecast_load_mw"], cfg),
        axis=1,
    )

    # Round for readability
    df["reserve_margin"] = df["reserve_margin"].round(4)
    df["spike_risk_score"] = df["spike_risk_score"].round(4)

    return df


def summarise_risk(risk_df: pd.DataFrame) -> dict:
    """Return a summary dict describing the risk window."""
    counts = risk_df["risk_level"].value_counts().to_dict()
    return {
        "total_periods": len(risk_df),
        "critical_count": counts.get("CRITICAL", 0),
        "high_count": counts.get("HIGH", 0),
        "medium_count": counts.get("MEDIUM", 0),
        "low_count": counts.get("LOW", 0),
        "max_risk_score": float(risk_df["spike_risk_score"].max()),
        "min_reserve_margin": float(risk_df["reserve_margin"].min()),
        "peak_forecast_mw": float(risk_df["forecast_load_mw"].max()),
    }


