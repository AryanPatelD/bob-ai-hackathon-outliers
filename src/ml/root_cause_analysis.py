"""
Evidence-based root cause analysis (RCA) for renewable asset anomalies.

Design principles:
  1. Separate WEATHER-EXPLAINED losses from UNEXPLAINED losses.
  2. Never claim a mechanical fault without supporting telemetry evidence.
  3. Each candidate cause is returned with a confidence score in [0, 1].
  4. Confidence scores are additive evidence — they do NOT sum to 1.
  5. Recommended inspection is triggered only when unexplained loss is
     material (> INSPECTION_THRESHOLD_MW) AND high-confidence weather
     explanation does not account for the majority of the loss.

Return schema
--------------
{
  asset_id, asset_name, asset_type,
  timestamp,
  expected_generation_mw,
  actual_generation_mw,
  deviation_mw,
  deviation_pct,
  weather_explained_loss_mw,
  unexplained_loss_mw,
  probable_causes: [{ cause, confidence, description }, ...],
  overall_confidence,
  recommended_inspection,
  inspection_notes,
}
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RCAConfig:
    # Cloud cover above which cloud shading is a credible cause (%)
    cloud_cover_threshold_high: float = float(os.getenv("RCA_CLOUD_HIGH", "60"))
    cloud_cover_threshold_med:  float = float(os.getenv("RCA_CLOUD_MED",  "30"))

    # Irradiance reduction relative to clear-sky
    irradiance_low_threshold: float = float(os.getenv("RCA_IRRADIANCE_LOW", "200"))   # W/m²

    # Temperature above which derating is material (°C)
    temp_derate_threshold: float = float(os.getenv("RCA_TEMP_DERATE", "35"))

    # Wind thresholds
    wind_cut_in_ms:  float = float(os.getenv("RCA_WIND_CUT_IN",  "3.0"))
    wind_rated_ms:   float = float(os.getenv("RCA_WIND_RATED",   "12.0"))
    wind_cut_out_ms: float = float(os.getenv("RCA_WIND_CUT_OUT", "25.0"))
    wind_low_ms:     float = float(os.getenv("RCA_WIND_LOW",     "5.0"))

    # Threshold for recommending physical inspection (MW unexplained loss)
    inspection_threshold_mw: float = float(os.getenv("RCA_INSPECTION_THRESHOLD_MW", "3.0"))

    # Performance ratio below which sensor/data issue is suspected
    sensor_suspect_ratio: float = float(os.getenv("RCA_SENSOR_SUSPECT_RATIO", "0.05"))


DEFAULT_RCA_CONFIG = RCAConfig()


# ---------------------------------------------------------------------------
# Solar RCA
# ---------------------------------------------------------------------------

def _solar_rca(
    actual_mw: float,
    expected_mw: float,
    weather: dict,
    capacity_mw: float,
    cfg: RCAConfig,
) -> tuple[float, float, list[dict]]:
    """
    Returns (weather_explained_loss_mw, unexplained_loss_mw, causes).
    """
    total_loss = expected_mw - actual_mw  # positive = underperforming
    if total_loss <= 0:
        return 0.0, 0.0, []

    causes: list[dict] = []
    weather_explained = 0.0

    cloud_cover = weather.get("cloud_cover", 0)
    irradiance  = weather.get("irradiance", 500)
    temperature = weather.get("temperature", 25)

    # ---- Cloud shading ----
    if cloud_cover >= cfg.cloud_cover_threshold_high:
        conf = 0.90
        exp_loss = total_loss * 0.70
        weather_explained += exp_loss
        causes.append({
            "cause": "cloud_cover",
            "confidence": conf,
            "description": f"Heavy cloud cover ({cloud_cover:.0f}%) explains most of the generation loss. "
                           f"Estimated weather-attributable loss: {exp_loss:.2f} MW.",
        })
    elif cloud_cover >= cfg.cloud_cover_threshold_med:
        conf = 0.60
        exp_loss = total_loss * 0.40
        weather_explained += exp_loss
        causes.append({
            "cause": "cloud_cover",
            "confidence": conf,
            "description": f"Moderate cloud cover ({cloud_cover:.0f}%) partially explains generation reduction. "
                           f"Estimated weather-attributable loss: {exp_loss:.2f} MW.",
        })

    # ---- Low irradiance (independent of cloud cover — could be haze, aerosols) ----
    if irradiance < cfg.irradiance_low_threshold and cloud_cover < cfg.cloud_cover_threshold_med:
        conf = 0.50
        exp_loss = total_loss * 0.30
        weather_explained += exp_loss
        causes.append({
            "cause": "irradiance_reduction",
            "confidence": conf,
            "description": f"Low irradiance ({irradiance:.0f} W/m²) without significant cloud cover — "
                           f"possibly haze, aerosols or sensor offset.",
        })

    # ---- Temperature derating ----
    if temperature > cfg.temp_derate_threshold:
        temp_loss_fraction = 0.0045 * (temperature - 25)
        exp_loss = capacity_mw * temp_loss_fraction
        weather_explained = min(weather_explained + exp_loss, total_loss)
        causes.append({
            "cause": "temperature_derating",
            "confidence": 0.85,
            "description": f"High ambient temperature ({temperature:.1f}°C) causes module derating "
                           f"(≈{temp_loss_fraction*100:.1f}% reduction). Est. loss: {exp_loss:.2f} MW.",
        })

    # Cap weather-explained to total loss
    weather_explained = min(weather_explained, total_loss)
    unexplained = max(0.0, total_loss - weather_explained)

    # ---- Inverter/string issue (unexplained residual) ----
    perf_ratio = actual_mw / expected_mw if expected_mw > 0 else 1.0
    if unexplained > cfg.inspection_threshold_mw and perf_ratio < 0.70:
        conf = 0.40 + 0.30 * (1 - perf_ratio)  # higher confidence at lower perf ratio
        causes.append({
            "cause": "inverter_or_string_issue",
            "confidence": round(min(0.80, conf), 2),
            "description": f"Unexplained generation loss of {unexplained:.2f} MW (perf ratio {perf_ratio:.2f}) "
                           f"suggests possible inverter failure, string disconnection, or soiling.",
        })

    # ---- Sensor / data issue ----
    if perf_ratio < cfg.sensor_suspect_ratio and actual_mw < 0.5:
        causes.append({
            "cause": "sensor_or_data_issue",
            "confidence": 0.35,
            "description": "Near-zero reported generation during daylight hours with reasonable expected output. "
                           "Possible meter, SCADA, or data pipeline fault — verify before dispatching maintenance.",
        })

    return round(weather_explained, 4), round(unexplained, 4), causes


# ---------------------------------------------------------------------------
# Wind RCA
# ---------------------------------------------------------------------------

def _wind_rca(
    actual_mw: float,
    expected_mw: float,
    weather: dict,
    capacity_mw: float,
    cfg: RCAConfig,
) -> tuple[float, float, list[dict]]:
    """
    Returns (weather_explained_loss_mw, unexplained_loss_mw, causes).
    """
    total_loss = expected_mw - actual_mw
    if total_loss <= 0:
        return 0.0, 0.0, []

    causes: list[dict] = []
    weather_explained = 0.0

    wind_speed = weather.get("wind_speed", 8.0)

    # ---- Insufficient wind ----
    if wind_speed < cfg.wind_cut_in_ms:
        weather_explained = total_loss
        causes.append({
            "cause": "insufficient_wind",
            "confidence": 0.95,
            "description": f"Wind speed ({wind_speed:.1f} m/s) is below cut-in speed "
                           f"({cfg.wind_cut_in_ms} m/s). Zero generation is expected and correct.",
        })
        return round(weather_explained, 4), 0.0, causes

    if wind_speed < cfg.wind_low_ms:
        exp_loss = total_loss * 0.80
        weather_explained += exp_loss
        causes.append({
            "cause": "insufficient_wind",
            "confidence": 0.80,
            "description": f"Low wind speed ({wind_speed:.1f} m/s) below rated speed "
                           f"({cfg.wind_rated_ms} m/s) explains most generation reduction.",
        })

    # ---- Excessive wind / cut-out ----
    if wind_speed >= cfg.wind_cut_out_ms:
        weather_explained = total_loss
        causes.append({
            "cause": "excessive_wind_cut_out",
            "confidence": 0.95,
            "description": f"Wind speed ({wind_speed:.1f} m/s) exceeds cut-out speed "
                           f"({cfg.wind_cut_out_ms} m/s). Turbines are designed to shut down for safety.",
        })
        return round(weather_explained, 4), 0.0, causes

    # Cap weather explained
    weather_explained = min(weather_explained, total_loss)
    unexplained = max(0.0, total_loss - weather_explained)

    perf_ratio = actual_mw / expected_mw if expected_mw > 0 else 1.0

    # ---- Turbine availability / mechanical issue ----
    if unexplained > cfg.inspection_threshold_mw and perf_ratio < 0.70:
        conf = 0.40 + 0.25 * (1 - perf_ratio)
        causes.append({
            "cause": "turbine_availability",
            "confidence": round(min(0.80, conf), 2),
            "description": f"Unexplained loss of {unexplained:.2f} MW at wind speed {wind_speed:.1f} m/s "
                           f"(perf ratio {perf_ratio:.2f}) may indicate turbine downtime, yaw misalignment, "
                           f"or gearbox degradation.  Physical inspection recommended.",
        })

    # ---- Grid curtailment ----
    if unexplained > cfg.inspection_threshold_mw and perf_ratio > 0.30:
        causes.append({
            "cause": "curtailment",
            "confidence": 0.30,
            "description": f"Partial generation shortfall at adequate wind speed could indicate "
                           f"grid curtailment or active power limit imposed by system operator.",
        })

    # ---- Sensor / data issue ----
    if perf_ratio < cfg.sensor_suspect_ratio and actual_mw < 0.5:
        causes.append({
            "cause": "sensor_or_data_issue",
            "confidence": 0.35,
            "description": "Near-zero generation reported at wind speeds above cut-in. "
                           "Consider verifying anemometer and SCADA meter accuracy before maintenance.",
        })

    return round(weather_explained, 4), round(unexplained, 4), causes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diagnose_asset(
    asset_id: str,
    asset_name: str,
    asset_type: str,
    actual_mw: float,
    expected_mw: float,
    capacity_mw: float,
    weather: dict,
    timestamp: Optional[datetime] = None,
    cfg: Optional[RCAConfig] = None,
) -> dict:
    """
    Run root cause analysis for a single asset observation.

    Parameters
    ----------
    asset_id / asset_name / asset_type : asset metadata
    actual_mw   : measured generation (MW)
    expected_mw : physics-model expected generation (MW)
    capacity_mw : nameplate capacity (MW)
    weather     : dict with keys: cloud_cover, irradiance, temperature,
                  wind_speed, wind_direction
    timestamp   : observation time (optional, for record-keeping)
    cfg         : RCAConfig

    Returns
    -------
    dict matching the AssetDiagnosis schema
    """
    if cfg is None:
        cfg = DEFAULT_RCA_CONFIG

    deviation_mw = actual_mw - expected_mw
    deviation_pct = deviation_mw / expected_mw if expected_mw > 0 else 0.0

    if asset_type == "solar":
        wx_loss, ux_loss, causes = _solar_rca(
            actual_mw, expected_mw, weather, capacity_mw, cfg
        )
    elif asset_type == "wind":
        wx_loss, ux_loss, causes = _wind_rca(
            actual_mw, expected_mw, weather, capacity_mw, cfg
        )
    else:
        wx_loss, ux_loss, causes = 0.0, max(0.0, expected_mw - actual_mw), []

    # Overall confidence: average of cause confidences, or low if no causes
    if causes:
        overall_conf = round(sum(c["confidence"] for c in causes) / len(causes), 3)
    else:
        overall_conf = 0.0

    # Recommend inspection when unexplained loss is material
    inspect = ux_loss > cfg.inspection_threshold_mw
    notes_parts = []
    if inspect:
        notes_parts.append(f"Unexplained loss of {ux_loss:.2f} MW warrants on-site inspection.")
    if not causes:
        notes_parts.append("Generation within expected range — no action required.")
    inspection_notes = " ".join(notes_parts) if notes_parts else "No anomaly detected."

    return {
        "asset_id": asset_id,
        "asset_name": asset_name,
        "asset_type": asset_type,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "expected_generation_mw": expected_mw,
        "actual_generation_mw": actual_mw,
        "deviation_mw": round(deviation_mw, 4),
        "deviation_pct": round(deviation_pct, 4),
        "weather_explained_loss_mw": wx_loss,
        "unexplained_loss_mw": ux_loss,
        "probable_causes": causes,
        "overall_confidence": overall_conf,
        "recommended_inspection": inspect,
        "inspection_notes": inspection_notes,
    }


def diagnose_all(
    assets_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    cfg: Optional[RCAConfig] = None,
) -> list[dict]:
    """
    Run RCA for every row in assets_df, joining weather by timestamp.

    Returns a list of diagnosis dicts.
    """
    if cfg is None:
        cfg = DEFAULT_RCA_CONFIG

    # Keep latest weather per timestamp
    weather_indexed = weather_df.set_index("timestamp")

    diagnoses = []
    for _, row in assets_df.iterrows():
        ts = row["timestamp"]
        # Nearest weather record
        if ts in weather_indexed.index:
            w = weather_indexed.loc[ts].to_dict()
        else:
            nearest = weather_indexed.index.get_indexer([ts], method="nearest")
            w = weather_indexed.iloc[nearest[0]].to_dict()

        diag = diagnose_asset(
            asset_id=row["asset_id"],
            asset_name=row["asset_name"],
            asset_type=row["asset_type"],
            actual_mw=row["actual_generation_mw"],
            expected_mw=row["expected_generation_mw"],
            capacity_mw=row["capacity_mw"],
            weather=w,
            timestamp=ts,
            cfg=cfg,
        )
        diagnoses.append(diag)
    return diagnoses
