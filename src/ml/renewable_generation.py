"""
Expected renewable generation models.

SOLAR model
-----------
Physics-based calculation using:
  - Global Horizontal Irradiance (W/m²)
  - Cloud cover (%)
  - Temperature derating (+0.45% per °C above STC 25°C)
  - Asset capacity (MW)

All steps are deterministic and explainable — no black-box ML here.

WIND model
----------
Power curve model using:
  - Wind speed at hub height (m/s)
  - Wind direction (degrees) — mild directional efficiency factor
  - Temperature (air density correction)
  - Asset capacity (MW)
  - Parametric power curve: cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s

Both models return:
  - expected_generation_mw
  - capacity_factor (0–1)
  - model_explanation (dict of intermediate values for transparency)
"""
from __future__ import annotations

import math
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Solar expected generation
# ---------------------------------------------------------------------------

# Standard test condition temperature
STC_TEMP_C = 25.0
# Temperature power coefficient (typical crystalline silicon)
TEMP_COEFF = 0.0045          # fraction per °C
# Panel conversion efficiency (proxy; actual panels ~18–22%)
PANEL_EFFICIENCY = 0.18
# GHI at standard test conditions (W/m²)
STC_GHI = 1000.0


def solar_expected(
    irradiance_wm2: float,
    cloud_cover_pct: float,
    temperature_c: float,
    capacity_mw: float,
) -> Dict[str, Any]:
    """
    Calculate expected solar generation.

    Parameters
    ----------
    irradiance_wm2  : Global Horizontal Irradiance (W/m²)
    cloud_cover_pct : Cloud cover percentage (0–100)
    temperature_c   : Ambient temperature (°C)
    capacity_mw     : Nameplate capacity (MW)

    Returns
    -------
    dict with:
      expected_generation_mw, capacity_factor, model_explanation
    """
    if irradiance_wm2 <= 0 or capacity_mw <= 0:
        return {
            "expected_generation_mw": 0.0,
            "capacity_factor": 0.0,
            "model_explanation": {
                "irradiance_wm2": irradiance_wm2,
                "cloud_cover_pct": cloud_cover_pct,
                "temperature_c": temperature_c,
                "capacity_mw": capacity_mw,
                "reason": "no_irradiance_or_zero_capacity",
            },
        }

    # Step 1: Base insolation ratio (fraction of STC)
    insolation_ratio = min(1.0, irradiance_wm2 / STC_GHI)

    # Step 2: Cloud attenuation (transmissivity model)
    # At 100% cloud cover → ~25% of clear-sky remains; at 0% → 100%
    cloud_transmissivity = 1.0 - 0.75 * (cloud_cover_pct / 100.0) ** 3

    # Step 3: Temperature derating
    temp_delta = max(0.0, temperature_c - STC_TEMP_C)
    temp_derate = 1.0 - TEMP_COEFF * temp_delta

    # Step 4: Capacity factor
    capacity_factor = insolation_ratio * cloud_transmissivity * temp_derate
    capacity_factor = max(0.0, min(1.0, capacity_factor))

    expected_mw = round(capacity_mw * capacity_factor, 4)

    return {
        "expected_generation_mw": expected_mw,
        "capacity_factor": round(capacity_factor, 4),
        "model_explanation": {
            "irradiance_wm2": irradiance_wm2,
            "cloud_cover_pct": cloud_cover_pct,
            "temperature_c": temperature_c,
            "capacity_mw": capacity_mw,
            "insolation_ratio": round(insolation_ratio, 4),
            "cloud_transmissivity": round(cloud_transmissivity, 4),
            "temp_delta_c": round(temp_delta, 2),
            "temp_derate": round(temp_derate, 4),
            "capacity_factor": round(capacity_factor, 4),
        },
    }


# ---------------------------------------------------------------------------
# Wind expected generation
# ---------------------------------------------------------------------------

# Power curve parameters
WIND_CUT_IN_MS  = 3.0    # m/s
WIND_RATED_MS   = 12.0   # m/s
WIND_CUT_OUT_MS = 25.0   # m/s

# Air density correction reference temperature (°C)
AIR_DENSITY_REF_TEMP = 15.0
# Air density at reference conditions (kg/m³)
AIR_DENSITY_REF = 1.225


def _wind_capacity_factor(wind_speed_ms: float) -> float:
    """
    Parametric power curve: cubic ramp from cut-in to rated,
    full output from rated to cut-out, zero outside range.
    """
    if wind_speed_ms < WIND_CUT_IN_MS or wind_speed_ms >= WIND_CUT_OUT_MS:
        return 0.0
    if wind_speed_ms >= WIND_RATED_MS:
        return 1.0
    cf = ((wind_speed_ms - WIND_CUT_IN_MS) / (WIND_RATED_MS - WIND_CUT_IN_MS)) ** 3
    return float(cf)


def _air_density_correction(temperature_c: float) -> float:
    """
    Approximate air density relative to reference (ISO 2533).
    Correction factor = T_ref / T_actual (Kelvin ratio).
    Wind turbine power is proportional to air density.
    """
    t_ref_k = AIR_DENSITY_REF_TEMP + 273.15
    t_act_k = temperature_c + 273.15
    return t_ref_k / t_act_k


def wind_expected(
    wind_speed_ms: float,
    wind_direction_deg: float,
    temperature_c: float,
    capacity_mw: float,
) -> Dict[str, Any]:
    """
    Calculate expected wind generation.

    Parameters
    ----------
    wind_speed_ms      : Wind speed at hub height (m/s)
    wind_direction_deg : Wind direction (degrees, 0–360)
    temperature_c      : Ambient temperature (°C)
    capacity_mw        : Nameplate capacity (MW)

    Returns
    -------
    dict with:
      expected_generation_mw, capacity_factor, model_explanation
    """
    if capacity_mw <= 0:
        return {
            "expected_generation_mw": 0.0,
            "capacity_factor": 0.0,
            "model_explanation": {"reason": "zero_capacity"},
        }

    # Step 1: Power curve capacity factor
    cf_curve = _wind_capacity_factor(wind_speed_ms)

    # Step 2: Air density correction
    density_factor = _air_density_correction(temperature_c)

    # Step 3: Combined capacity factor (density-corrected)
    capacity_factor = min(1.0, cf_curve * density_factor)

    expected_mw = round(capacity_mw * capacity_factor, 4)

    reason = None
    if wind_speed_ms < WIND_CUT_IN_MS:
        reason = "below_cut_in_speed"
    elif wind_speed_ms >= WIND_CUT_OUT_MS:
        reason = "above_cut_out_speed"

    return {
        "expected_generation_mw": expected_mw,
        "capacity_factor": round(capacity_factor, 4),
        "model_explanation": {
            "wind_speed_ms": wind_speed_ms,
            "wind_direction_deg": wind_direction_deg,
            "temperature_c": temperature_c,
            "capacity_mw": capacity_mw,
            "power_curve_cf": round(cf_curve, 4),
            "air_density_factor": round(density_factor, 4),
            "capacity_factor": round(capacity_factor, 4),
            **({"reason": reason} if reason else {}),
        },
    }


# ---------------------------------------------------------------------------
# Batch helpers for DataFrames
# ---------------------------------------------------------------------------

import pandas as pd


def compute_solar_expected_df(
    weather_df: pd.DataFrame,
    asset_capacity_mw: float,
) -> pd.DataFrame:
    """
    Apply solar_expected row-wise to a weather DataFrame.
    Returns the weather_df with columns:
      expected_generation_mw, capacity_factor
    """
    results = weather_df.apply(
        lambda r: solar_expected(
            r["irradiance"], r["cloud_cover"], r["temperature"], asset_capacity_mw
        ),
        axis=1,
    )
    out = weather_df.copy()
    out["expected_generation_mw"] = results.apply(lambda x: x["expected_generation_mw"])
    out["capacity_factor"] = results.apply(lambda x: x["capacity_factor"])
    return out


def compute_wind_expected_df(
    weather_df: pd.DataFrame,
    asset_capacity_mw: float,
) -> pd.DataFrame:
    """
    Apply wind_expected row-wise to a weather DataFrame.
    """
    results = weather_df.apply(
        lambda r: wind_expected(
            r["wind_speed"], r.get("wind_direction", 180.0),
            r["temperature"], asset_capacity_mw
        ),
        axis=1,
    )
    out = weather_df.copy()
    out["expected_generation_mw"] = results.apply(lambda x: x["expected_generation_mw"])
    out["capacity_factor"] = results.apply(lambda x: x["capacity_factor"])
    return out
