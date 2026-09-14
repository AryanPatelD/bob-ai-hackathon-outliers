"""
Phase 2 FastAPI routes — Grid Optimisation, Curtailment, Operator Brief.

Mounted on the main app as a sub-application or included as routers.
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from src.services.data_store import (
    load_assets, load_grid, load_weather, load_bess, get_or_train_model
)
from src.ml.load_forecasting import forecast_24h
from src.ml.renewable_generation import solar_expected, wind_expected
from src.ml.spike_detection import detect_spikes
from src.ml.anomaly_detection import detect_anomalies, anomaly_summary
from src.ml.root_cause_analysis import diagnose_asset, diagnose_all
from src.ml.grid_optimiser import run_optimisation, OptimiserConfig, BESSState
from src.ml.curtailment import compute_curtailment_analysis
from src.ml.operator_brief import generate_operator_brief
from src.data.scenarios import ALL_SCENARIOS

import pandas as pd
import numpy as np

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: build optimisation-ready forecast DataFrame from live data
# ---------------------------------------------------------------------------

def _build_live_forecast_df(horizon_hours: int = 24) -> pd.DataFrame:
    model, _ = get_or_train_model()
    grid_df = load_grid()
    weather_df = load_weather()
    assets_df = load_assets()

    # Load forecast
    fc_df = forecast_24h(model, grid_df, weather_df)

    # Renewable forecast per hour
    latest_weather = weather_df.sort_values("timestamp").tail(horizon_hours)
    asset_meta = assets_df[["asset_id", "asset_type", "capacity_mw"]].drop_duplicates()

    ren_by_ts: dict = {}
    for _, w in latest_weather.iterrows():
        total_ren = 0.0
        for _, asset in asset_meta.iterrows():
            if asset["asset_type"] == "solar":
                r = solar_expected(w["irradiance"], w["cloud_cover"], w["temperature"], asset["capacity_mw"])
            else:
                r = wind_expected(w["wind_speed"], w.get("wind_direction", 180.0), w["temperature"], asset["capacity_mw"])
            total_ren += r["expected_generation_mw"]
        ren_by_ts[w["timestamp"]] = total_ren

    # Merge onto forecast
    rows = []
    for _, row in fc_df.head(horizon_hours).iterrows():
        ts = row["timestamp"]
        # Nearest renewable value
        nearest_ts = min(ren_by_ts.keys(), key=lambda t: abs((t - ts).total_seconds()), default=None)
        renewable_mw = ren_by_ts.get(nearest_ts, 0.0)

        # Grid info
        grid_row = grid_df.sort_values("timestamp").tail(1).iloc[0]
        conventional_mw = float(grid_row.get("conventional_generation_mw", 900.0))
        available_cap = float(grid_row.get("available_capacity_mw", 2600.0))

        rows.append({
            "timestamp": ts,
            "demand_mw": float(row["forecast_load_mw"]),
            "renewable_mw": renewable_mw,
            "conventional_mw": conventional_mw,
            "available_capacity_mw": available_cap,
        })
    return pd.DataFrame(rows)


def _build_scenario_context(scenario_key: str) -> dict:
    """Load a pre-defined scenario and return all needed data."""
    if scenario_key not in ALL_SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario '{scenario_key}'. Valid: A, B, C")
    return ALL_SCENARIOS[scenario_key]()


def _opt_summary(opt_df: pd.DataFrame, scenario_name: str = "") -> dict:
    """Derive a human-readable summary from optimiser results."""
    results = opt_df.to_dict(orient="records")
    deficits = [r for r in results if r.get("net_balance_mw", 0) < 0]
    surpluses = [r for r in results if r.get("net_balance_mw", 0) > 0]
    bess_charge = sum(r.get("bess_action_mw", 0) for r in results if r.get("bess_action_mw", 0) > 0)
    bess_discharge = sum(abs(r.get("bess_action_mw", 0)) for r in results if r.get("bess_action_mw", 0) < 0)
    dr_intervals = [r for r in results if abs(r.get("demand_response_mw", 0)) > 0]
    curtailment_mwh = sum(r.get("curtailment_mw", 0) for r in results)
    backup_required = any(r.get("backup_generation_mw", 0) > 0 for r in results)

    return {
        "scenario_name": scenario_name,
        "total_intervals": len(results),
        "deficit_periods": len(deficits),
        "surplus_periods": len(surpluses),
        "balanced_periods": len(results) - len(deficits) - len(surpluses),
        "total_bess_charge_mwh": round(bess_charge, 2),
        "total_bess_discharge_mwh": round(bess_discharge, 2),
        "total_dr_intervals": len(dr_intervals),
        "backup_required": backup_required,
        "total_curtailment_mwh": round(curtailment_mwh, 2),
        "worst_deficit_mw": round(min((r.get("net_balance_mw", 0) for r in results), default=0), 2),
        "peak_surplus_mw": round(max((r.get("net_balance_mw", 0) for r in results), default=0), 2),
    }


# ---------------------------------------------------------------------------
# /optimisation/plan
# ---------------------------------------------------------------------------

@router.get("/optimisation/plan", tags=["Optimisation"])
def get_optimisation_plan(
    scenario: Optional[str] = Query(None, description="Demo scenario: A, B, or C"),
    horizon_hours: int = Query(24, ge=1, le=72),
):
    """
    Run the grid optimisation engine.

    If scenario is provided, runs against the pre-defined demo scenario.
    Otherwise, runs against live forecast + BESS data.

    Returns per-interval dispatch actions and aggregate summary.

    Data type: RULE-BASED OPTIMISATION RESULT
    """
    if scenario:
        ctx = _build_scenario_context(scenario.upper())
        forecast_df = ctx["forecast_df"]
        bess_df = ctx["bess_df"]
        scenario_name = ctx["name"]
    else:
        forecast_df = _build_live_forecast_df(horizon_hours)
        bess_df = load_bess()
        scenario_name = "Live Forecast"

    opt_df = run_optimisation(forecast_df, bess_df)
    summary = _opt_summary(opt_df, scenario_name)

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "RULE_BASED_OPTIMISATION_RESULT",
        "summary": summary,
        "results": opt_df.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# /optimisation/curtailment
# ---------------------------------------------------------------------------

@router.get("/optimisation/curtailment", tags=["Optimisation"])
def get_curtailment_plan(
    scenario: Optional[str] = Query(None, description="Demo scenario: A, B, or C"),
    horizon_hours: int = Query(24, ge=1, le=72),
):
    """
    Curtailment minimisation analysis.

    Shows how much surplus renewable energy was absorbed (BESS, flex load, export)
    vs curtailed, with a transparent explanation.

    Data type: DERIVED FROM OPTIMISATION RESULTS
    """
    if scenario:
        ctx = _build_scenario_context(scenario.upper())
        forecast_df = ctx["forecast_df"]
        bess_df = ctx["bess_df"]
    else:
        forecast_df = _build_live_forecast_df(horizon_hours)
        bess_df = load_bess()

    opt_df = run_optimisation(forecast_df, bess_df)
    curtailment = compute_curtailment_analysis(opt_df)

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "data_type": "DERIVED_FROM_OPTIMISATION",
        **curtailment,
    }


# ---------------------------------------------------------------------------
# /brief
# ---------------------------------------------------------------------------

@router.get("/brief", tags=["Operator"])
def get_operator_brief(
    scenario: Optional[str] = Query(None, description="Demo scenario: A, B, or C"),
):
    """
    Generate a full structured operator brief.

    Aggregates all GridWise backend services into one coherent brief.

    Data types: MEASURED DATA | MODEL PREDICTIONS | RULE/OPTIMISATION RESULTS | AI EXPLANATIONS

    ⚠️ All data is SYNTHETIC/DEMO — not real telemetry.
    """
    now = datetime.now(tz=timezone.utc)

    if scenario:
        ctx = _build_scenario_context(scenario.upper())
        forecast_df = ctx["forecast_df"]
        bess_df = ctx["bess_df"]
        assets_df = ctx["assets_df"]
        weather_scenario = ctx["weather"]
    else:
        model, _ = get_or_train_model()
        grid_df = load_grid()
        weather_df = load_weather()
        assets_df = load_assets().sort_values("timestamp").groupby("asset_id").last().reset_index()
        forecast_df = _build_live_forecast_df(24)
        bess_df = load_bess()
        weather_scenario = None

    # --- Load forecast dict ---
    load_fc_dict = {
        "horizon_hours": len(forecast_df),
        "forecast": [
            {"timestamp": str(r["timestamp"]), "forecast_load_mw": r["demand_mw"]}
            for _, r in forecast_df.iterrows()
        ]
    }

    # --- Spike risks ---
    fc_for_risk = forecast_df.rename(columns={"demand_mw": "forecast_load_mw"})
    risk_df = detect_spikes(fc_for_risk, forecast_df)
    risk_dict = {
        "risk_summary": risk_df.to_dict(orient="records"),
        "thresholds": {
            "critical_reserve_margin": 0.05,
            "high_reserve_margin": 0.10,
            "medium_reserve_margin": 0.20,
        }
    }

    # --- Renewable forecast dict (summary) ---
    total_ren = forecast_df["renewable_mw"].sum() if "renewable_mw" in forecast_df.columns else 0
    ren_dict = {
        "forecasts": [
            {"asset_type": "combined", "asset_id": "ALL",
             "expected_generation_mw": float(total_ren), "capacity_mw": 425.0}
        ]
    }

    # --- Anomalies ---
    anomaly_df = detect_anomalies(assets_df)
    a_summary = anomaly_summary(anomaly_df)
    anomaly_dict = {
        "anomalies": anomaly_df.to_dict(orient="records"),
        **a_summary,
    }

    # --- Diagnoses for underperforming assets ---
    warn_crit = anomaly_df[anomaly_df["severity"].isin(["WARNING", "CRITICAL"])]
    diagnoses = []
    if not warn_crit.empty:
        weather_for_rca = {}
        if weather_scenario:
            weather_for_rca = weather_scenario
        else:
            wdf = load_weather()
            latest_w = wdf.sort_values("timestamp").iloc[-1]
            weather_for_rca = latest_w.to_dict()

        for _, row in warn_crit.drop_duplicates("asset_id").iterrows():
            d = diagnose_asset(
                row["asset_id"], row["asset_name"], row["asset_type"],
                float(row["actual_generation_mw"]),
                float(row["expected_generation_mw"]),
                float(row["capacity_mw"]),
                weather_for_rca,
            )
            diagnoses.append(d)

    # --- Grid status (from latest grid data) ---
    loads_list = [r["demand_mw"] for _, r in forecast_df.iterrows()]
    peak_load_val = max(loads_list) if loads_list else 0
    grid_status_dict = {
        "timestamp": str(forecast_df["timestamp"].iloc[0]) if not forecast_df.empty else str(now),
        "grid_load_mw": peak_load_val,
        "available_capacity_mw": float(forecast_df["available_capacity_mw"].iloc[0]) if "available_capacity_mw" in forecast_df.columns else 2600.0,
        "conventional_generation_mw": float(forecast_df["conventional_mw"].iloc[0]) if "conventional_mw" in forecast_df.columns else 900.0,
        "reserve_margin_pct": (2600 - peak_load_val) / 2600 * 100,
        "overall_risk": risk_df["risk_level"].value_counts().idxmax() if not risk_df.empty else "LOW",
    }

    # --- Optimisation ---
    opt_df = run_optimisation(forecast_df, bess_df)
    summary = _opt_summary(opt_df, scenario or "Live")
    opt_dict = {"results": opt_df.to_dict(orient="records"), "summary": summary}

    # --- Curtailment ---
    curtailment_dict = compute_curtailment_analysis(opt_df)

    brief = generate_operator_brief(
        grid_status=grid_status_dict,
        load_forecast=load_fc_dict,
        spike_risks=risk_dict,
        renewable_forecast=ren_dict,
        anomalies=anomaly_dict,
        diagnoses=diagnoses,
        optimisation_plan=opt_dict,
        curtailment_plan=curtailment_dict,
        generated_at=now,
    )

    return brief


# ---------------------------------------------------------------------------
# /scenarios
# ---------------------------------------------------------------------------

@router.get("/scenarios", tags=["Optimisation"])
def list_scenarios():
    """List available demo scenarios."""
    return {
        "scenarios": [
            {"id": "A", "name": "Normal Grid Operation",
             "description": "Moderate demand, adequate capacity, all assets performing normally."},
            {"id": "B", "name": "Demand Spike + Generation Deficit",
             "description": "Evening demand surge to 2,200 MW with low renewable output. BESS partially depleted."},
            {"id": "C", "name": "Renewable Surplus + Asset Underperformance",
             "description": "High renewable generation with BESS nearly full and one solar asset severely underperforming."},
        ]
    }


# ---------------------------------------------------------------------------
# /demo  — Hackathon demo scenario (spike + anomaly combined)
# ---------------------------------------------------------------------------

@router.get("/demo", tags=["Optimisation"])
def get_demo_scenario():
    """
    Run the complete hackathon demo scenario.

    Scenario D: Approaching evening demand spike + SOL-02 inverter fault.
    Returns the full pipeline result: spike risks, anomaly, RCA, optimisation,
    curtailment, and operator brief in one response.

    Data type: REPRODUCIBLE SYNTHETIC DEMO — all values are deterministic.
    """
    from src.data.demo_scenario import run_demo_pipeline
    return run_demo_pipeline()

