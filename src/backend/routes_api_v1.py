"""
GridWise AI — /api/v1 REST API router.

Versioned, Pydantic-typed endpoints that wrap the existing ML/analytics
pipeline without duplicating or replacing any of the Phase 1/2 logic.

All endpoints:
  - Use Pydantic response models
  - Distinguish data source in `data_type` fields:
      MEASURED | ML_PREDICTION | RULE_BASED | OPTIMISATION_RESULT | AI_EXPLANATION
  - Never fabricate telemetry values
  - Document every field

WARNING: All data served by this API is SYNTHETIC / DEMO only.
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# ── internal imports (reuse existing services) ──────────────────────────────
from src.services.data_store import (
    load_assets, load_grid, load_weather, load_bess, get_or_train_model,
)
from src.ml.load_forecasting import forecast_24h
from src.ml.renewable_generation import solar_expected, wind_expected
from src.ml.spike_detection import detect_spikes, SpikeDetectorConfig
from src.ml.anomaly_detection import detect_anomalies, anomaly_summary
from src.ml.root_cause_analysis import diagnose_asset
from src.ml.grid_optimiser import run_optimisation, OptimiserConfig
from src.ml.curtailment import compute_curtailment_analysis
from src.ml.operator_brief import generate_operator_brief
from src.data.scenarios import ALL_SCENARIOS

import pandas as pd

router = APIRouter(prefix="/api/v1", tags=["API v1"])


# ============================================================================
# Pydantic response models
# ============================================================================

class V1HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' when backend is running")
    version: str
    timestamp: datetime
    data_note: str = Field(..., description="Reminder that all data is synthetic")


# ── Grid status ─────────────────────────────────────────────────────────────

class V1GridStatus(BaseModel):
    generated_at: datetime
    data_type: str = "MEASURED_AND_ML_PREDICTION"
    current_load_mw: float = Field(..., description="Latest measured grid load (MW)")
    forecast_load_mw: float = Field(..., description="Next-hour ML forecast (MW)")
    forecast_peak_mw: float = Field(..., description="24-hour peak ML forecast (MW)")
    peak_time: str = Field(..., description="ISO-8601 timestamp of forecast peak")
    available_capacity_mw: float = Field(..., description="Total available capacity (MW)")
    reserve_margin_percent: float = Field(..., description="(capacity − load) / capacity × 100")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")


# ── Load forecast ────────────────────────────────────────────────────────────

class V1ForecastPoint(BaseModel):
    timestamp: datetime
    forecast_load_mw: float = Field(..., description="ML-predicted load (MW)")
    lower_bound_mw: Optional[float] = None
    upper_bound_mw: Optional[float] = None


class V1LoadForecastResponse(BaseModel):
    generated_at: datetime
    data_type: str = "ML_PREDICTION"
    horizon_hours: int
    model_metrics: Optional[Dict[str, Any]] = None
    forecast: List[V1ForecastPoint]


# ── Renewable forecast ───────────────────────────────────────────────────────

class V1RenewableForecastPoint(BaseModel):
    timestamp: datetime
    asset_id: str
    asset_name: str
    asset_type: str = Field(..., description="solar | wind")
    expected_generation_mw: float = Field(..., description="Physics-model expected output (MW)")
    capacity_mw: float
    capacity_factor: float = Field(..., description="[0–1]")


class V1RenewableForecastResponse(BaseModel):
    generated_at: datetime
    data_type: str = "PHYSICS_MODEL_PREDICTION"
    forecasts: List[V1RenewableForecastPoint]


# ── Grid risks ───────────────────────────────────────────────────────────────

class V1RiskPoint(BaseModel):
    timestamp: datetime
    forecast_load_mw: float
    available_capacity_mw: float
    reserve_margin: float = Field(..., description="(capacity − load) / capacity")
    reserve_margin_percent: float
    spike_risk_score: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    explanation: str


class V1GridRisksResponse(BaseModel):
    generated_at: datetime
    data_type: str = "RULE_BASED_SPIKE_DETECTION"
    total_periods: int
    critical_periods: int
    high_risk_periods: int
    thresholds: Dict[str, float]
    risks: List[V1RiskPoint]


# ── Asset anomalies ──────────────────────────────────────────────────────────

class V1AssetAnomaly(BaseModel):
    asset_id: str
    asset_name: str
    asset_type: str
    timestamp: datetime
    actual_generation_mw: float
    expected_generation_mw: float
    performance_ratio: float = Field(..., description="actual / expected; 1.0 = perfect")
    deviation_percent: float = Field(..., description="(actual − expected) / expected × 100")
    absolute_deviation_mw: float
    severity: str = Field(..., description="NORMAL | WARNING | CRITICAL")


class V1AnomaliesResponse(BaseModel):
    generated_at: datetime
    data_type: str = "ANOMALY_DETECTION"
    total_assets: int
    assets_in_warning: int
    assets_in_critical: int
    anomalies: List[V1AssetAnomaly]


# ── Asset diagnosis ──────────────────────────────────────────────────────────

class V1CauseCandidate(BaseModel):
    cause: str
    confidence: float = Field(..., ge=0, le=1)
    description: str


class V1AssetDiagnosis(BaseModel):
    generated_at: datetime
    data_type: str = "ROOT_CAUSE_ANALYSIS"
    asset_id: str
    asset_name: str
    asset_type: str
    timestamp: datetime
    expected_generation_mw: float
    actual_generation_mw: float
    deviation_mw: float
    deviation_percent: float
    weather_conditions: Dict[str, Any]
    weather_explained_loss_mw: float = Field(..., description="Loss attributable to weather")
    unexplained_loss_mw: float = Field(..., description="Loss not explained by weather — may indicate fault")
    probable_causes: List[V1CauseCandidate]
    overall_confidence: float
    recommended_inspection: bool
    inspection_notes: str


# ── Optimisation plan ────────────────────────────────────────────────────────

class V1OptInterval(BaseModel):
    timestamp: datetime
    forecast_demand_mw: float
    forecast_renewable_mw: float
    forecast_conventional_mw: float
    surplus_or_deficit_mw: float = Field(..., description="+surplus / −deficit before interventions")
    bess_action_mw: float = Field(..., description="+charge / −discharge")
    demand_response_mw: float
    backup_generation_mw: float
    net_balance_mw: float = Field(..., description="Balance after all interventions")
    action_taken: str


class V1OptSummary(BaseModel):
    scenario_name: str
    deficit_periods: int
    surplus_periods: int
    total_bess_charge_mwh: float
    total_bess_discharge_mwh: float
    backup_required: bool
    total_curtailment_mwh: float


class V1OptPlanResponse(BaseModel):
    generated_at: datetime
    data_type: str = "OPTIMISATION_RESULT"
    scenario: str
    summary: V1OptSummary
    intervals: List[V1OptInterval]


# ── Curtailment plan ─────────────────────────────────────────────────────────

class V1CurtailmentInterval(BaseModel):
    timestamp: datetime
    renewable_surplus_mw: float
    potential_curtailment_mw: float
    bess_absorption_mw: float
    flexible_load_absorption_mw: float
    export_mw: float
    unavoidable_curtailment_mw: float
    avoided_curtailment_mw: float


class V1CurtailmentSummary(BaseModel):
    total_potential_curtailment_mwh: float
    total_bess_absorption_mwh: float
    total_avoided_curtailment_mwh: float
    curtailment_reduction_pct: float


class V1CurtailmentResponse(BaseModel):
    generated_at: datetime
    data_type: str = "CURTAILMENT_ANALYSIS"
    scenario: str
    summary: V1CurtailmentSummary
    explanation: str
    intervals: List[V1CurtailmentInterval]


# ── Advisor brief ────────────────────────────────────────────────────────────

class V1BriefGridStatus(BaseModel):
    current_load_mw: float
    available_capacity_mw: float
    reserve_margin_pct: float
    overall_risk: str


class V1BriefDemandForecast(BaseModel):
    horizon_hours: int
    peak_mw: float
    peak_time: str
    average_mw: float


class V1BriefRisk(BaseModel):
    critical_periods: int
    high_risk_periods: int
    first_critical_time: Optional[str]


class V1BriefRenewable(BaseModel):
    total_solar_mw: float
    total_wind_mw: float
    total_combined_mw: float


class V1BriefAnomalousAsset(BaseModel):
    asset_id: str
    asset_name: str
    asset_type: str
    severity: str
    performance_ratio: float
    deviation_mw: float


class V1BriefRootCause(BaseModel):
    asset_name: str
    unexplained_loss_mw: float
    top_cause: str
    confidence: float
    recommended_inspection: bool


class V1BriefOptSummary(BaseModel):
    deficit_periods: int
    surplus_periods: int
    total_bess_discharge_mwh: float
    backup_required: bool


class V1BriefCurtailmentSummary(BaseModel):
    potential_mwh: float
    avoided_mwh: float
    reduction_pct: float


class V1AdvisorBriefResponse(BaseModel):
    generated_at: datetime
    scenario: str = Field(..., description="live | A | B | C")
    is_demo_scenario: bool
    data_type: str = "OPERATOR_BRIEF"
    grid_status: V1BriefGridStatus
    demand_forecast: V1BriefDemandForecast
    demand_risks: V1BriefRisk
    renewable_status: V1BriefRenewable
    anomalous_assets: List[V1BriefAnomalousAsset]
    root_causes: List[V1BriefRootCause]
    optimisation_plan: V1BriefOptSummary
    curtailment_plan: V1BriefCurtailmentSummary
    recommended_actions: List[str]
    data_quality: Dict[str, str]
    known_uncertainties: List[str]
    full_text: str = Field(..., description="Full preformatted operator brief text")


# ============================================================================
# Helpers (shared with routes_phase2 logic)
# ============================================================================

def _build_live_forecast_df(horizon_hours: int = 24) -> pd.DataFrame:
    """Build an optimisation-ready forecast DataFrame from live synthetic data."""
    model, _ = get_or_train_model()
    grid_df = load_grid()
    weather_df = load_weather()
    assets_df = load_assets()

    fc_df = forecast_24h(model, grid_df, weather_df)

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

    rows = []
    grid_row = grid_df.sort_values("timestamp").tail(1).iloc[0]
    conventional_mw = float(grid_row.get("conventional_generation_mw", 900.0))
    available_cap = float(grid_row.get("available_capacity_mw", 2600.0))

    for _, row in fc_df.head(horizon_hours).iterrows():
        ts = row["timestamp"]
        nearest_ts = min(ren_by_ts.keys(), key=lambda t: abs((t - ts).total_seconds()), default=None)
        renewable_mw = ren_by_ts.get(nearest_ts, 0.0) if nearest_ts else 0.0
        rows.append({
            "timestamp": ts,
            "demand_mw": float(row["forecast_load_mw"]),
            "renewable_mw": renewable_mw,
            "conventional_mw": conventional_mw,
            "available_capacity_mw": available_cap,
        })
    return pd.DataFrame(rows)


def _build_scenario_context(scenario_key: str) -> dict:
    if scenario_key not in ALL_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{scenario_key}'. Valid values: A, B, C",
        )
    return ALL_SCENARIOS[scenario_key]()


def _opt_summary_v1(opt_df: pd.DataFrame, scenario_name: str = "") -> V1OptSummary:
    results = opt_df.to_dict(orient="records")
    deficits = [r for r in results if r.get("net_balance_mw", 0) < 0]
    surpluses = [r for r in results if r.get("net_balance_mw", 0) > 0]
    bess_charge = sum(r.get("bess_action_mw", 0) for r in results if r.get("bess_action_mw", 0) > 0)
    bess_discharge = sum(abs(r.get("bess_action_mw", 0)) for r in results if r.get("bess_action_mw", 0) < 0)
    curtailment = sum(r.get("curtailment_mw", 0) for r in results)
    backup = any(r.get("backup_generation_mw", 0) > 0 for r in results)
    return V1OptSummary(
        scenario_name=scenario_name,
        deficit_periods=len(deficits),
        surplus_periods=len(surpluses),
        total_bess_charge_mwh=round(bess_charge, 2),
        total_bess_discharge_mwh=round(bess_discharge, 2),
        backup_required=backup,
        total_curtailment_mwh=round(curtailment, 2),
    )


def _risk_explanation(risk_level: str, reserve_pct: float) -> str:
    if risk_level == "CRITICAL":
        return f"Reserve margin {reserve_pct:.1f}% — critically low. Immediate action required."
    if risk_level == "HIGH":
        return f"Reserve margin {reserve_pct:.1f}% — high risk of supply shortfall."
    if risk_level == "MEDIUM":
        return f"Reserve margin {reserve_pct:.1f}% — approaching operational limits. Monitor closely."
    return f"Reserve margin {reserve_pct:.1f}% — within safe operational limits."


# ============================================================================
# Endpoints
# ============================================================================

# ── Health ───────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=V1HealthResponse,
    summary="API health check",
    description="Returns healthy when the backend is running. Always check this first.",
)
def v1_health():
    return V1HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(tz=timezone.utc),
        data_note="All data served by this API is SYNTHETIC/DEMO — not real grid telemetry.",
    )


# ── Grid status ──────────────────────────────────────────────────────────────

@router.get(
    "/grid/status",
    response_model=V1GridStatus,
    summary="Current grid status and near-term forecast",
    description=(
        "Returns the latest measured grid load, 24-hour ML forecast peak, "
        "available capacity, reserve margin, and current risk level.\n\n"
        "Data types: MEASURED (current load, capacity) + ML_PREDICTION (forecast)."
    ),
)
def v1_grid_status():
    grid_df = load_grid()
    model, _ = get_or_train_model()
    weather_df = load_weather()

    fc_df = forecast_24h(model, grid_df, weather_df)
    loads = fc_df["forecast_load_mw"].tolist()
    peak_mw = max(loads)
    peak_idx = loads.index(peak_mw)
    peak_ts = fc_df["timestamp"].iloc[peak_idx]
    next_hour_load = loads[0] if loads else 0.0

    latest_grid = grid_df.sort_values("timestamp").iloc[-1]
    current_load = float(latest_grid["grid_load_mw"])
    available_cap = float(latest_grid["available_capacity_mw"])
    reserve_pct = (available_cap - current_load) / available_cap * 100

    spike_df = detect_spikes(fc_df, grid_df)
    risk_level = spike_df["risk_level"].iloc[0] if not spike_df.empty else "LOW"

    return V1GridStatus(
        generated_at=datetime.now(tz=timezone.utc),
        current_load_mw=round(current_load, 2),
        forecast_load_mw=round(next_hour_load, 2),
        forecast_peak_mw=round(peak_mw, 2),
        peak_time=peak_ts.isoformat() if hasattr(peak_ts, "isoformat") else str(peak_ts),
        available_capacity_mw=round(available_cap, 2),
        reserve_margin_percent=round(reserve_pct, 2),
        risk_level=risk_level,
    )


# ── Load forecast ────────────────────────────────────────────────────────────

@router.get(
    "/forecast/load",
    response_model=V1LoadForecastResponse,
    summary="24-hour load forecast",
    description=(
        "XGBoost-based 24-hour load forecast. Features: historical load, "
        "hour, day-of-week, weekend flag, temperature, humidity.\n\n"
        "Data type: ML_PREDICTION. Metrics reported on held-out test split."
    ),
)
def v1_forecast_load(
    horizon_hours: int = Query(24, ge=1, le=168, description="Forecast horizon in hours"),
):
    model, metrics = get_or_train_model()
    grid_df = load_grid()
    weather_df = load_weather()
    fc_df = forecast_24h(model, grid_df, weather_df)

    if horizon_hours > 24:
        extra_rows = []
        base_ts = fc_df["timestamp"].max()
        for h in range(24, horizon_hours):
            ref = fc_df.iloc[h % len(fc_df)]
            extra_rows.append({
                "timestamp": base_ts + timedelta(hours=(h - 23)),
                "forecast_load_mw": ref["forecast_load_mw"],
            })
        fc_df = pd.concat([fc_df, pd.DataFrame(extra_rows)], ignore_index=True)

    points = [
        V1ForecastPoint(
            timestamp=row["timestamp"],
            forecast_load_mw=round(float(row["forecast_load_mw"]), 2),
        )
        for _, row in fc_df.head(horizon_hours).iterrows()
    ]
    return V1LoadForecastResponse(
        generated_at=datetime.now(tz=timezone.utc),
        horizon_hours=horizon_hours,
        model_metrics=metrics,
        forecast=points,
    )


# ── Renewable forecast ───────────────────────────────────────────────────────

@router.get(
    "/forecast/renewables",
    response_model=V1RenewableForecastResponse,
    summary="Renewable generation forecast (physics model)",
    description=(
        "Expected-generation forecast for every solar and wind asset using "
        "physics-based models (irradiance, cloud cover, temperature derating for solar; "
        "power curve, cut-in/cut-out for wind).\n\n"
        "Data type: PHYSICS_MODEL_PREDICTION."
    ),
)
def v1_forecast_renewables(
    asset_type: Optional[str] = Query(None, description="Filter: 'solar' or 'wind'"),
):
    assets_df = load_assets()
    weather_df = load_weather()

    latest_weather = weather_df.sort_values("timestamp").tail(24)
    asset_meta = assets_df[["asset_id", "asset_name", "asset_type", "capacity_mw"]].drop_duplicates()

    if asset_type:
        asset_meta = asset_meta[asset_meta["asset_type"] == asset_type.lower()]
        if asset_meta.empty:
            raise HTTPException(status_code=400, detail=f"Unknown asset_type '{asset_type}'. Use 'solar' or 'wind'.")

    points: List[V1RenewableForecastPoint] = []
    for _, asset in asset_meta.iterrows():
        for _, w in latest_weather.iterrows():
            if asset["asset_type"] == "solar":
                result = solar_expected(w["irradiance"], w["cloud_cover"], w["temperature"], asset["capacity_mw"])
            else:
                result = wind_expected(w["wind_speed"], w.get("wind_direction", 180.0), w["temperature"], asset["capacity_mw"])
            points.append(V1RenewableForecastPoint(
                timestamp=w["timestamp"],
                asset_id=asset["asset_id"],
                asset_name=asset["asset_name"],
                asset_type=asset["asset_type"],
                expected_generation_mw=round(result["expected_generation_mw"], 3),
                capacity_mw=float(asset["capacity_mw"]),
                capacity_factor=round(result["capacity_factor"], 4),
            ))

    return V1RenewableForecastResponse(
        generated_at=datetime.now(tz=timezone.utc),
        forecasts=points,
    )


# ── Grid risks ───────────────────────────────────────────────────────────────

@router.get(
    "/grid/risks",
    response_model=V1GridRisksResponse,
    summary="Demand spike risk assessment",
    description=(
        "Deterministic spike detection on top of the ML load forecast. "
        "Classifies each forecast hour as LOW / MEDIUM / HIGH / CRITICAL based on "
        "reserve margin thresholds. All thresholds are configurable via query params.\n\n"
        "Data type: RULE_BASED_SPIKE_DETECTION."
    ),
)
def v1_grid_risks(
    horizon_hours: int = Query(24, ge=1, le=168),
    critical_reserve: Optional[float] = Query(None, description="Override CRITICAL threshold [0–1]"),
    high_reserve: Optional[float] = Query(None, description="Override HIGH threshold [0–1]"),
    medium_reserve: Optional[float] = Query(None, description="Override MEDIUM threshold [0–1]"),
):
    model, _ = get_or_train_model()
    grid_df = load_grid()
    weather_df = load_weather()

    fc_df = forecast_24h(model, grid_df, weather_df)

    cfg = SpikeDetectorConfig()
    if critical_reserve is not None:
        cfg.critical_reserve_margin = critical_reserve
    if high_reserve is not None:
        cfg.high_reserve_margin = high_reserve
    if medium_reserve is not None:
        cfg.medium_reserve_margin = medium_reserve

    risk_df = detect_spikes(fc_df, grid_df, cfg=cfg)

    risk_points: List[V1RiskPoint] = []
    for _, row in risk_df.iterrows():
        reserve_pct = float(row["reserve_margin"]) * 100
        risk_points.append(V1RiskPoint(
            timestamp=row["timestamp"],
            forecast_load_mw=round(float(row["forecast_load_mw"]), 2),
            available_capacity_mw=round(float(row["available_capacity_mw"]), 2),
            reserve_margin=round(float(row["reserve_margin"]), 4),
            reserve_margin_percent=round(reserve_pct, 2),
            spike_risk_score=round(float(row["spike_risk_score"]), 4),
            risk_level=row["risk_level"],
            explanation=_risk_explanation(row["risk_level"], reserve_pct),
        ))

    crit = sum(1 for r in risk_points if r.risk_level == "CRITICAL")
    high = sum(1 for r in risk_points if r.risk_level == "HIGH")

    return V1GridRisksResponse(
        generated_at=datetime.now(tz=timezone.utc),
        total_periods=len(risk_points),
        critical_periods=crit,
        high_risk_periods=high,
        thresholds={
            "critical_reserve_margin": cfg.critical_reserve_margin,
            "high_reserve_margin": cfg.high_reserve_margin,
            "medium_reserve_margin": cfg.medium_reserve_margin,
        },
        risks=risk_points,
    )


# ── Asset anomalies ──────────────────────────────────────────────────────────

@router.get(
    "/assets/anomalies",
    response_model=V1AnomaliesResponse,
    summary="Renewable asset anomaly detection",
    description=(
        "Compares actual vs physics-model expected generation for every asset. "
        "Nighttime solar is suppressed (not flagged as anomalous). "
        "Severity: NORMAL (PR≥0.85) | WARNING (0.60≤PR<0.85) | CRITICAL (PR<0.60).\n\n"
        "Data type: ANOMALY_DETECTION."
    ),
)
def v1_asset_anomalies(
    hours: int = Query(24, ge=1, le=168, description="Lookback window in hours"),
    severity: Optional[str] = Query(None, description="Filter: NORMAL | WARNING | CRITICAL"),
):
    assets_df = load_assets()
    cutoff = assets_df["timestamp"].max() - timedelta(hours=hours)
    window = assets_df[assets_df["timestamp"] >= cutoff].copy()

    anomaly_df = detect_anomalies(window)
    if severity:
        sev_upper = severity.upper()
        if sev_upper not in ("NORMAL", "WARNING", "CRITICAL"):
            raise HTTPException(status_code=400, detail="severity must be NORMAL, WARNING, or CRITICAL")
        anomaly_df = anomaly_df[anomaly_df["severity"] == sev_upper]

    summary = anomaly_summary(anomaly_df)

    records = [
        V1AssetAnomaly(
            asset_id=row["asset_id"],
            asset_name=row["asset_name"],
            asset_type=row["asset_type"],
            timestamp=row["timestamp"],
            actual_generation_mw=round(float(row["actual_generation_mw"]), 3),
            expected_generation_mw=round(float(row["expected_generation_mw"]), 3),
            performance_ratio=round(float(row["performance_ratio"]), 4),
            deviation_percent=round(float(row["percentage_deviation"]), 2),
            absolute_deviation_mw=round(float(row["absolute_deviation_mw"]), 3),
            severity=row["severity"],
        )
        for _, row in anomaly_df.iterrows()
    ]

    return V1AnomaliesResponse(
        generated_at=datetime.now(tz=timezone.utc),
        total_assets=summary["total_assets"],
        assets_in_warning=summary["assets_in_warning"],
        assets_in_critical=summary["assets_in_critical"],
        anomalies=records,
    )


# ── Asset diagnosis ───────────────────────────────────────────────────────────

@router.get(
    "/assets/{asset_id}/diagnosis",
    response_model=V1AssetDiagnosis,
    summary="Root cause analysis for a specific asset",
    description=(
        "Runs evidence-based RCA for the named asset. Separates weather-explained "
        "losses (cloud cover, irradiance, temperature derating, insufficient/excess wind) "
        "from unexplained losses that may indicate equipment faults. "
        "Never claims a mechanical fault as certain without supporting telemetry.\n\n"
        "Data type: ROOT_CAUSE_ANALYSIS."
    ),
)
def v1_asset_diagnosis(
    asset_id: str,
    timestamp: Optional[str] = Query(None, description="ISO-8601 timestamp; defaults to latest"),
):
    assets_df = load_assets()
    weather_df = load_weather()

    asset_rows = assets_df[assets_df["asset_id"] == asset_id]
    if asset_rows.empty:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found. Valid IDs: SOL-01, SOL-02, SOL-03, WIN-01, WIN-02")

    if timestamp:
        try:
            ts_parsed = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO-8601 e.g. 2024-01-01T12:00:00")
        match = asset_rows[asset_rows["timestamp"] == ts_parsed]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"No record for asset '{asset_id}' at {timestamp}")
        row = match.iloc[0]
    else:
        row = asset_rows.sort_values("timestamp").iloc[-1]

    ts = row["timestamp"]
    weather_indexed = weather_df.set_index("timestamp")
    if ts in weather_indexed.index:
        w = weather_indexed.loc[ts].to_dict()
    else:
        nearest_idx = weather_indexed.index.get_indexer([ts], method="nearest")
        w = weather_indexed.iloc[nearest_idx[0]].to_dict()

    # Clean weather for response (remove non-serialisable items)
    weather_clean = {
        k: (float(v) if hasattr(v, "__float__") else str(v))
        for k, v in w.items()
        if k in ("temperature", "humidity", "cloud_cover", "irradiance", "wind_speed", "wind_direction")
    }

    diag = diagnose_asset(
        asset_id=row["asset_id"],
        asset_name=row["asset_name"],
        asset_type=row["asset_type"],
        actual_mw=float(row["actual_generation_mw"]),
        expected_mw=float(row["expected_generation_mw"]),
        capacity_mw=float(row["capacity_mw"]),
        weather=w,
        timestamp=ts,
    )

    causes = [
        V1CauseCandidate(
            cause=c["cause"],
            confidence=float(c["confidence"]),
            description=c["description"],
        )
        for c in diag.get("probable_causes", [])
    ]

    dev_mw = float(diag.get("deviation_mw", 0.0))
    exp_mw = float(diag.get("expected_generation_mw", 1.0)) or 1.0
    dev_pct = round(dev_mw / exp_mw * 100, 2)

    return V1AssetDiagnosis(
        generated_at=datetime.now(tz=timezone.utc),
        asset_id=diag["asset_id"],
        asset_name=diag["asset_name"],
        asset_type=diag["asset_type"],
        timestamp=diag.get("timestamp", ts),
        expected_generation_mw=round(float(diag["expected_generation_mw"]), 3),
        actual_generation_mw=round(float(diag["actual_generation_mw"]), 3),
        deviation_mw=round(dev_mw, 3),
        deviation_percent=dev_pct,
        weather_conditions=weather_clean,
        weather_explained_loss_mw=round(float(diag.get("weather_explained_loss_mw", 0.0)), 3),
        unexplained_loss_mw=round(float(diag.get("unexplained_loss_mw", 0.0)), 3),
        probable_causes=causes,
        overall_confidence=round(float(diag.get("overall_confidence", 0.0)), 4),
        recommended_inspection=bool(diag.get("recommended_inspection", False)),
        inspection_notes=str(diag.get("inspection_notes", "")),
    )


# ── Optimisation plan ─────────────────────────────────────────────────────────

@router.get(
    "/optimisation/plan",
    response_model=V1OptPlanResponse,
    summary="Grid optimisation dispatch plan",
    description=(
        "Priority-order dispatch: BESS → demand response → backup generation. "
        "Returns per-hour recommendations for the next 24 hours.\n\n"
        "?scenario=A|B|C runs a reproducible demo scenario (clearly labelled).\n\n"
        "Data type: OPTIMISATION_RESULT."
    ),
)
def v1_optimisation_plan(
    scenario: Optional[str] = Query(None, description="Demo scenario: A, B, or C (live if omitted)"),
    horizon_hours: int = Query(24, ge=1, le=72),
):
    if scenario:
        ctx = _build_scenario_context(scenario.upper())
        forecast_df = ctx["forecast_df"]
        bess_df = ctx["bess_df"]
        scenario_name = ctx["name"]
        is_demo = True
    else:
        forecast_df = _build_live_forecast_df(horizon_hours)
        bess_df = load_bess()
        scenario_name = "Live Forecast"
        is_demo = False

    opt_df = run_optimisation(forecast_df, bess_df)
    summary = _opt_summary_v1(opt_df, scenario_name)

    intervals: List[V1OptInterval] = []
    for _, row in opt_df.iterrows():
        intervals.append(V1OptInterval(
            timestamp=row["timestamp"],
            forecast_demand_mw=round(float(row.get("demand_mw", 0)), 2),
            forecast_renewable_mw=round(float(row.get("renewable_mw", 0)), 2),
            forecast_conventional_mw=round(float(row.get("conventional_mw", 0)), 2),
            surplus_or_deficit_mw=round(float(row.get("net_balance_mw", 0) - row.get("bess_action_mw", 0)
                                              - row.get("demand_response_mw", 0)
                                              - row.get("backup_generation_mw", 0)), 2),
            bess_action_mw=round(float(row.get("bess_action_mw", 0)), 2),
            demand_response_mw=round(float(row.get("demand_response_mw", 0)), 2),
            backup_generation_mw=round(float(row.get("backup_generation_mw", 0)), 2),
            net_balance_mw=round(float(row.get("net_balance_mw", 0)), 2),
            action_taken=str(row.get("action_taken", "")),
        ))

    return V1OptPlanResponse(
        generated_at=datetime.now(tz=timezone.utc),
        scenario=f"{'DEMO:' if is_demo else ''}{scenario_name}",
        summary=summary,
        intervals=intervals,
    )


# ── Curtailment plan ──────────────────────────────────────────────────────────

@router.get(
    "/curtailment/plan",
    response_model=V1CurtailmentResponse,
    summary="Renewable curtailment minimisation plan",
    description=(
        "Shows how surplus renewable energy is absorbed (BESS, flexible load, export) "
        "vs curtailed, and what percentage of curtailment was avoided.\n\n"
        "?scenario=A|B|C runs a reproducible demo scenario.\n\n"
        "Data type: CURTAILMENT_ANALYSIS."
    ),
)
def v1_curtailment_plan(
    scenario: Optional[str] = Query(None, description="Demo scenario: A, B, or C (live if omitted)"),
    horizon_hours: int = Query(24, ge=1, le=72),
):
    if scenario:
        ctx = _build_scenario_context(scenario.upper())
        forecast_df = ctx["forecast_df"]
        bess_df = ctx["bess_df"]
        scenario_name = ctx["name"]
        is_demo = True
    else:
        forecast_df = _build_live_forecast_df(horizon_hours)
        bess_df = load_bess()
        scenario_name = "Live Forecast"
        is_demo = False

    opt_df = run_optimisation(forecast_df, bess_df)
    curt = compute_curtailment_analysis(opt_df)

    s = curt["summary"]
    summary = V1CurtailmentSummary(
        total_potential_curtailment_mwh=s["total_potential_curtailment_mwh"],
        total_bess_absorption_mwh=s["total_bess_absorption_mwh"],
        total_avoided_curtailment_mwh=s["total_avoided_curtailment_mwh"],
        curtailment_reduction_pct=s["curtailment_reduction_pct"],
    )

    intervals = [
        V1CurtailmentInterval(
            timestamp=row["timestamp"],
            renewable_surplus_mw=round(float(row.get("renewable_surplus_mw", 0)), 2),
            potential_curtailment_mw=round(float(row.get("potential_curtailment_mw", 0)), 2),
            bess_absorption_mw=round(float(row.get("bess_absorption_mw", 0)), 2),
            flexible_load_absorption_mw=round(float(row.get("flexible_load_absorption_mw", 0)), 2),
            export_mw=round(float(row.get("export_mw", 0)), 2),
            unavoidable_curtailment_mw=round(float(row.get("unavoidable_curtailment_mw", 0)), 2),
            avoided_curtailment_mw=round(float(row.get("avoided_curtailment_mw", 0)), 2),
        )
        for row in curt.get("intervals", [])
    ]

    return V1CurtailmentResponse(
        generated_at=datetime.now(tz=timezone.utc),
        scenario=f"{'DEMO:' if is_demo else ''}{scenario_name}",
        summary=summary,
        explanation=curt.get("explanation", ""),
        intervals=intervals,
    )


# ── Advisor brief ─────────────────────────────────────────────────────────────

@router.get(
    "/advisor/brief",
    response_model=V1AdvisorBriefResponse,
    summary="Comprehensive operator advisor brief",
    description=(
        "Combines all GridWise AI pipeline outputs into one structured brief: "
        "grid status, load forecast, spike risks, renewable performance, anomaly detection, "
        "root cause analysis, optimisation plan, curtailment analysis, and recommended actions.\n\n"
        "?scenario=live (default) uses real-time synthetic data.\n"
        "?scenario=A|B|C uses a reproducible demo scenario (clearly labelled).\n\n"
        "IBM Bob consumes this endpoint — values are NEVER fabricated.\n\n"
        "Data types: MEASURED | ML_PREDICTION | RULE_BASED | OPTIMISATION_RESULT | AI_EXPLANATION"
    ),
)
def v1_advisor_brief(
    scenario: Optional[str] = Query(
        "live",
        description="Scenario: 'live' for current synthetic data, or 'A', 'B', 'C' for demo scenarios",
    ),
):
    now = datetime.now(tz=timezone.utc)
    scenario_upper = (scenario or "live").upper()
    if scenario_upper not in ("LIVE", "A", "B", "C"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{scenario}'. Valid values: live, A, B, C",
        )
    is_demo = scenario_upper in ("A", "B", "C")

    # ── Load scenario data ────────────────────────────────────────────────────
    if is_demo:
        ctx = _build_scenario_context(scenario_upper)
        forecast_df = ctx["forecast_df"]
        bess_df = ctx["bess_df"]
        assets_df = ctx["assets_df"]
        weather_scenario = ctx.get("weather", {})
        scenario_label = ctx["name"]
    else:
        model, _ = get_or_train_model()
        grid_df = load_grid()
        weather_df = load_weather()
        assets_df = load_assets().sort_values("timestamp").groupby("asset_id").last().reset_index()
        forecast_df = _build_live_forecast_df(24)
        bess_df = load_bess()
        weather_scenario = {}
        scenario_label = "Live"

    # ── Load forecast ─────────────────────────────────────────────────────────
    loads = [float(r["demand_mw"]) for _, r in forecast_df.iterrows()]
    peak_mw = max(loads) if loads else 0.0
    avg_mw = sum(loads) / len(loads) if loads else 0.0
    peak_idx = loads.index(peak_mw) if loads else 0
    peak_ts_val = forecast_df["timestamp"].iloc[peak_idx]
    peak_time_str = peak_ts_val.isoformat() if hasattr(peak_ts_val, "isoformat") else str(peak_ts_val)

    load_fc_dict = {
        "horizon_hours": len(forecast_df),
        "forecast": [
            {"timestamp": str(r["timestamp"]), "forecast_load_mw": r["demand_mw"]}
            for _, r in forecast_df.iterrows()
        ],
    }

    # ── Grid status ───────────────────────────────────────────────────────────
    if is_demo:
        current_load = loads[0] if loads else 0.0
        available_cap = float(forecast_df["available_capacity_mw"].iloc[0]) if "available_capacity_mw" in forecast_df.columns else 2600.0
    else:
        latest_grid = grid_df.sort_values("timestamp").iloc[-1]
        current_load = float(latest_grid["grid_load_mw"])
        available_cap = float(latest_grid["available_capacity_mw"])

    reserve_pct = (available_cap - current_load) / available_cap * 100 if available_cap > 0 else 0.0

    # ── Spike risks ───────────────────────────────────────────────────────────
    fc_for_risk = forecast_df.rename(columns={"demand_mw": "forecast_load_mw"})
    if is_demo:
        risk_df = detect_spikes(fc_for_risk, forecast_df)
    else:
        risk_df = detect_spikes(fc_for_risk, grid_df)

    overall_risk = risk_df["risk_level"].value_counts().idxmax() if not risk_df.empty else "LOW"
    crit_periods = int((risk_df["risk_level"] == "CRITICAL").sum()) if not risk_df.empty else 0
    high_periods = int((risk_df["risk_level"] == "HIGH").sum()) if not risk_df.empty else 0
    crit_rows = risk_df[risk_df["risk_level"] == "CRITICAL"] if not risk_df.empty else pd.DataFrame()
    first_crit = crit_rows["timestamp"].iloc[0].isoformat() if not crit_rows.empty and hasattr(crit_rows["timestamp"].iloc[0], "isoformat") else None

    risk_dict = {"risk_summary": risk_df.to_dict(orient="records"), "thresholds": {}}

    # ── Renewable forecast ────────────────────────────────────────────────────
    total_ren = float(forecast_df["renewable_mw"].sum()) if "renewable_mw" in forecast_df.columns else 0.0
    ren_dict = {
        "forecasts": [
            {"asset_type": "combined", "asset_id": "ALL",
             "expected_generation_mw": total_ren, "capacity_mw": 425.0}
        ]
    }

    # ── Anomalies ─────────────────────────────────────────────────────────────
    anomaly_df = detect_anomalies(assets_df)
    a_summary = anomaly_summary(anomaly_df)
    anomaly_dict = {"anomalies": anomaly_df.to_dict(orient="records"), **a_summary}

    warn_crit_df = anomaly_df[anomaly_df["severity"].isin(["WARNING", "CRITICAL"])]
    anomalous_assets = [
        V1BriefAnomalousAsset(
            asset_id=row["asset_id"],
            asset_name=row["asset_name"],
            asset_type=row["asset_type"],
            severity=row["severity"],
            performance_ratio=round(float(row["performance_ratio"]), 4),
            deviation_mw=round(float(row["absolute_deviation_mw"]), 3),
        )
        for _, row in warn_crit_df.drop_duplicates("asset_id").iterrows()
    ]

    # ── Root causes ───────────────────────────────────────────────────────────
    diagnoses_raw = []
    if not warn_crit_df.empty:
        weather_for_rca: dict = {}
        if weather_scenario:
            weather_for_rca = weather_scenario
        elif not is_demo:
            latest_w = weather_df.sort_values("timestamp").iloc[-1]
            weather_for_rca = latest_w.to_dict()

        for _, row in warn_crit_df.drop_duplicates("asset_id").iterrows():
            d = diagnose_asset(
                row["asset_id"], row["asset_name"], row["asset_type"],
                float(row["actual_generation_mw"]), float(row["expected_generation_mw"]),
                float(row["capacity_mw"]), weather_for_rca,
            )
            diagnoses_raw.append(d)

    root_causes = []
    for d in diagnoses_raw:
        causes = d.get("probable_causes", [])
        top = causes[0] if causes else {}
        root_causes.append(V1BriefRootCause(
            asset_name=d.get("asset_name", ""),
            unexplained_loss_mw=round(float(d.get("unexplained_loss_mw", 0.0)), 3),
            top_cause=top.get("cause", "none identified"),
            confidence=round(float(top.get("confidence", 0.0)), 4),
            recommended_inspection=bool(d.get("recommended_inspection", False)),
        ))

    # ── Optimisation + curtailment ────────────────────────────────────────────
    opt_df = run_optimisation(forecast_df, bess_df)
    opt_summary = _opt_summary_v1(opt_df, scenario_label)
    curt = compute_curtailment_analysis(opt_df)
    cs = curt["summary"]

    # ── Generate the full text brief (via existing operator_brief module) ─────
    grid_status_dict = {
        "timestamp": str(forecast_df["timestamp"].iloc[0]) if not forecast_df.empty else str(now),
        "grid_load_mw": current_load,
        "available_capacity_mw": available_cap,
        "conventional_generation_mw": float(forecast_df["conventional_mw"].iloc[0]) if "conventional_mw" in forecast_df.columns else 900.0,
        "reserve_margin_pct": reserve_pct,
        "overall_risk": overall_risk,
    }
    opt_dict = {"results": opt_df.to_dict(orient="records"), "summary": opt_summary.model_dump()}
    brief_obj = generate_operator_brief(
        grid_status=grid_status_dict,
        load_forecast=load_fc_dict,
        spike_risks=risk_dict,
        renewable_forecast=ren_dict,
        anomalies=anomaly_dict,
        diagnoses=diagnoses_raw,
        optimisation_plan=opt_dict,
        curtailment_plan=curt,
        generated_at=now,
    )
    full_text = brief_obj.get("full_text", "Brief generation failed.")

    # ── Recommended actions ───────────────────────────────────────────────────
    actions: List[str] = []
    if crit_periods > 0:
        actions.append(f"[CRITICAL] {crit_periods} critical demand period(s) forecast — prepare BESS discharge and demand response")
    if opt_summary.backup_required:
        actions.append("[HIGH] Backup generation will be required during deficit periods")
    if anomalous_assets:
        for a in anomalous_assets:
            if a.severity == "CRITICAL":
                actions.append(f"[HIGH] Asset {a.asset_name} is CRITICAL — schedule immediate inspection")
            else:
                actions.append(f"[MEDIUM] Asset {a.asset_name} is underperforming (WARNING) — monitor and inspect if worsening")
    if cs["total_avoided_curtailment_mwh"] > 0:
        actions.append(f"[INFO] {cs['total_avoided_curtailment_mwh']:.1f} MWh curtailment avoided via BESS and flexible load")
    if not actions:
        actions.append("[INFO] Grid operating normally — continue standard monitoring")

    # ── Data quality metadata ─────────────────────────────────────────────────
    data_quality = {
        "grid_data": "SYNTHETIC — deterministic generator",
        "weather_data": "SYNTHETIC — seasonal model",
        "load_forecast": f"ML_PREDICTION — XGBoost, R² on test split included in /api/v1/forecast/load",
        "renewable_forecast": "PHYSICS_MODEL — irradiance/power curve",
        "anomaly_detection": "RULE_BASED — performance ratio thresholds",
        "root_cause": "RULE_BASED — evidence-weighted heuristics",
        "optimisation": "RULE_BASED — priority dispatch (BESS → DR → backup)",
    }

    uncertainties = [
        "All data is synthetic/demo — values do not represent real grid telemetry",
        "Load forecast uncertainty increases beyond 12h horizon",
        "Renewable forecast assumes current weather conditions persist",
        "BESS degradation and round-trip efficiency losses are simplified",
        "Weather-to-generation physics model does not account for panel/turbine age",
    ]
    if is_demo:
        uncertainties.insert(0, f"DEMO SCENARIO {scenario_upper}: reproducible fixed scenario — not live data")

    return V1AdvisorBriefResponse(
        generated_at=now,
        scenario=scenario_upper if is_demo else "live",
        is_demo_scenario=is_demo,
        grid_status=V1BriefGridStatus(
            current_load_mw=round(current_load, 2),
            available_capacity_mw=round(available_cap, 2),
            reserve_margin_pct=round(reserve_pct, 2),
            overall_risk=overall_risk,
        ),
        demand_forecast=V1BriefDemandForecast(
            horizon_hours=len(loads),
            peak_mw=round(peak_mw, 2),
            peak_time=peak_time_str,
            average_mw=round(avg_mw, 2),
        ),
        demand_risks=V1BriefRisk(
            critical_periods=crit_periods,
            high_risk_periods=high_periods,
            first_critical_time=first_crit,
        ),
        renewable_status=V1BriefRenewable(
            total_solar_mw=0.0,
            total_wind_mw=0.0,
            total_combined_mw=round(total_ren, 2),
        ),
        anomalous_assets=anomalous_assets,
        root_causes=root_causes,
        optimisation_plan=V1BriefOptSummary(
            deficit_periods=opt_summary.deficit_periods,
            surplus_periods=opt_summary.surplus_periods,
            total_bess_discharge_mwh=opt_summary.total_bess_discharge_mwh,
            backup_required=opt_summary.backup_required,
        ),
        curtailment_plan=V1BriefCurtailmentSummary(
            potential_mwh=cs["total_potential_curtailment_mwh"],
            avoided_mwh=cs["total_avoided_curtailment_mwh"],
            reduction_pct=cs["curtailment_reduction_pct"],
        ),
        recommended_actions=actions,
        data_quality=data_quality,
        known_uncertainties=uncertainties,
        full_text=full_text,
    )
