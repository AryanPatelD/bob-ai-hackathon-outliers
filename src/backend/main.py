"""
GridWise AI — FastAPI application.

Endpoints
---------
GET  /health
GET  /assets
GET  /assets/anomalies
GET  /assets/{asset_id}/diagnosis
GET  /forecast/load
GET  /forecast/renewables
GET  /grid/risk
GET  /docs  (auto-generated OpenAPI)
"""
from __future__ import annotations

import sys
import os

# Ensure the workspace root is on sys.path when running from src/backend/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Internal imports
from src.services.data_store import (
    load_assets, load_grid, load_weather, load_bess, get_or_train_model
)
from src.ml.load_forecasting import forecast_24h
from src.ml.renewable_generation import solar_expected, wind_expected
from src.ml.spike_detection import detect_spikes, summarise_risk, SpikeDetectorConfig
from src.ml.anomaly_detection import detect_anomalies, anomaly_summary, AnomalyConfig
from src.ml.root_cause_analysis import diagnose_asset, diagnose_all, RCAConfig
from src.models.schemas import (
    HealthResponse,
    LoadForecastResponse, LoadForecastPoint,
    RenewableForecastResponse, RenewableForecastPoint,
    GridRiskResponse, GridRiskPoint,
    AnomalyResponse, AssetAnomaly,
    AssetDiagnosis, CauseCandidate,
    AssetType, AnomalySeverity, RiskLevel,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GridWise AI — Grid Load Optimisation & Renewable Energy Advisor",
    description=(
        "Phase 1 + 2 API — Data, ML, Analytics, Optimisation & Operator Advisor.\n\n"
        "WARNING: All data is **synthetic/demo data** and is NOT real grid telemetry."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase 2 routes
from src.backend.routes_phase2 import router as phase2_router  # noqa: E402
app.include_router(phase2_router)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """API health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

@app.get("/assets", tags=["Assets"])
def get_assets(
    asset_type: Optional[str] = Query(None, description="Filter by 'solar' or 'wind'"),
    latest_only: bool = Query(True, description="Return only the most recent record per asset"),
):
    """
    Return all renewable assets with their current generation status.

    ⚠️ Synthetic data.
    """
    df = load_assets()
    if asset_type:
        df = df[df["asset_type"] == asset_type.lower()]
    if latest_only:
        df = df.sort_values("timestamp").groupby("asset_id").last().reset_index()
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

@app.get("/assets/anomalies", response_model=AnomalyResponse, tags=["Assets"])
def get_anomalies(
    hours: int = Query(24, ge=1, le=168, description="Lookback window in hours"),
    severity: Optional[str] = Query(None, description="Filter by NORMAL / WARNING / CRITICAL"),
):
    """
    Detect anomalies across all renewable assets within the lookback window.

    Compares actual vs expected generation.  Does NOT flag poor-weather
    periods as asset failures.

    ⚠️ Synthetic data.
    """
    assets_df = load_assets()
    cutoff = assets_df["timestamp"].max() - timedelta(hours=hours)
    assets_window = assets_df[assets_df["timestamp"] >= cutoff].copy()

    anomaly_df = detect_anomalies(assets_window)

    if severity:
        anomaly_df = anomaly_df[anomaly_df["severity"] == severity.upper()]

    summary = anomaly_summary(anomaly_df)

    records: List[AssetAnomaly] = []
    for _, row in anomaly_df.iterrows():
        records.append(
            AssetAnomaly(
                asset_id=row["asset_id"],
                asset_name=row["asset_name"],
                asset_type=AssetType(row["asset_type"]),
                timestamp=row["timestamp"],
                actual_generation_mw=float(row["actual_generation_mw"]),
                expected_generation_mw=float(row["expected_generation_mw"]),
                absolute_deviation_mw=float(row["absolute_deviation_mw"]),
                percentage_deviation=float(row["percentage_deviation"]),
                performance_ratio=float(row["performance_ratio"]),
                severity=AnomalySeverity(row["severity"]),
            )
        )

    return AnomalyResponse(
        generated_at=datetime.now(tz=timezone.utc),
        anomalies=records,
        total_assets=summary["total_assets"],
        assets_in_warning=summary["assets_in_warning"],
        assets_in_critical=summary["assets_in_critical"],
    )


# ---------------------------------------------------------------------------
# Asset diagnosis (RCA)
# ---------------------------------------------------------------------------

@app.get("/assets/{asset_id}/diagnosis", tags=["Assets"])
def get_asset_diagnosis(
    asset_id: str,
    timestamp: Optional[str] = Query(None, description="ISO-8601 timestamp; defaults to latest"),
):
    """
    Run root cause analysis for a specific asset.

    Separates weather-explained losses from unexplained losses and
    returns probable causes with confidence scores.

    ⚠️ Synthetic data.
    """
    assets_df = load_assets()
    weather_df = load_weather()

    asset_rows = assets_df[assets_df["asset_id"] == asset_id]
    if asset_rows.empty:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")

    if timestamp:
        try:
            ts = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
        row_match = asset_rows[asset_rows["timestamp"] == ts]
        if row_match.empty:
            raise HTTPException(status_code=404, detail=f"No record for asset '{asset_id}' at {timestamp}")
        row = row_match.iloc[0]
    else:
        row = asset_rows.sort_values("timestamp").iloc[-1]

    ts = row["timestamp"]
    # Get nearest weather
    weather_indexed = weather_df.set_index("timestamp")
    if ts in weather_indexed.index:
        w = weather_indexed.loc[ts].to_dict()
    else:
        nearest_idx = weather_indexed.index.get_indexer([ts], method="nearest")
        w = weather_indexed.iloc[nearest_idx[0]].to_dict()

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
    return diag


# ---------------------------------------------------------------------------
# Load forecast
# ---------------------------------------------------------------------------

@app.get("/forecast/load", response_model=LoadForecastResponse, tags=["Forecasting"])
def get_load_forecast(
    horizon_hours: int = Query(24, ge=1, le=168, description="Forecast horizon in hours"),
):
    """
    24-hour (or custom horizon) load forecast using the trained XGBoost model.

    ⚠️ Synthetic data.
    """
    model, metrics = get_or_train_model()
    grid_df = load_grid()
    weather_df = load_weather()

    fc_df = forecast_24h(model, grid_df, weather_df)

    # Extend to requested horizon by repeating the pattern if needed
    if horizon_hours > 24:
        import pandas as pd
        extra_rows = []
        base_ts = fc_df["timestamp"].max()
        for h in range(24, horizon_hours):
            mirror_h = h % 24
            ref = fc_df.iloc[mirror_h % len(fc_df)]
            extra_rows.append({
                "timestamp": base_ts + timedelta(hours=(h - 23)),
                "forecast_load_mw": ref["forecast_load_mw"],
            })
        import pandas as pd
        fc_df = pd.concat([fc_df, pd.DataFrame(extra_rows)], ignore_index=True)

    points = [
        LoadForecastPoint(
            timestamp=row["timestamp"],
            forecast_load_mw=float(row["forecast_load_mw"]),
        )
        for _, row in fc_df.head(horizon_hours).iterrows()
    ]

    return LoadForecastResponse(
        generated_at=datetime.now(tz=timezone.utc),
        horizon_hours=horizon_hours,
        forecast=points,
        model_metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Renewable forecast
# ---------------------------------------------------------------------------

@app.get("/forecast/renewables", response_model=RenewableForecastResponse, tags=["Forecasting"])
def get_renewable_forecast(
    asset_type: Optional[str] = Query(None, description="'solar' or 'wind'"),
):
    """
    Expected generation forecast for all renewable assets based on physics models.

    Uses the latest 24 hours of weather data.

    ⚠️ Synthetic data.
    """
    assets_df = load_assets()
    weather_df = load_weather()

    # Last 24 weather records
    latest_weather = weather_df.sort_values("timestamp").tail(24)
    # Unique assets
    asset_meta = assets_df[["asset_id", "asset_name", "asset_type", "capacity_mw"]].drop_duplicates()

    if asset_type:
        asset_meta = asset_meta[asset_meta["asset_type"] == asset_type.lower()]

    points: List[RenewableForecastPoint] = []
    for _, asset in asset_meta.iterrows():
        for _, w in latest_weather.iterrows():
            if asset["asset_type"] == "solar":
                result = solar_expected(
                    w["irradiance"], w["cloud_cover"],
                    w["temperature"], asset["capacity_mw"]
                )
            else:
                result = wind_expected(
                    w["wind_speed"], w.get("wind_direction", 180.0),
                    w["temperature"], asset["capacity_mw"]
                )
            points.append(
                RenewableForecastPoint(
                    timestamp=w["timestamp"],
                    asset_id=asset["asset_id"],
                    asset_type=AssetType(asset["asset_type"]),
                    expected_generation_mw=result["expected_generation_mw"],
                    capacity_mw=float(asset["capacity_mw"]),
                    capacity_factor=result["capacity_factor"],
                )
            )

    return RenewableForecastResponse(
        generated_at=datetime.now(tz=timezone.utc),
        forecasts=points,
    )


# ---------------------------------------------------------------------------
# Grid risk
# ---------------------------------------------------------------------------

@app.get("/grid/risk", response_model=GridRiskResponse, tags=["Grid"])
def get_grid_risk(
    horizon_hours: int = Query(24, ge=1, le=168),
    critical_reserve: Optional[float] = Query(None, description="Override CRITICAL reserve margin threshold"),
    high_reserve: Optional[float] = Query(None, description="Override HIGH reserve margin threshold"),
    medium_reserve: Optional[float] = Query(None, description="Override MEDIUM reserve margin threshold"),
):
    """
    Forecast demand spike risk for the next N hours.

    Returns risk level (LOW/MEDIUM/HIGH/CRITICAL) and spike risk score
    per hour based on reserve margin.

    ⚠️ Synthetic data.
    """
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

    risk_points = [
        GridRiskPoint(
            timestamp=row["timestamp"],
            forecast_load_mw=float(row["forecast_load_mw"]),
            available_capacity_mw=float(row["available_capacity_mw"]),
            reserve_margin=float(row["reserve_margin"]),
            spike_risk_score=float(row["spike_risk_score"]),
            risk_level=RiskLevel(row["risk_level"]),
        )
        for _, row in risk_df.iterrows()
    ]

    return GridRiskResponse(
        generated_at=datetime.now(tz=timezone.utc),
        risk_summary=risk_points,
        thresholds={
            "critical_reserve_margin": cfg.critical_reserve_margin,
            "high_reserve_margin": cfg.high_reserve_margin,
            "medium_reserve_margin": cfg.medium_reserve_margin,
            "spike_abs_threshold_mw": cfg.spike_abs_threshold_mw,
        },
    )


# ---------------------------------------------------------------------------
# Dashboard (serves the HTML operator dashboard)
# ---------------------------------------------------------------------------

from fastapi.responses import HTMLResponse
import pathlib as _pathlib

@app.get("/dashboard", response_class=HTMLResponse, tags=["System"], include_in_schema=False)
def get_dashboard():
    """Serve the operator dashboard HTML."""
    html_path = _pathlib.Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


@app.get("/", response_class=HTMLResponse, tags=["System"], include_in_schema=False)
def landing_page():
    """Serve the GridWise AI landing page."""
    html_path = _pathlib.Path(__file__).parent / "landing.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content='<meta http-equiv="refresh" content="0;url=/dashboard" />')


@app.get("/login", response_class=HTMLResponse, tags=["System"], include_in_schema=False)
def login_page():
    """Serve the login page."""
    html_path = _pathlib.Path(__file__).parent / "login.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Login page not found</h1>", status_code=404)


@app.get("/signup", response_class=HTMLResponse, tags=["System"], include_in_schema=False)
def signup_page():
    """Serve the account creation page."""
    html_path = _pathlib.Path(__file__).parent / "signup.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Signup page not found</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True)
