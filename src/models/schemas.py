"""
Pydantic data models / schemas for GridWise AI.

All models are used for both internal data exchange and API serialisation.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AssetType(str, Enum):
    SOLAR = "solar"
    WIND = "wind"


class AssetStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DERATED = "derated"
    MAINTENANCE = "maintenance"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnomalySeverity(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Grid record
# ---------------------------------------------------------------------------

class GridRecord(BaseModel):
    timestamp: datetime
    grid_load_mw: float = Field(..., description="Measured grid load in MW")
    available_capacity_mw: float = Field(..., description="Total available generation capacity in MW")
    conventional_generation_mw: float = Field(..., description="Conventional (thermal/hydro/nuclear) generation in MW")


# ---------------------------------------------------------------------------
# Weather record
# ---------------------------------------------------------------------------

class WeatherRecord(BaseModel):
    timestamp: datetime
    temperature: float = Field(..., description="Ambient temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover (%)")
    irradiance: float = Field(..., ge=0, description="Global horizontal irradiance (W/m²)")
    wind_speed: float = Field(..., ge=0, description="Wind speed at hub height (m/s)")
    wind_direction: float = Field(..., ge=0, lt=360, description="Wind direction (degrees)")


# ---------------------------------------------------------------------------
# Renewable asset
# ---------------------------------------------------------------------------

class RenewableAsset(BaseModel):
    asset_id: str
    asset_name: str
    asset_type: AssetType
    capacity_mw: float = Field(..., gt=0)
    actual_generation_mw: float = Field(..., ge=0)
    expected_generation_mw: float = Field(..., ge=0)
    status: AssetStatus = AssetStatus.ONLINE


# ---------------------------------------------------------------------------
# BESS (Battery Energy Storage System)
# ---------------------------------------------------------------------------

class BESSRecord(BaseModel):
    timestamp: datetime
    asset_id: str
    capacity_mwh: float = Field(..., gt=0, description="Nominal BESS capacity (MWh)")
    current_soc: float = Field(..., ge=0, le=1, description="State of charge [0–1]")
    max_charge_mw: float = Field(..., ge=0)
    max_discharge_mw: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Forecast outputs
# ---------------------------------------------------------------------------

class LoadForecastPoint(BaseModel):
    timestamp: datetime
    forecast_load_mw: float
    lower_bound_mw: Optional[float] = None
    upper_bound_mw: Optional[float] = None


class LoadForecastResponse(BaseModel):
    generated_at: datetime
    horizon_hours: int
    forecast: List[LoadForecastPoint]
    model_metrics: Optional[dict] = None


class RenewableForecastPoint(BaseModel):
    timestamp: datetime
    asset_id: str
    asset_type: AssetType
    expected_generation_mw: float
    capacity_mw: float
    capacity_factor: float


class RenewableForecastResponse(BaseModel):
    generated_at: datetime
    forecasts: List[RenewableForecastPoint]


# ---------------------------------------------------------------------------
# Demand spike / grid risk
# ---------------------------------------------------------------------------

class GridRiskPoint(BaseModel):
    timestamp: datetime
    forecast_load_mw: float
    available_capacity_mw: float
    reserve_margin: float = Field(..., description="(capacity - load) / capacity")
    spike_risk_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel


class GridRiskResponse(BaseModel):
    generated_at: datetime
    risk_summary: List[GridRiskPoint]
    thresholds: dict


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class AssetAnomaly(BaseModel):
    asset_id: str
    asset_name: str
    asset_type: AssetType
    timestamp: datetime
    actual_generation_mw: float
    expected_generation_mw: float
    absolute_deviation_mw: float
    percentage_deviation: float
    performance_ratio: float
    severity: AnomalySeverity


class AnomalyResponse(BaseModel):
    generated_at: datetime
    anomalies: List[AssetAnomaly]
    total_assets: int
    assets_in_warning: int
    assets_in_critical: int


# ---------------------------------------------------------------------------
# Root cause analysis
# ---------------------------------------------------------------------------

class CauseCandidate(BaseModel):
    cause: str
    confidence: float = Field(..., ge=0, le=1)
    description: str


class AssetDiagnosis(BaseModel):
    asset_id: str
    asset_name: str
    asset_type: AssetType
    timestamp: datetime
    expected_generation_mw: float
    actual_generation_mw: float
    deviation_mw: float
    deviation_pct: float
    weather_explained_loss_mw: float
    unexplained_loss_mw: float
    probable_causes: List[CauseCandidate]
    overall_confidence: float
    recommended_inspection: bool
    inspection_notes: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
