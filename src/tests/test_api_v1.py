"""
Test suite for /api/v1 endpoints.

Covers:
  - GET /api/v1/health
  - GET /api/v1/grid/status
  - GET /api/v1/forecast/load
  - GET /api/v1/forecast/renewables (all assets, solar-only, wind-only)
  - GET /api/v1/grid/risks (default + custom thresholds)
  - GET /api/v1/assets/anomalies (default, severity filter, bad severity)
  - GET /api/v1/assets/{id}/diagnosis (SOL-01, WIN-01, missing asset)
  - GET /api/v1/optimisation/plan (live + scenario B)
  - GET /api/v1/curtailment/plan (live + scenario C)
  - GET /api/v1/advisor/brief (live, scenario A, scenario B, scenario C)
  - Response field completeness
  - Data type labels present
  - Synthetic data disclaimer present
  - No fabricated fields (all numeric fields are real floats)
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from fastapi.testclient import TestClient
from src.backend.main import app

client = TestClient(app)


# ============================================================================
# Helpers
# ============================================================================

def _get(path: str, params: dict | None = None):
    r = client.get(path, params=params or {})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:300]}"
    return r.json()


# ============================================================================
# Health
# ============================================================================

class TestHealth:
    def test_health_ok(self):
        d = _get("/api/v1/health")
        assert d["status"] == "healthy"

    def test_health_has_version(self):
        d = _get("/api/v1/health")
        assert "version" in d

    def test_health_has_timestamp(self):
        d = _get("/api/v1/health")
        assert "timestamp" in d

    def test_health_data_note(self):
        d = _get("/api/v1/health")
        assert "SYNTHETIC" in d["data_note"].upper() or "synthetic" in d["data_note"]


# ============================================================================
# Grid Status
# ============================================================================

class TestGridStatus:
    def test_status_ok(self):
        d = _get("/api/v1/grid/status")
        assert "current_load_mw" in d

    def test_status_has_all_fields(self):
        d = _get("/api/v1/grid/status")
        for field in ("current_load_mw", "forecast_load_mw", "forecast_peak_mw",
                      "peak_time", "available_capacity_mw", "reserve_margin_percent", "risk_level"):
            assert field in d, f"Missing field: {field}"

    def test_status_load_positive(self):
        d = _get("/api/v1/grid/status")
        assert d["current_load_mw"] >= 0
        assert d["forecast_peak_mw"] >= 0

    def test_status_risk_level_valid(self):
        d = _get("/api/v1/grid/status")
        assert d["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_status_reserve_margin_range(self):
        d = _get("/api/v1/grid/status")
        # Reserve margin can go negative in edge cases but should be a real number
        assert isinstance(d["reserve_margin_percent"], (int, float))

    def test_status_data_type_field(self):
        d = _get("/api/v1/grid/status")
        assert "data_type" in d


# ============================================================================
# Load Forecast
# ============================================================================

class TestLoadForecast:
    def test_forecast_ok(self):
        d = _get("/api/v1/forecast/load")
        assert "forecast" in d
        assert len(d["forecast"]) > 0

    def test_forecast_default_24h(self):
        d = _get("/api/v1/forecast/load")
        assert d["horizon_hours"] == 24
        assert len(d["forecast"]) == 24

    def test_forecast_custom_horizon(self):
        d = _get("/api/v1/forecast/load", {"horizon_hours": 12})
        assert d["horizon_hours"] == 12
        assert len(d["forecast"]) == 12

    def test_forecast_points_have_timestamp_and_load(self):
        d = _get("/api/v1/forecast/load")
        for pt in d["forecast"]:
            assert "timestamp" in pt
            assert "forecast_load_mw" in pt
            assert isinstance(pt["forecast_load_mw"], (int, float))

    def test_forecast_load_values_positive(self):
        d = _get("/api/v1/forecast/load")
        for pt in d["forecast"]:
            assert pt["forecast_load_mw"] >= 0

    def test_forecast_model_metrics_present(self):
        d = _get("/api/v1/forecast/load")
        # model_metrics is Optional but should be present after training
        assert "model_metrics" in d

    def test_forecast_data_type(self):
        d = _get("/api/v1/forecast/load")
        assert d["data_type"] == "ML_PREDICTION"


# ============================================================================
# Renewable Forecast
# ============================================================================

class TestRenewableForecast:
    def test_renewables_ok(self):
        d = _get("/api/v1/forecast/renewables")
        assert "forecasts" in d
        assert len(d["forecasts"]) > 0

    def test_renewables_solar_filter(self):
        d = _get("/api/v1/forecast/renewables", {"asset_type": "solar"})
        assert all(p["asset_type"] == "solar" for p in d["forecasts"])

    def test_renewables_wind_filter(self):
        d = _get("/api/v1/forecast/renewables", {"asset_type": "wind"})
        assert all(p["asset_type"] == "wind" for p in d["forecasts"])

    def test_renewables_invalid_type(self):
        r = client.get("/api/v1/forecast/renewables", params={"asset_type": "nuclear"})
        assert r.status_code == 400

    def test_renewables_point_fields(self):
        d = _get("/api/v1/forecast/renewables")
        pt = d["forecasts"][0]
        for f in ("timestamp", "asset_id", "asset_type", "expected_generation_mw",
                  "capacity_mw", "capacity_factor"):
            assert f in pt, f"Missing field: {f}"

    def test_renewables_capacity_factor_range(self):
        d = _get("/api/v1/forecast/renewables")
        for pt in d["forecasts"]:
            assert 0.0 <= pt["capacity_factor"] <= 1.0, f"capacity_factor out of range: {pt['capacity_factor']}"

    def test_renewables_generation_non_negative(self):
        d = _get("/api/v1/forecast/renewables")
        for pt in d["forecasts"]:
            assert pt["expected_generation_mw"] >= 0


# ============================================================================
# Grid Risks
# ============================================================================

class TestGridRisks:
    def test_risks_ok(self):
        d = _get("/api/v1/grid/risks")
        assert "risks" in d
        assert len(d["risks"]) > 0

    def test_risks_fields(self):
        d = _get("/api/v1/grid/risks")
        for f in ("total_periods", "critical_periods", "high_risk_periods", "thresholds"):
            assert f in d, f"Missing field: {f}"

    def test_risks_levels_valid(self):
        d = _get("/api/v1/grid/risks")
        valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for r in d["risks"]:
            assert r["risk_level"] in valid

    def test_risks_score_range(self):
        d = _get("/api/v1/grid/risks")
        for r in d["risks"]:
            assert 0.0 <= r["spike_risk_score"] <= 1.0

    def test_risks_explanation_present(self):
        d = _get("/api/v1/grid/risks")
        for r in d["risks"]:
            assert "explanation" in r
            assert len(r["explanation"]) > 0

    def test_risks_custom_thresholds(self):
        d = _get("/api/v1/grid/risks", {"critical_reserve": 0.01, "high_reserve": 0.05})
        # With very low thresholds, fewer or zero critical periods expected
        assert isinstance(d["critical_periods"], int)

    def test_risks_reserve_margin_percent_present(self):
        d = _get("/api/v1/grid/risks")
        for r in d["risks"]:
            assert "reserve_margin_percent" in r


# ============================================================================
# Asset Anomalies
# ============================================================================

class TestAssetAnomalies:
    def test_anomalies_ok(self):
        d = _get("/api/v1/assets/anomalies")
        assert "anomalies" in d

    def test_anomalies_summary_fields(self):
        d = _get("/api/v1/assets/anomalies")
        for f in ("total_assets", "assets_in_warning", "assets_in_critical"):
            assert f in d

    def test_anomalies_severity_filter_warning(self):
        d = _get("/api/v1/assets/anomalies", {"severity": "WARNING"})
        for a in d["anomalies"]:
            assert a["severity"] == "WARNING"

    def test_anomalies_severity_filter_critical(self):
        d = _get("/api/v1/assets/anomalies", {"severity": "CRITICAL"})
        for a in d["anomalies"]:
            assert a["severity"] == "CRITICAL"

    def test_anomalies_severity_filter_normal(self):
        d = _get("/api/v1/assets/anomalies", {"severity": "NORMAL"})
        for a in d["anomalies"]:
            assert a["severity"] == "NORMAL"

    def test_anomalies_invalid_severity(self):
        r = client.get("/api/v1/assets/anomalies", params={"severity": "UNKNOWN"})
        assert r.status_code == 400

    def test_anomalies_performance_ratio_range(self):
        d = _get("/api/v1/assets/anomalies")
        for a in d["anomalies"]:
            assert a["performance_ratio"] >= 0.0

    def test_anomalies_point_fields(self):
        d = _get("/api/v1/assets/anomalies")
        if d["anomalies"]:
            a = d["anomalies"][0]
            for f in ("asset_id", "asset_name", "asset_type", "actual_generation_mw",
                      "expected_generation_mw", "performance_ratio", "deviation_percent", "severity"):
                assert f in a, f"Missing field: {f}"

    def test_anomalies_data_type(self):
        d = _get("/api/v1/assets/anomalies")
        assert d["data_type"] == "ANOMALY_DETECTION"


# ============================================================================
# Asset Diagnosis
# ============================================================================

class TestAssetDiagnosis:
    def test_diagnosis_sol01(self):
        d = _get("/api/v1/assets/SOL-01/diagnosis")
        assert d["asset_id"] == "SOL-01"

    def test_diagnosis_win01(self):
        d = _get("/api/v1/assets/WIN-01/diagnosis")
        assert d["asset_id"] == "WIN-01"

    def test_diagnosis_not_found(self):
        r = client.get("/api/v1/assets/NONEXISTENT/diagnosis")
        assert r.status_code == 404

    def test_diagnosis_has_all_fields(self):
        d = _get("/api/v1/assets/SOL-01/diagnosis")
        for f in ("asset_id", "asset_name", "asset_type", "expected_generation_mw",
                  "actual_generation_mw", "deviation_mw", "weather_conditions",
                  "weather_explained_loss_mw", "unexplained_loss_mw",
                  "probable_causes", "recommended_inspection", "inspection_notes"):
            assert f in d, f"Missing field: {f}"

    def test_diagnosis_weather_conditions_present(self):
        d = _get("/api/v1/assets/SOL-01/diagnosis")
        w = d["weather_conditions"]
        assert isinstance(w, dict)
        assert len(w) > 0

    def test_diagnosis_unexplained_loss_non_negative(self):
        d = _get("/api/v1/assets/SOL-01/diagnosis")
        assert d["unexplained_loss_mw"] >= 0.0

    def test_diagnosis_probable_causes_list(self):
        d = _get("/api/v1/assets/SOL-01/diagnosis")
        assert isinstance(d["probable_causes"], list)
        for c in d["probable_causes"]:
            assert "cause" in c
            assert "confidence" in c
            assert 0.0 <= c["confidence"] <= 1.0

    def test_diagnosis_data_type(self):
        d = _get("/api/v1/assets/SOL-01/diagnosis")
        assert d["data_type"] == "ROOT_CAUSE_ANALYSIS"

    def test_diagnosis_wind_asset(self):
        d = _get("/api/v1/assets/WIN-01/diagnosis")
        assert d["asset_type"] == "wind"
        assert isinstance(d["probable_causes"], list)


# ============================================================================
# Optimisation Plan
# ============================================================================

class TestOptimisationPlan:
    def test_opt_ok(self):
        d = _get("/api/v1/optimisation/plan")
        assert "intervals" in d
        assert len(d["intervals"]) > 0

    def test_opt_summary_fields(self):
        d = _get("/api/v1/optimisation/plan")
        s = d["summary"]
        for f in ("deficit_periods", "surplus_periods", "total_bess_charge_mwh",
                  "total_bess_discharge_mwh", "backup_required"):
            assert f in s, f"Missing summary field: {f}"

    def test_opt_interval_fields(self):
        d = _get("/api/v1/optimisation/plan")
        iv = d["intervals"][0]
        for f in ("timestamp", "forecast_demand_mw", "forecast_renewable_mw",
                  "bess_action_mw", "demand_response_mw", "backup_generation_mw",
                  "net_balance_mw", "action_taken"):
            assert f in iv, f"Missing interval field: {f}"

    def test_opt_scenario_b(self):
        d = _get("/api/v1/optimisation/plan", {"scenario": "B"})
        assert "DEMO" in d["scenario"] or "B" in d["scenario"]

    def test_opt_invalid_scenario(self):
        r = client.get("/api/v1/optimisation/plan", params={"scenario": "Z"})
        assert r.status_code == 400

    def test_opt_data_type(self):
        d = _get("/api/v1/optimisation/plan")
        assert d["data_type"] == "OPTIMISATION_RESULT"


# ============================================================================
# Curtailment Plan
# ============================================================================

class TestCurtailmentPlan:
    def test_curt_ok(self):
        d = _get("/api/v1/curtailment/plan")
        assert "summary" in d
        assert "intervals" in d

    def test_curt_summary_fields(self):
        d = _get("/api/v1/curtailment/plan")
        s = d["summary"]
        for f in ("total_potential_curtailment_mwh", "total_bess_absorption_mwh",
                  "total_avoided_curtailment_mwh", "curtailment_reduction_pct"):
            assert f in s, f"Missing summary field: {f}"

    def test_curt_interval_fields(self):
        d = _get("/api/v1/curtailment/plan")
        if d["intervals"]:
            iv = d["intervals"][0]
            for f in ("timestamp", "renewable_surplus_mw", "potential_curtailment_mw",
                      "bess_absorption_mw", "unavoidable_curtailment_mw", "avoided_curtailment_mw"):
                assert f in iv, f"Missing interval field: {f}"

    def test_curt_scenario_c(self):
        d = _get("/api/v1/curtailment/plan", {"scenario": "C"})
        assert "C" in d["scenario"] or "DEMO" in d["scenario"]

    def test_curt_explanation_present(self):
        d = _get("/api/v1/curtailment/plan")
        assert "explanation" in d
        assert isinstance(d["explanation"], str)

    def test_curt_data_type(self):
        d = _get("/api/v1/curtailment/plan")
        assert d["data_type"] == "CURTAILMENT_ANALYSIS"


# ============================================================================
# Advisor Brief
# ============================================================================

class TestAdvisorBrief:
    def test_brief_live(self):
        d = _get("/api/v1/advisor/brief", {"scenario": "live"})
        assert d["scenario"] == "live"
        assert d["is_demo_scenario"] is False

    def test_brief_scenario_a(self):
        d = _get("/api/v1/advisor/brief", {"scenario": "A"})
        assert d["scenario"] == "A"
        assert d["is_demo_scenario"] is True

    def test_brief_scenario_b(self):
        d = _get("/api/v1/advisor/brief", {"scenario": "B"})
        assert d["scenario"] == "B"
        assert d["is_demo_scenario"] is True

    def test_brief_scenario_c(self):
        d = _get("/api/v1/advisor/brief", {"scenario": "C"})
        assert d["scenario"] == "C"
        assert d["is_demo_scenario"] is True

    def test_brief_top_level_fields(self):
        d = _get("/api/v1/advisor/brief")
        for f in ("generated_at", "scenario", "is_demo_scenario", "grid_status",
                  "demand_forecast", "demand_risks", "renewable_status",
                  "anomalous_assets", "root_causes", "optimisation_plan",
                  "curtailment_plan", "recommended_actions", "data_quality",
                  "known_uncertainties", "full_text"):
            assert f in d, f"Missing top-level field: {f}"

    def test_brief_grid_status_fields(self):
        d = _get("/api/v1/advisor/brief")
        gs = d["grid_status"]
        for f in ("current_load_mw", "available_capacity_mw", "reserve_margin_pct", "overall_risk"):
            assert f in gs

    def test_brief_demand_forecast_fields(self):
        d = _get("/api/v1/advisor/brief")
        df = d["demand_forecast"]
        for f in ("horizon_hours", "peak_mw", "peak_time", "average_mw"):
            assert f in df

    def test_brief_recommended_actions_list(self):
        d = _get("/api/v1/advisor/brief")
        assert isinstance(d["recommended_actions"], list)
        assert len(d["recommended_actions"]) >= 1

    def test_brief_full_text_non_empty(self):
        d = _get("/api/v1/advisor/brief")
        assert len(d["full_text"]) > 100

    def test_brief_known_uncertainties_list(self):
        d = _get("/api/v1/advisor/brief")
        assert isinstance(d["known_uncertainties"], list)
        assert len(d["known_uncertainties"]) >= 1

    def test_brief_data_quality_dict(self):
        d = _get("/api/v1/advisor/brief")
        assert isinstance(d["data_quality"], dict)
        assert len(d["data_quality"]) >= 3

    def test_brief_demo_uncertainty_label(self):
        """Demo scenarios must clearly identify themselves."""
        d = _get("/api/v1/advisor/brief", {"scenario": "B"})
        uncertainties_text = " ".join(d["known_uncertainties"]).upper()
        assert "DEMO" in uncertainties_text or "SCENARIO" in uncertainties_text

    def test_brief_data_type(self):
        d = _get("/api/v1/advisor/brief")
        assert d["data_type"] == "OPERATOR_BRIEF"

    def test_brief_invalid_scenario(self):
        r = client.get("/api/v1/advisor/brief", params={"scenario": "Z"})
        assert r.status_code == 400

    def test_brief_curtailment_plan_fields(self):
        d = _get("/api/v1/advisor/brief")
        cp = d["curtailment_plan"]
        for f in ("potential_mwh", "avoided_mwh", "reduction_pct"):
            assert f in cp

    def test_brief_optimisation_plan_fields(self):
        d = _get("/api/v1/advisor/brief")
        op = d["optimisation_plan"]
        for f in ("deficit_periods", "surplus_periods", "total_bess_discharge_mwh", "backup_required"):
            assert f in op
