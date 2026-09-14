"""
Phase 2 test suite — Grid Optimisation, Curtailment, RCA edge cases.

Tests:
  - Supply deficit scenarios
  - Renewable surplus scenarios
  - Empty BESS
  - Full BESS (no charging headroom)
  - BESS power limits respected
  - Severe demand spike
  - Multiple anomalous assets
  - Weather-caused renewable reduction
  - Unexplained renewable reduction
  - Optimisation energy conservation
  - Constraint verification
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.ml.grid_optimiser import (
    run_optimisation, optimise_interval, BESSState, OptimiserConfig
)
from src.ml.curtailment import compute_curtailment_analysis
from src.ml.anomaly_detection import detect_anomalies, AnomalyConfig
from src.ml.root_cause_analysis import diagnose_asset, RCAConfig
from src.ml.renewable_generation import solar_expected, wind_expected
from src.data.scenarios import scenario_a_normal, scenario_b_demand_spike, scenario_c_surplus_and_anomaly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bess(soc: float = 0.50, capacity_mwh: float = 200.0,
               max_charge: float = 50.0, max_discharge: float = 50.0) -> List[BESSState]:
    return [BESSState("BESS-01", capacity_mwh, soc, max_charge, max_discharge)]


def _make_bess_df(soc: float = 0.50, capacity_mwh: float = 200.0,
                  max_charge: float = 50.0, max_discharge: float = 50.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "timestamp": datetime(2023, 7, 1, 0, tzinfo=timezone.utc),
        "asset_id": "BESS-01",
        "capacity_mwh": capacity_mwh,
        "current_soc": soc,
        "max_charge_mw": max_charge,
        "max_discharge_mw": max_discharge,
    }])


def _make_forecast_df(demands, renewables=None, conventionals=None, capacity=2600.0):
    base = datetime(2023, 7, 1, 0, tzinfo=timezone.utc)
    rows = []
    for i, d in enumerate(demands):
        rows.append({
            "timestamp": base + timedelta(hours=i),
            "demand_mw": d,
            "renewable_mw": (renewables[i] if renewables else 200.0),
            "conventional_mw": (conventionals[i] if conventionals else 900.0),
            "available_capacity_mw": capacity,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 1. Supply deficit — BESS discharge + demand response + backup
# ===========================================================================

class TestSupplyDeficit:
    def test_bess_discharges_in_deficit(self):
        """BESS should discharge to cover deficit."""
        # demand=1600, renewable=100, conventional=1000 → deficit=500 MW
        cfg = OptimiserConfig()
        bess = _make_bess(soc=0.80, capacity_mwh=200.0, max_discharge=100.0)
        result, updated = optimise_interval(
            datetime(2023, 7, 1, 18, tzinfo=timezone.utc),
            demand_mw=1600.0, renewable_mw=100.0, conventional_mw=1000.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        assert result.bess_action_mw < 0, "BESS should discharge (negative action)"
        assert result.net_balance_mw < 0, "Should be deficit scenario"

    def test_bess_soc_not_below_min(self):
        """SOC must not drop below configured minimum."""
        cfg = OptimiserConfig(bess_soc_min=0.10)
        bess = _make_bess(soc=0.15, capacity_mwh=200.0, max_discharge=100.0)
        result, updated = optimise_interval(
            datetime(2023, 7, 1, 18, tzinfo=timezone.utc),
            demand_mw=2000.0, renewable_mw=50.0, conventional_mw=1000.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        for b in updated:
            assert b.current_soc >= cfg.bess_soc_min - 0.001, "SOC below minimum!"

    def test_demand_response_covers_remaining_deficit(self):
        """Demand response activates when BESS is empty and deficit remains."""
        cfg = OptimiserConfig(bess_soc_min=0.10, demand_response_capacity_mw=200.0)
        # BESS empty
        bess = _make_bess(soc=0.10, capacity_mwh=200.0, max_discharge=50.0)
        result, _ = optimise_interval(
            datetime(2023, 7, 1, 18, tzinfo=timezone.utc),
            demand_mw=1500.0, renewable_mw=50.0, conventional_mw=900.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        # BESS can't discharge at all at min SOC — should trigger DR
        assert result.demand_response_mw > 0 or result.backup_generation_mw > 0

    def test_severe_spike_requires_backup(self):
        """When BESS + DR can't cover deficit, backup generation is required."""
        cfg = OptimiserConfig(
            demand_response_capacity_mw=50.0,
            bess_soc_min=0.10,
        )
        # Empty BESS, deficit 800 MW — too much for DR alone
        bess = _make_bess(soc=0.10, capacity_mwh=100.0, max_discharge=50.0)
        result, _ = optimise_interval(
            datetime(2023, 7, 1, 18, tzinfo=timezone.utc),
            demand_mw=2000.0, renewable_mw=100.0, conventional_mw=1000.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        assert result.backup_generation_mw > 0, "Backup generation must be called"
        assert "backup_generation_required" in result.constraints_violated

    def test_empty_bess_no_discharge(self):
        """Empty BESS at min SOC should not attempt to discharge."""
        cfg = OptimiserConfig(bess_soc_min=0.10)
        bess = _make_bess(soc=0.10)  # at minimum
        result, updated = optimise_interval(
            datetime(2023, 7, 1, 18, tzinfo=timezone.utc),
            demand_mw=1500.0, renewable_mw=200.0, conventional_mw=900.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        # BESS discharge should be zero or near zero
        assert result.bess_action_mw >= -0.5, "Empty BESS should not discharge"

    def test_bess_power_limit_respected(self):
        """BESS discharge must not exceed max_discharge_mw."""
        cfg = OptimiserConfig()
        max_d = 30.0
        bess = _make_bess(soc=0.90, capacity_mwh=200.0, max_discharge=max_d)
        result, _ = optimise_interval(
            datetime(2023, 7, 1, 18, tzinfo=timezone.utc),
            demand_mw=2200.0, renewable_mw=100.0, conventional_mw=1000.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        assert abs(result.bess_action_mw) <= max_d + 0.1, "BESS discharge exceeded power limit"


# ===========================================================================
# 2. Renewable surplus
# ===========================================================================

class TestRenewableSurplus:
    def test_bess_charges_in_surplus(self):
        """BESS should charge when there is surplus."""
        cfg = OptimiserConfig()
        bess = _make_bess(soc=0.30)  # plenty of headroom
        result, updated = optimise_interval(
            datetime(2023, 7, 1, 12, tzinfo=timezone.utc),
            demand_mw=800.0, renewable_mw=600.0, conventional_mw=500.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        assert result.bess_action_mw > 0, "BESS should charge in surplus"
        assert result.net_balance_mw > 0, "Should be surplus scenario"

    def test_full_bess_no_charging(self):
        """Full BESS should not absorb more charge."""
        cfg = OptimiserConfig(bess_soc_max=0.95)
        bess = _make_bess(soc=0.95)  # at max
        result, updated = optimise_interval(
            datetime(2023, 7, 1, 12, tzinfo=timezone.utc),
            demand_mw=800.0, renewable_mw=600.0, conventional_mw=500.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        assert result.bess_action_mw <= 0.1, "Full BESS should not charge"

    def test_bess_charge_limit_respected(self):
        """BESS charging must not exceed max_charge_mw."""
        cfg = OptimiserConfig()
        max_c = 20.0
        bess = _make_bess(soc=0.20, capacity_mwh=200.0, max_charge=max_c)
        result, _ = optimise_interval(
            datetime(2023, 7, 1, 12, tzinfo=timezone.utc),
            demand_mw=500.0, renewable_mw=800.0, conventional_mw=300.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        assert result.bess_action_mw <= max_c + 0.1, "BESS charge exceeded power limit"

    def test_surplus_curtailment_last_resort(self):
        """Curtailment should only occur when all absorption options are exhausted."""
        cfg = OptimiserConfig(
            bess_soc_max=0.95,
            demand_response_capacity_mw=10.0,
            export_capacity_mw=10.0,
        )
        bess = _make_bess(soc=0.95)  # full
        # Huge surplus: 900 MW above demand, tiny DR + export + full BESS
        result, _ = optimise_interval(
            datetime(2023, 7, 1, 12, tzinfo=timezone.utc),
            demand_mw=500.0, renewable_mw=1400.0, conventional_mw=0.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        # Should have curtailment since all absorption maxed out
        assert result.curtailment_mw > 0
        # Curtailment = surplus - all_absorbed
        absorbed = (result.bess_action_mw + abs(min(0, result.demand_response_mw)) + result.export_mw)
        assert result.curtailment_mw <= result.net_balance_mw + 0.5


# ===========================================================================
# 3. Energy conservation
# ===========================================================================

class TestEnergyConservation:
    def test_energy_balance(self):
        """Total supply + discharge should >= demand after optimisation."""
        cfg = OptimiserConfig()
        bess = _make_bess(soc=0.60)
        result, _ = optimise_interval(
            datetime(2023, 7, 1, 14, tzinfo=timezone.utc),
            demand_mw=1400.0, renewable_mw=200.0, conventional_mw=1000.0,
            available_capacity_mw=2600.0, bess_states=bess, cfg=cfg
        )
        effective_supply = (
            result.renewable_mw + result.conventional_mw
            + abs(min(0, result.bess_action_mw))  # discharge
            + result.backup_generation_mw
        )
        effective_demand = result.demand_mw - result.demand_response_mw
        # Supply should cover effective demand (within rounding)
        assert effective_supply >= effective_demand - 1.0

    def test_batch_optimisation_runs(self):
        """Full 24-hour optimisation should produce 24 records."""
        demands = [1200 + 100 * i % 5 for i in range(24)]
        fc_df = _make_forecast_df(demands)
        bess_df = _make_bess_df(soc=0.50)
        result_df = run_optimisation(fc_df, bess_df)
        assert len(result_df) == 24
        assert "action_taken" in result_df.columns
        assert "bess_soc_after" in result_df.columns


# ===========================================================================
# 4. Scenario tests
# ===========================================================================

class TestScenarios:
    def test_scenario_a_no_backup(self):
        """Scenario A (normal) should not require backup generation."""
        ctx = scenario_a_normal()
        opt_df = run_optimisation(ctx["forecast_df"], ctx["bess_df"])
        assert not any(opt_df["backup_generation_mw"] > 0), "Normal scenario should not need backup"

    def test_scenario_b_has_deficit(self):
        """Scenario B (spike) must have at least some deficit intervals."""
        ctx = scenario_b_demand_spike()
        opt_df = run_optimisation(ctx["forecast_df"], ctx["bess_df"])
        assert any(opt_df["net_balance_mw"] < 0), "Spike scenario must have deficit periods"

    def test_scenario_c_has_curtailment(self):
        """Scenario C (surplus + full BESS) must have some curtailment."""
        ctx = scenario_c_surplus_and_anomaly()
        opt_df = run_optimisation(ctx["forecast_df"], ctx["bess_df"])
        curtailment = compute_curtailment_analysis(opt_df)
        # With BESS nearly full and high surplus, there should be curtailment
        assert curtailment["summary"]["total_potential_curtailment_mwh"] > 0


# ===========================================================================
# 5. Curtailment analysis
# ===========================================================================

class TestCurtailmentAnalysis:
    def _run_surplus_scenario(self, soc=0.20):
        demands = [700.0] * 24
        renewables = [600.0] * 24
        conventionals = [400.0] * 24
        fc_df = _make_forecast_df(demands, renewables, conventionals)
        bess_df = _make_bess_df(soc=soc)
        opt_df = run_optimisation(fc_df, bess_df)
        return compute_curtailment_analysis(opt_df)

    def test_bess_absorption_counted(self):
        c = self._run_surplus_scenario(soc=0.20)
        assert c["summary"]["total_bess_absorption_mwh"] > 0

    def test_avoided_plus_unavoidable_equals_potential(self):
        c = self._run_surplus_scenario(soc=0.50)
        s = c["summary"]
        # Avoided + unavoidable should approximately equal potential
        total = s["total_avoided_curtailment_mwh"] + s["total_unavoidable_curtailment_mwh"]
        assert abs(total - s["total_potential_curtailment_mwh"]) < 0.1

    def test_no_surplus_no_curtailment(self):
        """When there's no surplus, curtailment should be zero."""
        demands = [1800.0] * 24  # high demand, low renewables
        renewables = [100.0] * 24
        conventionals = [1200.0] * 24
        fc_df = _make_forecast_df(demands, renewables, conventionals)
        bess_df = _make_bess_df(soc=0.50)
        opt_df = run_optimisation(fc_df, bess_df)
        c = compute_curtailment_analysis(opt_df)
        assert c["summary"]["total_unavoidable_curtailment_mwh"] == 0.0


# ===========================================================================
# 6. Multiple anomalous assets
# ===========================================================================

class TestMultipleAnomalies:
    def _make_multi_asset_df(self):
        ts = datetime(2023, 7, 1, 12, tzinfo=timezone.utc)
        return pd.DataFrame([
            {"timestamp": ts, "asset_id": "SOL-01", "asset_name": "Solar A",
             "asset_type": "solar", "capacity_mw": 50.0,
             "actual_generation_mw": 5.0, "expected_generation_mw": 40.0, "status": "online"},
            {"timestamp": ts, "asset_id": "SOL-02", "asset_name": "Solar B",
             "asset_type": "solar", "capacity_mw": 75.0,
             "actual_generation_mw": 10.0, "expected_generation_mw": 65.0, "status": "online"},
            {"timestamp": ts, "asset_id": "WIN-01", "asset_name": "Wind A",
             "asset_type": "wind", "capacity_mw": 80.0,
             "actual_generation_mw": 75.0, "expected_generation_mw": 78.0, "status": "online"},
        ])

    def test_multiple_critical_detected(self):
        df = self._make_multi_asset_df()
        anomaly_df = detect_anomalies(df)
        crits = anomaly_df[anomaly_df["severity"] == "CRITICAL"]
        assert len(crits) >= 2, "Both severely underperforming solar assets should be CRITICAL"

    def test_normal_asset_not_flagged(self):
        df = self._make_multi_asset_df()
        anomaly_df = detect_anomalies(df)
        win_row = anomaly_df[anomaly_df["asset_id"] == "WIN-01"].iloc[0]
        assert win_row["severity"] == "NORMAL"


# ===========================================================================
# 7. Weather vs unexplained causes
# ===========================================================================

class TestWeatherVsUnexplained:
    def test_heavy_cloud_mostly_weather_explained(self):
        """Loss during heavy cloud cover should be majority weather-explained."""
        result = diagnose_asset(
            "SOL-01", "Test", "solar",
            actual_mw=8.0, expected_mw=40.0, capacity_mw=50.0,
            weather={"cloud_cover": 90, "irradiance": 80, "temperature": 22,
                     "wind_speed": 5.0, "wind_direction": 180}
        )
        assert result["weather_explained_loss_mw"] >= result["unexplained_loss_mw"], \
            "Heavy cloud scenario should be mostly weather-explained"

    def test_clear_sky_low_generation_is_unexplained(self):
        """Low generation on a clear sunny day is unexplained."""
        result = diagnose_asset(
            "SOL-01", "Test", "solar",
            actual_mw=5.0, expected_mw=45.0, capacity_mw=50.0,
            weather={"cloud_cover": 5, "irradiance": 900, "temperature": 25,
                     "wind_speed": 5.0, "wind_direction": 180}
        )
        assert result["unexplained_loss_mw"] > result["weather_explained_loss_mw"], \
            "Clear sky with low generation should be largely unexplained"
        assert result["recommended_inspection"] is True

    def test_wind_below_cut_in_is_fully_explained(self):
        """Zero wind generation at cut-in is 100% weather-explained."""
        result = diagnose_asset(
            "WIN-01", "Test", "wind",
            actual_mw=0.0, expected_mw=5.0, capacity_mw=80.0,
            weather={"cloud_cover": 10, "irradiance": 0, "temperature": 15,
                     "wind_speed": 1.5, "wind_direction": 200}
        )
        assert result["unexplained_loss_mw"] == 0.0
        assert result["recommended_inspection"] is False

    def test_zero_generation_high_wind_is_unexplained(self):
        """Zero generation at rated wind speed is anomalous."""
        result = diagnose_asset(
            "WIN-01", "Test", "wind",
            actual_mw=0.0, expected_mw=80.0, capacity_mw=80.0,
            weather={"cloud_cover": 10, "irradiance": 0, "temperature": 15,
                     "wind_speed": 14.0, "wind_direction": 200}
        )
        # Should have significant unexplained loss
        assert result["unexplained_loss_mw"] > 30.0
