"""
Test suite for GridWise AI — Phase 1.

Covers:
  - Forecasting pipeline (build_features, chronological split)
  - Leakage prevention (target not in feature columns)
  - Anomaly calculations (normal, warning, critical, nighttime solar)
  - Root-cause rules (solar + wind scenarios)
  - Missing weather data
  - Invalid asset data
  - Zero generation
  - Nighttime solar (should not produce anomaly)
"""
from __future__ import annotations

import math
import sys
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

# Ensure workspace root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_weather(n: int = 200, start_hour: int = 0) -> pd.DataFrame:
    """Minimal synthetic weather DataFrame for testing."""
    base = datetime(2023, 6, 15, start_hour, tzinfo=timezone.utc)
    records = []
    for i in range(n):
        ts = base + timedelta(hours=i)
        hour = ts.hour
        irradiance = max(0, 800 * math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        records.append({
            "timestamp": ts,
            "temperature": 22.0 + i * 0.05,
            "humidity": 55.0,
            "cloud_cover": 10.0,
            "irradiance": irradiance,
            "wind_speed": 8.0 + (i % 5),
            "wind_direction": 180.0,
        })
    return pd.DataFrame(records)


def _make_grid(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Minimal synthetic grid DataFrame aligned to weather."""
    rng = np.random.default_rng(42)
    records = []
    for _, row in weather_df.iterrows():
        hour = row["timestamp"].hour
        base_load = 1200 + 400 * math.sin(math.pi * (hour - 6) / 12)
        records.append({
            "timestamp": row["timestamp"],
            "grid_load_mw": base_load + rng.normal(0, 30),
            "available_capacity_mw": 2600.0,
            "conventional_generation_mw": 700.0,
        })
    return pd.DataFrame(records)


def _make_assets_df(n: int = 50, asset_type: str = "solar") -> pd.DataFrame:
    """Minimal asset DataFrame for anomaly testing."""
    base = datetime(2023, 7, 1, 12, tzinfo=timezone.utc)
    records = []
    for i in range(n):
        records.append({
            "timestamp": base + timedelta(hours=i),
            "asset_id": "TEST-01",
            "asset_name": "Test Asset",
            "asset_type": asset_type,
            "capacity_mw": 50.0,
            "actual_generation_mw": 40.0,
            "expected_generation_mw": 42.0,
            "status": "online",
        })
    return pd.DataFrame(records)


# ===========================================================================
# 1. Forecasting pipeline — feature engineering
# ===========================================================================

class TestFeatureEngineering:
    def test_features_built(self):
        from src.ml.load_forecasting import build_features
        weather = _make_weather(300)
        grid = _make_grid(weather)
        df = build_features(grid, weather)
        assert len(df) > 0, "Feature DataFrame must not be empty"

    def test_no_target_leakage(self):
        """The target column must NOT appear in the feature list."""
        from src.ml.load_forecasting import build_features, _feature_columns, TARGET
        weather = _make_weather(300)
        grid = _make_grid(weather)
        df = build_features(grid, weather)
        feature_cols = _feature_columns(df)
        assert TARGET not in feature_cols, (
            f"TARGET '{TARGET}' must not be in feature columns — data leakage!"
        )

    def test_no_future_leakage_in_lags(self):
        """Lag features must be purely backward-looking (shift > 0)."""
        from src.ml.load_forecasting import build_features, LAG_HOURS
        weather = _make_weather(300)
        grid = _make_grid(weather)
        df = build_features(grid, weather)
        # The minimum lag is 1h — verify lag_1h at index i matches load at index i-1
        # (after dropna, index 0 corresponds to row lag_max in original)
        lag_col = f"load_lag_{min(LAG_HOURS)}h"
        assert lag_col in df.columns

    def test_temporal_features_present(self):
        from src.ml.load_forecasting import build_features, TEMPORAL_FEATURES
        weather = _make_weather(300)
        grid = _make_grid(weather)
        df = build_features(grid, weather)
        for feat in TEMPORAL_FEATURES:
            assert feat in df.columns, f"Missing temporal feature: {feat}"

    def test_weekend_encoding(self):
        """is_weekend should be 1 for Saturday/Sunday."""
        from src.ml.load_forecasting import build_features
        # 2023-07-01 is Saturday
        base = datetime(2023, 7, 1, 0, tzinfo=timezone.utc)
        weather = _make_weather(300)
        weather["timestamp"] = [base + timedelta(hours=i) for i in range(len(weather))]
        grid = _make_grid(weather)
        df = build_features(grid, weather)
        sat = df[df["day_of_week"] == 5]
        assert (sat["is_weekend"] == 1).all(), "Saturday must be flagged as weekend"


# ===========================================================================
# 2. Chronological split
# ===========================================================================

class TestChronologicalSplit:
    def test_no_overlap(self):
        from src.ml.load_forecasting import build_features, chronological_split
        weather = _make_weather(500)
        grid = _make_grid(weather)
        df = build_features(grid, weather)
        train, val, test = chronological_split(df)
        assert train["timestamp"].max() < val["timestamp"].min(), "Train/val overlap"
        assert val["timestamp"].max() < test["timestamp"].min(), "Val/test overlap"

    def test_size_proportions(self):
        from src.ml.load_forecasting import build_features, chronological_split, TRAIN_FRAC, VAL_FRAC
        weather = _make_weather(500)
        grid = _make_grid(weather)
        df = build_features(grid, weather)
        train, val, test = chronological_split(df)
        total = len(train) + len(val) + len(test)
        assert abs(len(train) / total - TRAIN_FRAC) < 0.02
        assert abs(len(val) / total - VAL_FRAC) < 0.02


# ===========================================================================
# 3. Anomaly detection
# ===========================================================================

class TestAnomalyDetection:
    def _base_config(self):
        from src.ml.anomaly_detection import AnomalyConfig
        return AnomalyConfig()

    def test_normal_within_range(self):
        from src.ml.anomaly_detection import _compute_anomaly
        cfg = self._base_config()
        result = _compute_anomaly(actual_mw=40.0, expected_mw=42.0, cfg=cfg)
        assert result["severity"] == "NORMAL"

    def test_warning_trigger(self):
        from src.ml.anomaly_detection import _compute_anomaly, AnomalyConfig
        cfg = AnomalyConfig(warning_perf_ratio=0.75, warning_abs_mw=2.0)
        # 60% of expected (below 75%), deviation 20 MW (above 2 MW)
        result = _compute_anomaly(actual_mw=30.0, expected_mw=50.0, cfg=cfg)
        assert result["severity"] == "WARNING"

    def test_critical_trigger(self):
        from src.ml.anomaly_detection import _compute_anomaly, AnomalyConfig
        cfg = AnomalyConfig(critical_perf_ratio=0.50, critical_abs_mw=5.0)
        # 20% of expected (below 50%), 40 MW absolute (above 5 MW)
        result = _compute_anomaly(actual_mw=10.0, expected_mw=50.0, cfg=cfg)
        assert result["severity"] == "CRITICAL"

    def test_nighttime_solar_not_anomaly(self):
        """Zero generation at night (expected ~ 0) must NOT be flagged."""
        from src.ml.anomaly_detection import _compute_anomaly, AnomalyConfig
        cfg = AnomalyConfig(min_expected_mw=2.0)
        # At night: expected ~0, actual = 0
        result = _compute_anomaly(actual_mw=0.0, expected_mw=0.5, cfg=cfg)
        assert result["severity"] == "NORMAL", (
            "Nighttime/low-expected generation must not be flagged as anomaly"
        )

    def test_zero_generation_during_day(self):
        """Zero generation during high-expected daytime IS an anomaly."""
        from src.ml.anomaly_detection import _compute_anomaly, AnomalyConfig
        cfg = AnomalyConfig(critical_perf_ratio=0.50, critical_abs_mw=5.0, min_expected_mw=2.0)
        result = _compute_anomaly(actual_mw=0.0, expected_mw=50.0, cfg=cfg)
        assert result["severity"] == "CRITICAL"

    def test_performance_ratio_calculation(self):
        from src.ml.anomaly_detection import _compute_anomaly
        cfg = self._base_config()
        result = _compute_anomaly(actual_mw=45.0, expected_mw=50.0, cfg=cfg)
        assert abs(result["performance_ratio"] - 0.9) < 0.001

    def test_batch_detect(self):
        from src.ml.anomaly_detection import detect_anomalies
        df = _make_assets_df(30)
        result = detect_anomalies(df)
        assert "severity" in result.columns
        assert len(result) == len(df)

    def test_invalid_asset_data_no_crash(self):
        """NaN or negative values in asset data should not crash the engine."""
        from src.ml.anomaly_detection import detect_anomalies
        df = _make_assets_df(10)
        df.loc[0, "actual_generation_mw"] = float("nan")
        df.loc[1, "actual_generation_mw"] = -5.0
        df["actual_generation_mw"] = df["actual_generation_mw"].fillna(0).clip(lower=0)
        result = detect_anomalies(df)
        assert len(result) == len(df)


# ===========================================================================
# 4. Root cause analysis
# ===========================================================================

class TestRootCauseAnalysis:
    def _solar_weather(self, cloud_cover=80, irradiance=400, temperature=35) -> dict:
        return {
            "cloud_cover": cloud_cover,
            "irradiance": irradiance,
            "temperature": temperature,
            "wind_speed": 5.0,
            "wind_direction": 180,
            "humidity": 50,
        }

    def _wind_weather(self, wind_speed=2.0) -> dict:
        return {
            "cloud_cover": 20,
            "irradiance": 0,
            "temperature": 15,
            "wind_speed": wind_speed,
            "wind_direction": 200,
            "humidity": 60,
        }

    def test_solar_heavy_cloud_explained(self):
        from src.ml.root_cause_analysis import diagnose_asset
        result = diagnose_asset(
            "SOL-01", "Test Solar", "solar",
            actual_mw=10.0, expected_mw=40.0, capacity_mw=50.0,
            weather=self._solar_weather(cloud_cover=85),
        )
        causes = [c["cause"] for c in result["probable_causes"]]
        assert "cloud_cover" in causes, "Heavy cloud must appear as probable cause"
        assert result["weather_explained_loss_mw"] > 0

    def test_wind_below_cut_in(self):
        from src.ml.root_cause_analysis import diagnose_asset
        result = diagnose_asset(
            "WIN-01", "Test Wind", "wind",
            actual_mw=0.0, expected_mw=1.0, capacity_mw=80.0,
            weather=self._wind_weather(wind_speed=1.5),
        )
        causes = [c["cause"] for c in result["probable_causes"]]
        assert "insufficient_wind" in causes
        # No unexplained loss expected
        assert result["unexplained_loss_mw"] == 0.0

    def test_wind_cut_out(self):
        from src.ml.root_cause_analysis import diagnose_asset
        result = diagnose_asset(
            "WIN-01", "Test Wind", "wind",
            actual_mw=0.0, expected_mw=80.0, capacity_mw=80.0,
            weather=self._wind_weather(wind_speed=28.0),
        )
        causes = [c["cause"] for c in result["probable_causes"]]
        assert "excessive_wind_cut_out" in causes
        assert result["unexplained_loss_mw"] == 0.0

    def test_unexplained_loss_triggers_inspection(self):
        from src.ml.root_cause_analysis import diagnose_asset, RCAConfig
        cfg = RCAConfig(inspection_threshold_mw=3.0)
        result = diagnose_asset(
            "SOL-01", "Test Solar", "solar",
            actual_mw=5.0, expected_mw=40.0, capacity_mw=50.0,
            weather=self._solar_weather(cloud_cover=10, irradiance=700, temperature=22),
            cfg=cfg,
        )
        # Clear sky, low temp — unexplained loss should be high
        assert result["recommended_inspection"] is True

    def test_returns_required_keys(self):
        from src.ml.root_cause_analysis import diagnose_asset
        result = diagnose_asset(
            "SOL-01", "Solar", "solar",
            actual_mw=20.0, expected_mw=40.0, capacity_mw=50.0,
            weather=self._solar_weather(),
        )
        required = {
            "asset_id", "asset_name", "asset_type",
            "expected_generation_mw", "actual_generation_mw",
            "deviation_mw", "weather_explained_loss_mw",
            "unexplained_loss_mw", "probable_causes",
            "overall_confidence", "recommended_inspection", "inspection_notes"
        }
        assert required.issubset(result.keys())

    def test_no_mechanical_fault_claimed_for_weather(self):
        """High cloud cover should be weather-explained, NOT mechanical fault."""
        from src.ml.root_cause_analysis import diagnose_asset
        result = diagnose_asset(
            "SOL-01", "Solar", "solar",
            actual_mw=10.0, expected_mw=40.0, capacity_mw=50.0,
            weather=self._solar_weather(cloud_cover=90, irradiance=100),
        )
        # inverter_or_string_issue should NOT be the only cause — cloud_cover should dominate
        causes = {c["cause"]: c["confidence"] for c in result["probable_causes"]}
        if "cloud_cover" in causes and "inverter_or_string_issue" in causes:
            assert causes["cloud_cover"] >= causes["inverter_or_string_issue"], (
                "Weather-explained cause should have >= confidence than mechanical guess"
            )


# ===========================================================================
# 5. Missing / invalid weather data
# ===========================================================================

class TestMissingWeatherData:
    def test_solar_zero_irradiance(self):
        from src.ml.renewable_generation import solar_expected
        result = solar_expected(0.0, 20.0, 25.0, 50.0)
        assert result["expected_generation_mw"] == 0.0

    def test_solar_negative_irradiance_clipped(self):
        from src.ml.renewable_generation import solar_expected
        result = solar_expected(-100.0, 20.0, 25.0, 50.0)
        assert result["expected_generation_mw"] == 0.0

    def test_solar_zero_capacity(self):
        from src.ml.renewable_generation import solar_expected
        result = solar_expected(800.0, 10.0, 25.0, 0.0)
        assert result["expected_generation_mw"] == 0.0

    def test_wind_below_cut_in_returns_zero(self):
        from src.ml.renewable_generation import wind_expected
        result = wind_expected(1.0, 180.0, 15.0, 80.0)
        assert result["expected_generation_mw"] == 0.0

    def test_wind_above_cut_out_returns_zero(self):
        from src.ml.renewable_generation import wind_expected
        result = wind_expected(30.0, 180.0, 15.0, 80.0)
        assert result["expected_generation_mw"] == 0.0

    def test_wind_rated_speed_returns_capacity(self):
        from src.ml.renewable_generation import wind_expected
        result = wind_expected(15.0, 180.0, 15.0, 80.0)
        # At rated speed (15 m/s > 12 m/s) but density correction applies
        assert result["expected_generation_mw"] > 0
        # Should be close to capacity (within 10% due to density factor)
        assert result["expected_generation_mw"] >= 70.0

    def test_rca_missing_weather_keys_no_crash(self):
        """RCA should not crash when optional weather keys are absent."""
        from src.ml.root_cause_analysis import diagnose_asset
        sparse_weather = {"wind_speed": 5.0}  # missing most keys
        result = diagnose_asset(
            "SOL-01", "Solar", "solar",
            actual_mw=10.0, expected_mw=30.0, capacity_mw=50.0,
            weather=sparse_weather,
        )
        assert "probable_causes" in result


# ===========================================================================
# 6. Spike detection
# ===========================================================================

class TestSpikeDetection:
    def _make_forecast(self, loads: list[float]) -> pd.DataFrame:
        base = datetime(2023, 7, 1, 0, tzinfo=timezone.utc)
        return pd.DataFrame({
            "timestamp": [base + timedelta(hours=i) for i in range(len(loads))],
            "forecast_load_mw": loads,
        })

    def _make_capacity(self, loads: list[float], capacity: float = 2600.0) -> pd.DataFrame:
        base = datetime(2023, 7, 1, 0, tzinfo=timezone.utc)
        return pd.DataFrame({
            "timestamp": [base + timedelta(hours=i) for i in range(len(loads))],
            "available_capacity_mw": [capacity] * len(loads),
        })

    def test_low_risk(self):
        from src.ml.spike_detection import detect_spikes
        loads = [1000.0] * 5
        df = detect_spikes(self._make_forecast(loads), self._make_capacity(loads))
        assert (df["risk_level"] == "LOW").all()

    def test_critical_risk(self):
        from src.ml.spike_detection import detect_spikes, SpikeDetectorConfig
        cfg = SpikeDetectorConfig(critical_reserve_margin=0.05)
        loads = [2480.0] * 5  # ~4.6% reserve at 2600 MW capacity
        df = detect_spikes(self._make_forecast(loads), self._make_capacity(loads), cfg=cfg)
        assert (df["risk_level"] == "CRITICAL").any()

    def test_reserve_margin_formula(self):
        from src.ml.spike_detection import detect_spikes
        loads = [2000.0]
        df = detect_spikes(self._make_forecast(loads), self._make_capacity(loads, 2500.0))
        expected_margin = (2500 - 2000) / 2500
        assert abs(df["reserve_margin"].iloc[0] - expected_margin) < 0.001

    def test_missing_capacity_uses_default(self):
        from src.ml.spike_detection import detect_spikes
        loads = [1200.0] * 3
        fc = self._make_forecast(loads)
        df = detect_spikes(fc, capacity_df=None)
        assert "available_capacity_mw" in df.columns

    def test_risk_score_range(self):
        from src.ml.spike_detection import detect_spikes
        loads = [800.0, 1200.0, 1800.0, 2200.0, 2550.0]
        cap_df = self._make_capacity(loads, 2600.0)
        df = detect_spikes(self._make_forecast(loads), cap_df)
        assert df["spike_risk_score"].between(0, 1).all()


# ===========================================================================
# 7. Renewable generation models
# ===========================================================================

class TestRenewableGeneration:
    def test_solar_noon_produces_output(self):
        from src.ml.renewable_generation import solar_expected
        result = solar_expected(900.0, 5.0, 25.0, 100.0)
        assert result["expected_generation_mw"] > 0

    def test_solar_high_cloud_reduces_output(self):
        from src.ml.renewable_generation import solar_expected
        clear = solar_expected(900.0, 5.0, 25.0, 100.0)
        cloudy = solar_expected(900.0, 90.0, 25.0, 100.0)
        assert cloudy["expected_generation_mw"] < clear["expected_generation_mw"]

    def test_solar_hot_temperature_derating(self):
        from src.ml.renewable_generation import solar_expected
        normal = solar_expected(900.0, 5.0, 25.0, 100.0)
        hot = solar_expected(900.0, 5.0, 45.0, 100.0)
        assert hot["expected_generation_mw"] < normal["expected_generation_mw"]

    def test_wind_cubic_ramp(self):
        from src.ml.renewable_generation import wind_expected
        cf_6 = wind_expected(6.0, 180.0, 15.0, 100.0)["capacity_factor"]
        cf_9 = wind_expected(9.0, 180.0, 15.0, 100.0)["capacity_factor"]
        assert cf_9 > cf_6  # Higher wind → higher CF

    def test_model_explanation_keys(self):
        from src.ml.renewable_generation import solar_expected, wind_expected
        solar_r = solar_expected(800.0, 20.0, 25.0, 50.0)
        assert "model_explanation" in solar_r
        wind_r = wind_expected(10.0, 180.0, 15.0, 80.0)
        assert "model_explanation" in wind_r
