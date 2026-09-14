"""
End-to-end pipeline audit script for GridWise AI Phase 3.
Tests the complete journey: Data -> Forecast -> Detection -> RCA -> Optimisation -> Brief
"""
import sys
sys.path.insert(0, '.')

import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

PASS = []
FAIL = []

def check(name, fn):
    try:
        result = fn()
        PASS.append(name)
        return result
    except Exception as e:
        FAIL.append((name, str(e)))
        traceback.print_exc()
        return None

# ── 1. Data loading ──────────────────────────────────────────────────────────
def test_data():
    from src.services.data_store import load_grid, load_weather, load_assets, load_bess
    grid = load_grid()
    weather = load_weather()
    assets = load_assets()
    bess = load_bess()
    assert len(grid) > 0
    assert len(weather) > 0
    assert len(assets) > 0
    assert len(bess) > 0
    assert 'timestamp' in grid.columns
    assert 'grid_load_mw' in grid.columns
    assert 'irradiance' in weather.columns
    assert 'wind_speed' in weather.columns
    return {'grid': len(grid), 'weather': len(weather), 'assets': len(assets), 'bess': len(bess)}

r = check("1. Data Loading", test_data)
if r: print(f"   Grid: {r['grid']}h  Weather: {r['weather']}h  Assets: {r['assets']}  BESS: {r['bess']}")

# ── 2. Load Forecasting ──────────────────────────────────────────────────────
def test_forecast():
    from src.services.data_store import load_grid, load_weather, get_or_train_model
    from src.ml.load_forecasting import forecast_24h
    model, metrics = get_or_train_model()
    assert model is not None
    grid = load_grid()
    weather = load_weather()
    fc = forecast_24h(model, grid, weather)
    assert len(fc) == 24
    assert 'forecast_load_mw' in fc.columns
    assert fc['forecast_load_mw'].between(100, 3000).all(), "Forecast values out of plausible range"
    assert not fc['forecast_load_mw'].isna().any(), "NaN in forecast"
    return {'points': len(fc), 'peak': fc['forecast_load_mw'].max(), 'r2_test': metrics.get('test',{}).get('r2','?')}

r = check("2. Load Forecasting", test_forecast)
if r: print(f"   24h forecast: peak={r['peak']:.1f} MW  R2_test={r['r2_test']:.3f}" if isinstance(r.get('r2_test'),float) else f"   24h forecast ok: {r['points']} points")

# ── 3. Spike Detection ───────────────────────────────────────────────────────
def test_spike():
    from src.services.data_store import load_grid, load_weather, get_or_train_model
    from src.ml.load_forecasting import forecast_24h
    from src.ml.spike_detection import detect_spikes, SpikeDetectorConfig
    model, _ = get_or_train_model()
    grid = load_grid()
    weather = load_weather()
    fc = forecast_24h(model, grid, weather)
    fc_renamed = fc.rename(columns={'forecast_load_mw': 'forecast_load_mw'})
    risk_df = detect_spikes(fc, grid)
    assert 'risk_level' in risk_df.columns
    assert 'reserve_margin' in risk_df.columns
    assert 'spike_risk_score' in risk_df.columns
    assert risk_df['spike_risk_score'].between(0, 1).all()
    valid_levels = {'LOW','MEDIUM','HIGH','CRITICAL'}
    assert set(risk_df['risk_level'].unique()).issubset(valid_levels)
    return risk_df['risk_level'].value_counts().to_dict()

r = check("3. Spike Detection", test_spike)
if r: print(f"   Risk levels: {r}")

# ── 4. Renewable Forecast ────────────────────────────────────────────────────
def test_renewable_forecast():
    from src.ml.renewable_generation import solar_expected, wind_expected
    # Solar — midday clear sky
    s = solar_expected(900.0, 5.0, 25.0, 100.0)
    assert 0 < s['expected_generation_mw'] <= 100.0
    assert 0 < s['capacity_factor'] <= 1.0
    # Wind — rated speed
    w = wind_expected(12.0, 180.0, 15.0, 80.0)
    assert 0 < w['expected_generation_mw'] <= 80.0
    # Night solar
    n = solar_expected(0.0, 0.0, 20.0, 100.0)
    assert n['expected_generation_mw'] == 0.0
    # Cut-in wind
    c = wind_expected(1.0, 180.0, 15.0, 80.0)
    assert c['expected_generation_mw'] == 0.0
    return {'solar_midday': s['expected_generation_mw'], 'wind_rated': w['expected_generation_mw']}

r = check("4. Renewable Forecast", test_renewable_forecast)
if r: print(f"   Solar midday: {r['solar_midday']:.1f} MW  Wind rated: {r['wind_rated']:.1f} MW")

# ── 5. Anomaly Detection ─────────────────────────────────────────────────────
def test_anomaly():
    from src.services.data_store import load_assets
    from src.ml.anomaly_detection import detect_anomalies, anomaly_summary, AnomalyConfig
    assets = load_assets()
    # Use last 24 hours
    cutoff = assets['timestamp'].max() - pd.Timedelta(hours=24)
    window = assets[assets['timestamp'] >= cutoff]
    anomaly_df = detect_anomalies(window)
    assert 'severity' in anomaly_df.columns
    assert 'performance_ratio' in anomaly_df.columns
    assert set(anomaly_df['severity'].unique()).issubset({'NORMAL','WARNING','CRITICAL'})
    # Night check: expected ~0 solar at midnight should be NORMAL
    night_row = window[(window['asset_type']=='solar') & (window['timestamp'].dt.hour.isin([0,1,2,3]))].copy()
    if not night_row.empty:
        night_anomaly = detect_anomalies(night_row)
        night_crits = night_anomaly[night_anomaly['severity']=='CRITICAL']
        assert len(night_crits) == 0, f"Night solar flagged CRITICAL: {len(night_crits)} records"
    summary = anomaly_summary(anomaly_df)
    return summary

r = check("5. Anomaly Detection", test_anomaly)
if r: print(f"   Assets: {r['total_assets']} | Warning: {r['assets_in_warning']} | Critical: {r['assets_in_critical']}")

# ── 6. Root Cause Analysis ───────────────────────────────────────────────────
def test_rca():
    from src.ml.root_cause_analysis import diagnose_asset
    # Heavy cloud — weather should explain most
    r = diagnose_asset('SOL-01','Solar A','solar', actual_mw=8.0, expected_mw=40.0, capacity_mw=50.0,
        weather={'cloud_cover':90,'irradiance':80,'temperature':22,'wind_speed':5,'wind_direction':180})
    assert r['weather_explained_loss_mw'] >= r['unexplained_loss_mw']
    # Clear sky fault
    r2 = diagnose_asset('SOL-01','Solar A','solar', actual_mw=5.0, expected_mw=45.0, capacity_mw=50.0,
        weather={'cloud_cover':5,'irradiance':900,'temperature':25,'wind_speed':5,'wind_direction':180})
    assert r2['recommended_inspection'] is True
    # Wind cut-in
    r3 = diagnose_asset('WIN-01','Wind A','wind', actual_mw=0.0, expected_mw=2.0, capacity_mw=80.0,
        weather={'cloud_cover':10,'irradiance':0,'temperature':15,'wind_speed':1.5,'wind_direction':200})
    assert r3['unexplained_loss_mw'] == 0.0
    required = {'asset_id','expected_generation_mw','actual_generation_mw','deviation_mw',
                'weather_explained_loss_mw','unexplained_loss_mw','probable_causes',
                'overall_confidence','recommended_inspection','inspection_notes'}
    assert required.issubset(r.keys())
    return {'causes_found': len(r['probable_causes']), 'inspection': r2['recommended_inspection']}

r = check("6. Root Cause Analysis", test_rca)
if r: print(f"   Causes found: {r['causes_found']}  Inspection recommended on fault: {r['inspection']}")

# ── 7. Grid Optimisation ─────────────────────────────────────────────────────
def test_optimisation():
    from src.data.scenarios import scenario_b_demand_spike, scenario_c_surplus_and_anomaly
    from src.ml.grid_optimiser import run_optimisation, OptimiserConfig
    # Scenario B: should trigger BESS + DR + backup
    ctx_b = scenario_b_demand_spike()
    opt_b = run_optimisation(ctx_b['forecast_df'], ctx_b['bess_df'])
    assert (opt_b['net_balance_mw'] < 0).any(), "No deficit in spike scenario"
    assert (opt_b['backup_generation_mw'] > 0).any(), "No backup in spike scenario"
    # Scenario C: should have surplus + curtailment
    ctx_c = scenario_c_surplus_and_anomaly()
    opt_c = run_optimisation(ctx_c['forecast_df'], ctx_c['bess_df'])
    # Constraint: SOC never below min
    cfg = OptimiserConfig()
    assert (opt_c['bess_soc_after'] >= cfg.bess_soc_min - 0.001).all(), "SOC violated"
    return {
        'b_deficits': int((opt_b['net_balance_mw']<0).sum()),
        'b_backup_hours': int((opt_b['backup_generation_mw']>0).sum()),
        'c_surpluses': int((opt_c['net_balance_mw']>0).sum()),
    }

r = check("7. Grid Optimisation", test_optimisation)
if r: print(f"   B deficits={r['b_deficits']}h backup={r['b_backup_hours']}h | C surpluses={r['c_surpluses']}h")

# ── 8. Curtailment Minimisation ──────────────────────────────────────────────
def test_curtailment():
    from src.data.scenarios import scenario_c_surplus_and_anomaly
    from src.ml.grid_optimiser import run_optimisation
    from src.ml.curtailment import compute_curtailment_analysis
    ctx = scenario_c_surplus_and_anomaly()
    opt = run_optimisation(ctx['forecast_df'], ctx['bess_df'])
    c = compute_curtailment_analysis(opt)
    s = c['summary']
    assert s['total_potential_curtailment_mwh'] >= 0
    assert s['curtailment_reduction_pct'] >= 0
    assert abs((s['total_avoided_curtailment_mwh'] + s['total_unavoidable_curtailment_mwh'])
               - s['total_potential_curtailment_mwh']) < 0.5, "Curtailment accounting mismatch"
    assert len(c['explanation']) > 10, "No curtailment explanation generated"
    return s

r = check("8. Curtailment Minimisation", test_curtailment)
if r: print(f"   Potential={r['total_potential_curtailment_mwh']:.1f}  Avoided={r['total_avoided_curtailment_mwh']:.1f}  Reduction={r['curtailment_reduction_pct']:.1f}%")

# ── 9. Operator Brief ────────────────────────────────────────────────────────
def test_brief():
    from src.data.scenarios import scenario_b_demand_spike
    from src.ml.grid_optimiser import run_optimisation
    from src.ml.curtailment import compute_curtailment_analysis
    from src.ml.operator_brief import generate_operator_brief
    ctx = scenario_b_demand_spike()
    opt = run_optimisation(ctx['forecast_df'], ctx['bess_df'])
    curt = compute_curtailment_analysis(opt)
    brief = generate_operator_brief(
        grid_status={'timestamp':'2023-07-15','grid_load_mw':2150,'available_capacity_mw':2600,
                     'conventional_generation_mw':1400,'reserve_margin_pct':17.3,'overall_risk':'HIGH'},
        load_forecast={'horizon_hours':24,'forecast':[{'timestamp':'2023-07-15 18:00','forecast_load_mw':2150}]},
        spike_risks={'risk_summary':[{'timestamp':'2023-07-15 18:00','risk_level':'HIGH','forecast_load_mw':2150,'reserve_margin':0.173,'spike_risk_score':0.42}],
                     'thresholds':{'critical_reserve_margin':0.05,'high_reserve_margin':0.10,'medium_reserve_margin':0.20}},
        renewable_forecast={'forecasts':[{'asset_type':'wind','asset_id':'WIN-01','expected_generation_mw':5,'capacity_mw':200}]},
        anomalies={'anomalies':[],'total_assets':5,'assets_in_warning':0,'assets_in_critical':0},
        diagnoses=[],
        optimisation_plan={'results':opt.to_dict(orient='records'),'summary':{}},
        curtailment_plan=curt,
    )
    required_sections = {'GRID STATUS','DEMAND FORECAST','DEMAND SPIKE RISKS','RECOMMENDED ACTIONS',
                         'BESS PLAN','CURTAILMENT MINIMISATION PLAN','CONFIDENCE / DATA QUALITY'}
    assert required_sections.issubset(brief['sections'].keys())
    assert len(brief['full_text']) > 500
    return {'sections': len(brief['sections']), 'chars': len(brief['full_text'])}

r = check("9. Operator Brief", test_brief)
if r: print(f"   Sections: {r['sections']}  Total: {r['chars']} chars")

# ── 10. API route imports ────────────────────────────────────────────────────
def test_api_imports():
    from src.backend.main import app
    routes = {r.path for r in app.routes if hasattr(r,'methods')}
    required = {'/health','/assets','/assets/anomalies','/assets/{asset_id}/diagnosis',
                '/forecast/load','/forecast/renewables','/grid/risk',
                '/optimisation/plan','/optimisation/curtailment','/brief','/scenarios','/dashboard','/'}
    missing = required - routes
    assert not missing, f"Missing routes: {missing}"
    return len(routes)

r = check("10. API Route Registration", test_api_imports)
if r: print(f"   {r} routes registered")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  AUDIT RESULTS: {len(PASS)} PASSED  {len(FAIL)} FAILED")
print("=" * 60)
for name in PASS:
    print(f"  PASS  {name}")
for name, err in FAIL:
    print(f"  FAIL  {name}: {err}")
