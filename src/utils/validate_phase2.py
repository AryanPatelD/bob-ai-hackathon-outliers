import sys
sys.path.insert(0, '.')
from src.backend.main import app

print('=== All Routes ===')
for route in app.routes:
    if hasattr(route, 'methods') and route.path not in ['/openapi.json', '/docs', '/docs/oauth2-redirect', '/redoc']:
        for method in route.methods:
            print(f'  {method:6s} {route.path}')

print()
print('=== Scenario Smoke Test ===')
from src.data.scenarios import scenario_a_normal, scenario_b_demand_spike, scenario_c_surplus_and_anomaly
from src.ml.grid_optimiser import run_optimisation
from src.ml.curtailment import compute_curtailment_analysis

scenarios = [('A-Normal', scenario_a_normal), ('B-Spike', scenario_b_demand_spike), ('C-Surplus', scenario_c_surplus_and_anomaly)]
for name, fn in scenarios:
    ctx = fn()
    opt = run_optimisation(ctx['forecast_df'], ctx['bess_df'])
    curt = compute_curtailment_analysis(opt)
    deficits = int((opt['net_balance_mw'] < 0).sum())
    surpluses = int((opt['net_balance_mw'] > 0).sum())
    backup = int((opt['backup_generation_mw'] > 0).sum())
    avoided = curt['summary']['total_avoided_curtailment_mwh']
    print(f'  Scenario {name}: {len(opt)}h | deficits={deficits} surpluses={surpluses} backup={backup}h | avoided_curtailment={avoided:.1f}MWh')

print()
print('=== Operator Brief smoke (scenario C) ===')
from src.ml.operator_brief import generate_operator_brief
brief = generate_operator_brief(
    grid_status={'timestamp': '2023-07-15 12:00', 'grid_load_mw': 700, 'available_capacity_mw': 2600,
                 'conventional_generation_mw': 200, 'reserve_margin_pct': 73.1, 'overall_risk': 'LOW'},
    load_forecast={'horizon_hours': 24, 'forecast': [{'timestamp': '2023-07-15 12:00', 'forecast_load_mw': 700}]},
    spike_risks={'risk_summary': [{'timestamp': '2023-07-15 12:00', 'risk_level': 'LOW', 'forecast_load_mw': 700,
                                   'reserve_margin': 0.73, 'spike_risk_score': 0.09}], 'thresholds': {'critical_reserve_margin': 0.05, 'high_reserve_margin': 0.10, 'medium_reserve_margin': 0.20}},
    renewable_forecast={'forecasts': [{'asset_type': 'solar', 'asset_id': 'SOL-01', 'expected_generation_mw': 120.0, 'capacity_mw': 225.0},
                                       {'asset_type': 'wind', 'asset_id': 'WIN-01', 'expected_generation_mw': 200.0, 'capacity_mw': 200.0}]},
    anomalies={'anomalies': [], 'total_assets': 5, 'assets_in_warning': 0, 'assets_in_critical': 0},
    diagnoses=[],
    optimisation_plan={'results': opt.to_dict(orient='records'), 'summary': {}},
    curtailment_plan=curt,
)
print('  Brief sections:', list(brief['sections'].keys()))
print('  Full text length:', len(brief['full_text']), 'chars')
print()
print('All validation passed.')
