"""
Integrated Operator Brief Generator — Phase 2.

Assembles a structured, human-readable brief for grid operators from:
  - grid status
  - load forecast + spike risks
  - renewable forecast + anomalies
  - root cause diagnoses
  - optimisation plan
  - curtailment analysis
  - BESS plan
  - demand response plan

Sections:
  GRID STATUS | DEMAND FORECAST | PEAK DEMAND WINDOW | DEMAND SPIKE RISKS |
  RENEWABLE FORECAST | UNDERPERFORMING ASSETS | ROOT CAUSES |
  SUPPLY/DEMAND GAP | RECOMMENDED ACTIONS | BESS PLAN |
  DEMAND RESPONSE PLAN | CURTAILMENT MINIMISATION PLAN |
  CONFIDENCE / DATA QUALITY | KNOWN UNCERTAINTIES
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _ts(ts) -> str:
    """Format a timestamp for display."""
    try:
        if hasattr(ts, "strftime"):
            return ts.strftime("%Y-%m-%d %H:%M UTC")
        return str(ts)
    except Exception:
        return str(ts)


def _risk_badge(level: str) -> str:
    labels = {"LOW": "[LOW]", "MEDIUM": "[MEDIUM]", "HIGH": "[HIGH]", "CRITICAL": "[CRITICAL]"}
    return labels.get(level.upper(), f"[{level}]")


def generate_operator_brief(
    grid_status: dict,
    load_forecast: dict,
    spike_risks: dict,
    renewable_forecast: dict,
    anomalies: dict,
    diagnoses: List[dict],
    optimisation_plan: dict,
    curtailment_plan: dict,
    generated_at: Optional[datetime] = None,
) -> dict:
    """
    Generate a full structured operator brief.

    All inputs are dicts as returned by their respective service functions.
    Returns a dict with a 'sections' list and a 'full_text' string.
    """
    if generated_at is None:
        generated_at = datetime.now(tz=timezone.utc)

    sections: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. GRID STATUS
    # ------------------------------------------------------------------
    gs = grid_status
    sections["GRID STATUS"] = (
        f"Timestamp      : {_ts(gs.get('timestamp', 'N/A'))}\n"
        f"Grid Load      : {gs.get('grid_load_mw', 'N/A'):.1f} MW\n"
        f"Available Cap  : {gs.get('available_capacity_mw', 'N/A'):.1f} MW\n"
        f"Conventional   : {gs.get('conventional_generation_mw', 'N/A'):.1f} MW\n"
        f"Reserve Margin : {gs.get('reserve_margin_pct', 0):.1f}%\n"
        f"Overall Risk   : {_risk_badge(gs.get('overall_risk', 'UNKNOWN'))}"
    )

    # ------------------------------------------------------------------
    # 2. DEMAND FORECAST
    # ------------------------------------------------------------------
    fc_points = load_forecast.get("forecast", [])
    if fc_points:
        loads = [p.get("forecast_load_mw", 0) for p in fc_points]
        peak_load = max(loads)
        min_load = min(loads)
        peak_ts = fc_points[loads.index(peak_load)].get("timestamp", "N/A")
        sections["DEMAND FORECAST"] = (
            f"Horizon        : {load_forecast.get('horizon_hours', 24)} hours\n"
            f"Peak Forecast  : {peak_load:.1f} MW at {_ts(peak_ts)}\n"
            f"Minimum Load   : {min_load:.1f} MW\n"
            f"Average Load   : {sum(loads)/len(loads):.1f} MW"
        )
    else:
        sections["DEMAND FORECAST"] = "No forecast data available."

    # ------------------------------------------------------------------
    # 3. PEAK DEMAND WINDOW
    # ------------------------------------------------------------------
    if fc_points:
        high_loads = [(p.get("timestamp"), p.get("forecast_load_mw", 0))
                      for p in fc_points if p.get("forecast_load_mw", 0) > peak_load * 0.90]
        if high_loads:
            peak_start = _ts(high_loads[0][0])
            peak_end = _ts(high_loads[-1][0])
            sections["PEAK DEMAND WINDOW"] = (
                f"Window         : {peak_start} to {peak_end}\n"
                f"Duration       : {len(high_loads)} hours above 90% of peak\n"
                f"Peak           : {peak_load:.1f} MW"
            )
        else:
            sections["PEAK DEMAND WINDOW"] = "No pronounced peak window detected."
    else:
        sections["PEAK DEMAND WINDOW"] = "Insufficient data."

    # ------------------------------------------------------------------
    # 4. DEMAND SPIKE RISKS
    # ------------------------------------------------------------------
    risk_summary = spike_risks.get("risk_summary", [])
    if risk_summary:
        critical = [r for r in risk_summary if r.get("risk_level") == "CRITICAL"]
        high = [r for r in risk_summary if r.get("risk_level") == "HIGH"]
        lines = [
            f"Critical Periods : {len(critical)}",
            f"High Risk Periods: {len(high)}",
        ]
        if critical:
            lines.append(f"First Critical   : {_ts(critical[0].get('timestamp'))} "
                         f"({critical[0].get('forecast_load_mw', 0):.1f} MW, "
                         f"reserve {critical[0].get('reserve_margin', 0)*100:.1f}%)")
        sections["DEMAND SPIKE RISKS"] = "\n".join(lines)
    else:
        sections["DEMAND SPIKE RISKS"] = "No spike risk data available."

    # ------------------------------------------------------------------
    # 5. RENEWABLE FORECAST
    # ------------------------------------------------------------------
    rf = renewable_forecast.get("forecasts", [])
    if rf:
        solar = [p for p in rf if p.get("asset_type") == "solar"]
        wind = [p for p in rf if p.get("asset_type") == "wind"]
        total_solar = sum(p.get("expected_generation_mw", 0) for p in solar)
        total_wind = sum(p.get("expected_generation_mw", 0) for p in wind)
        sections["RENEWABLE FORECAST"] = (
            f"Solar Expected  : {total_solar:.1f} MW total across {len(set(p['asset_id'] for p in solar))} assets\n"
            f"Wind Expected   : {total_wind:.1f} MW total across {len(set(p['asset_id'] for p in wind))} assets\n"
            f"Combined Total  : {total_solar + total_wind:.1f} MW"
        )
    else:
        sections["RENEWABLE FORECAST"] = "No renewable forecast data available."

    # ------------------------------------------------------------------
    # 6. UNDERPERFORMING ASSETS
    # ------------------------------------------------------------------
    anomaly_list = anomalies.get("anomalies", [])
    warn_crit = [a for a in anomaly_list if a.get("severity") in ("WARNING", "CRITICAL")]
    if warn_crit:
        lines = [f"{'Asset':<20} {'Type':<8} {'Severity':<10} {'Perf Ratio':<12} {'Dev MW':>8}"]
        lines.append("-" * 62)
        for a in warn_crit[:10]:  # cap at 10 for brief
            lines.append(
                f"{a.get('asset_name','?'):<20} {a.get('asset_type','?'):<8} "
                f"{a.get('severity','?'):<10} {a.get('performance_ratio',0):<12.2f} "
                f"{a.get('absolute_deviation_mw',0):>8.1f}"
            )
        sections["UNDERPERFORMING ASSETS"] = "\n".join(lines)
    else:
        sections["UNDERPERFORMING ASSETS"] = "All assets performing within expected range."

    # ------------------------------------------------------------------
    # 7. ROOT CAUSES
    # ------------------------------------------------------------------
    if diagnoses:
        lines = []
        for d in diagnoses[:5]:  # top 5 diagnoses
            causes = d.get("probable_causes", [])
            top_cause = causes[0]["cause"] if causes else "none identified"
            lines.append(
                f"  {d.get('asset_name','?')}: unexplained loss {d.get('unexplained_loss_mw',0):.1f} MW | "
                f"top cause: {top_cause} (conf {causes[0].get('confidence',0):.0%})"
                if causes else
                f"  {d.get('asset_name','?')}: unexplained loss {d.get('unexplained_loss_mw',0):.1f} MW | "
                f"no causes identified"
            )
            if d.get("recommended_inspection"):
                lines.append(f"    --> INSPECTION RECOMMENDED: {d.get('inspection_notes','')}")
        sections["ROOT CAUSES"] = "\n".join(lines) if lines else "No root cause analysis available."
    else:
        sections["ROOT CAUSES"] = "No anomalies requiring root cause analysis."

    # ------------------------------------------------------------------
    # 8. SUPPLY/DEMAND GAP
    # ------------------------------------------------------------------
    opt_results = optimisation_plan.get("results", [])
    if opt_results:
        deficits = [r for r in opt_results if r.get("net_balance_mw", 0) < 0]
        surpluses = [r for r in opt_results if r.get("net_balance_mw", 0) > 0]
        max_deficit = min((r.get("net_balance_mw", 0) for r in opt_results), default=0)
        max_surplus = max((r.get("net_balance_mw", 0) for r in opt_results), default=0)
        sections["SUPPLY/DEMAND GAP"] = (
            f"Deficit Periods : {len(deficits)} hours\n"
            f"Surplus Periods : {len(surpluses)} hours\n"
            f"Worst Deficit   : {max_deficit:.1f} MW\n"
            f"Peak Surplus    : {max_surplus:.1f} MW"
        )
    else:
        sections["SUPPLY/DEMAND GAP"] = "No optimisation data available."

    # ------------------------------------------------------------------
    # 9. RECOMMENDED ACTIONS
    # ------------------------------------------------------------------
    if opt_results:
        actions = {}
        for r in opt_results:
            action = r.get("action_taken", "")
            if "BESS" in action:
                actions["bess"] = actions.get("bess", 0) + 1
            if "Demand response" in action or "response" in action.lower():
                actions["demand_response"] = actions.get("demand_response", 0) + 1
            if "Backup" in action or "backup" in action.lower():
                actions["backup"] = actions.get("backup", 0) + 1
            if "Curtail" in action:
                actions["curtailment"] = actions.get("curtailment", 0) + 1
        lines = []
        if actions.get("bess", 0) > 0:
            lines.append(f"  BESS dispatch activated for {actions['bess']} intervals")
        if actions.get("demand_response", 0) > 0:
            lines.append(f"  Demand response / load shifting for {actions['demand_response']} intervals")
        if actions.get("backup", 0) > 0:
            lines.append(f"  [ALERT] Backup generation required for {actions['backup']} intervals")
        if actions.get("curtailment", 0) > 0:
            lines.append(f"  Renewable curtailment for {actions['curtailment']} intervals")
        sections["RECOMMENDED ACTIONS"] = "\n".join(lines) if lines else "No actions required — grid balanced."
    else:
        sections["RECOMMENDED ACTIONS"] = "No action data available."

    # ------------------------------------------------------------------
    # 10. BESS PLAN
    # ------------------------------------------------------------------
    if opt_results:
        bess_charge = sum(r.get("bess_action_mw", 0) for r in opt_results if r.get("bess_action_mw", 0) > 0)
        bess_discharge = sum(abs(r.get("bess_action_mw", 0)) for r in opt_results if r.get("bess_action_mw", 0) < 0)
        soc_values = [r.get("bess_soc_after", 0) for r in opt_results if r.get("bess_soc_after") is not None]
        sections["BESS PLAN"] = (
            f"Total Charging  : {bess_charge:.1f} MWh\n"
            f"Total Discharge : {bess_discharge:.1f} MWh\n"
            f"Final SOC       : {soc_values[-1]*100:.1f}% (started: {soc_values[0]*100:.1f}%)\n"
            f"Net BESS Energy : {bess_charge - bess_discharge:+.1f} MWh"
        )
    else:
        sections["BESS PLAN"] = "No BESS data available."

    # ------------------------------------------------------------------
    # 11. DEMAND RESPONSE PLAN
    # ------------------------------------------------------------------
    if opt_results:
        dr_intervals = [r for r in opt_results if abs(r.get("demand_response_mw", 0)) > 0]
        total_dr = sum(abs(r.get("demand_response_mw", 0)) for r in dr_intervals)
        sections["DEMAND RESPONSE PLAN"] = (
            f"DR Activations  : {len(dr_intervals)} intervals\n"
            f"Total DR Volume : {total_dr:.1f} MW-intervals\n"
            f"Purpose         : {'Load reduction during deficit; load increase during surplus' if dr_intervals else 'N/A'}"
        )
    else:
        sections["DEMAND RESPONSE PLAN"] = "No demand response required."

    # ------------------------------------------------------------------
    # 12. CURTAILMENT MINIMISATION PLAN
    # ------------------------------------------------------------------
    c_summary = curtailment_plan.get("summary", {})
    if c_summary:
        sections["CURTAILMENT MINIMISATION PLAN"] = (
            f"Potential Curtailment : {c_summary.get('total_potential_curtailment_mwh',0):.1f} MWh\n"
            f"BESS Absorption       : {c_summary.get('total_bess_absorption_mwh',0):.1f} MWh\n"
            f"Flex Load Absorption  : {c_summary.get('total_flexible_load_absorption_mwh',0):.1f} MWh\n"
            f"Export                : {c_summary.get('total_export_mwh',0):.1f} MWh\n"
            f"Unavoidable Curtailm. : {c_summary.get('total_unavoidable_curtailment_mwh',0):.1f} MWh\n"
            f"Avoided Curtailment   : {c_summary.get('total_avoided_curtailment_mwh',0):.1f} MWh\n"
            f"Reduction             : {c_summary.get('curtailment_reduction_pct',0):.1f}%\n"
            f"Analysis              : {curtailment_plan.get('explanation','')}"
        )
    else:
        sections["CURTAILMENT MINIMISATION PLAN"] = "No curtailment data available."

    # ------------------------------------------------------------------
    # 13. CONFIDENCE / DATA QUALITY
    # ------------------------------------------------------------------
    sections["CONFIDENCE / DATA QUALITY"] = (
        "Data Source     : SYNTHETIC / DEMO — not real telemetry\n"
        "Load Forecast   : XGBoost model (test R2=0.90) — model predictions\n"
        "Renewable Fcst  : Physics-based deterministic model — model predictions\n"
        "Anomaly Detect  : Deterministic threshold rules — rule-based results\n"
        "Root Cause      : Evidence-weighted heuristic rules — not calibrated\n"
        "Optimisation    : Deterministic priority-order dispatch — rule-based\n"
        "Note: All confidence scores are relative indicators, not probabilities."
    )

    # ------------------------------------------------------------------
    # 14. KNOWN UNCERTAINTIES
    # ------------------------------------------------------------------
    sections["KNOWN UNCERTAINTIES"] = (
        "1. Weather forecast accuracy: solar/wind output sensitive to forecast errors\n"
        "2. Demand elasticity: DR capacity assumed constant; actual response varies\n"
        "3. BESS degradation: SOC bounds assumed constant; real degradation not modelled\n"
        "4. Interconnection: export capacity treated as static limit\n"
        "5. Asset availability: maintenance schedules not included\n"
        "6. Market prices: economic dispatch not considered (Phase 3)\n"
        "7. Load forecast: no event-driven demand (holidays, extreme weather events)"
    )

    # ------------------------------------------------------------------
    # Assemble full text
    # ------------------------------------------------------------------
    lines = [
        "=" * 70,
        f"  GRIDWISE AI — OPERATOR BRIEF",
        f"  Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 70,
    ]
    for title, content in sections.items():
        lines.append(f"\n{'─' * 70}")
        lines.append(f"  {title}")
        lines.append(f"{'─' * 70}")
        for line in content.split("\n"):
            lines.append(f"  {line}")

    lines.append("\n" + "=" * 70)
    lines.append("  END OF BRIEF — GridWise AI Phase 2")
    lines.append("=" * 70)

    return {
        "generated_at": generated_at.isoformat(),
        "sections": sections,
        "full_text": "\n".join(lines),
    }
