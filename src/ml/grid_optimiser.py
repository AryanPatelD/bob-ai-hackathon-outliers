"""
Grid Optimisation Engine — Phase 2.

For every forecast interval, computes:
  net_balance = renewable_generation + conventional_generation - demand

Deficit handling (priority order, configurable):
  1. BESS discharge
  2. Demand response / load shifting
  3. Additional / backup generation

Surplus handling (priority order, configurable):
  1. BESS charging
  2. Flexible load shifting
  3. Export (if available)
  4. Renewable curtailment (last resort)

Constraints enforced:
  - BESS SOC bounds [soc_min, soc_max]
  - BESS energy capacity (MWh)
  - BESS max charge / discharge rate (MW)
  - Reserve margin requirement
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimiserConfig:
    # Reserve constraint — minimum headroom as fraction of demand
    min_reserve_fraction: float = float(os.getenv("OPT_MIN_RESERVE", "0.10"))

    # BESS constraints
    bess_soc_min: float = float(os.getenv("BESS_SOC_MIN", "0.10"))
    bess_soc_max: float = float(os.getenv("BESS_SOC_MAX", "0.95"))

    # Demand response capacity (MW available for flexible load shifting)
    demand_response_capacity_mw: float = float(os.getenv("DR_CAPACITY_MW", "150.0"))

    # Export capacity (MW) — grid connection to export surplus
    export_capacity_mw: float = float(os.getenv("EXPORT_CAPACITY_MW", "100.0"))

    # Interval duration in hours (default 1h)
    interval_hours: float = 1.0


DEFAULT_OPT_CONFIG = OptimiserConfig()


# ---------------------------------------------------------------------------
# Per-interval state and result
# ---------------------------------------------------------------------------

@dataclass
class BESSState:
    asset_id: str
    capacity_mwh: float
    current_soc: float          # 0–1
    max_charge_mw: float
    max_discharge_mw: float

    @property
    def available_energy_mwh(self) -> float:
        """Energy available to discharge (above soc_min)."""
        return 0.0  # computed in optimiser with config

    @property
    def available_headroom_mwh(self) -> float:
        """Capacity available to absorb charge (below soc_max)."""
        return 0.0  # computed in optimiser with config


@dataclass
class IntervalResult:
    timestamp: object
    # Inputs
    demand_mw: float
    renewable_mw: float
    conventional_mw: float
    available_capacity_mw: float
    # Balance
    net_balance_mw: float           # positive = surplus, negative = deficit
    # Actions
    bess_action_mw: float           # positive = charging, negative = discharging
    demand_response_mw: float       # positive = load reduced
    backup_generation_mw: float     # activated backup
    export_mw: float                # surplus exported
    curtailment_mw: float           # unavoidable renewable curtailment
    # Outcomes
    reserve_margin: float
    bess_soc_after: float
    action_taken: str
    constraints_violated: List[str]
    explanation: str


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

def optimise_interval(
    timestamp,
    demand_mw: float,
    renewable_mw: float,
    conventional_mw: float,
    available_capacity_mw: float,
    bess_states: List[BESSState],
    cfg: OptimiserConfig,
) -> tuple[IntervalResult, List[BESSState]]:
    """
    Optimise a single time interval.

    Returns (IntervalResult, updated_bess_states).
    """
    net_balance = renewable_mw + conventional_mw - demand_mw
    # Combined BESS across all units
    total_bess_capacity_mwh = sum(b.capacity_mwh for b in bess_states)
    total_bess_soc = (
        sum(b.current_soc * b.capacity_mwh for b in bess_states) / total_bess_capacity_mwh
        if total_bess_capacity_mwh > 0 else 0.0
    )

    bess_action_mw = 0.0
    demand_response_mw = 0.0
    backup_generation_mw = 0.0
    export_mw = 0.0
    curtailment_mw = 0.0
    constraints_violated: List[str] = []
    action_parts: List[str] = []
    explanation_parts: List[str] = []

    updated_bess = [BESSState(
        b.asset_id, b.capacity_mwh, b.current_soc, b.max_charge_mw, b.max_discharge_mw
    ) for b in bess_states]

    if net_balance < 0:
        # ----------------------------------------------------------------
        # DEFICIT: need to cover abs(net_balance)
        # ----------------------------------------------------------------
        deficit = abs(net_balance)
        remaining_deficit = deficit
        explanation_parts.append(
            f"Deficit of {deficit:.1f} MW (renewable {renewable_mw:.1f} + "
            f"conventional {conventional_mw:.1f} - demand {demand_mw:.1f})."
        )

        # Priority 1: BESS discharge
        for i, bess in enumerate(bess_states):
            if remaining_deficit <= 0:
                break
            avail_energy = max(0.0, (bess.current_soc - cfg.bess_soc_min) * bess.capacity_mwh)
            max_discharge = min(bess.max_discharge_mw, avail_energy / cfg.interval_hours)
            discharge = min(remaining_deficit, max_discharge)
            if discharge > 0:
                bess_action_mw -= discharge  # negative = discharging
                energy_used = discharge * cfg.interval_hours
                new_soc = bess.current_soc - energy_used / bess.capacity_mwh
                updated_bess[i].current_soc = max(cfg.bess_soc_min, new_soc)
                remaining_deficit -= discharge
                action_parts.append(f"BESS {bess.asset_id} discharge {discharge:.1f} MW")
                explanation_parts.append(
                    f"BESS {bess.asset_id}: discharging {discharge:.1f} MW "
                    f"(SOC {bess.current_soc:.2f} -> {updated_bess[i].current_soc:.2f})."
                )

        # Priority 2: Demand response
        if remaining_deficit > 0:
            dr = min(remaining_deficit, cfg.demand_response_capacity_mw)
            demand_response_mw = dr
            remaining_deficit -= dr
            action_parts.append(f"Demand response {dr:.1f} MW")
            explanation_parts.append(
                f"Activating {dr:.1f} MW demand response / load shifting to cover deficit."
            )

        # Priority 3: Backup generation
        if remaining_deficit > 0:
            backup_generation_mw = remaining_deficit
            remaining_deficit = 0.0
            action_parts.append(f"Backup generation {backup_generation_mw:.1f} MW")
            explanation_parts.append(
                f"Calling {backup_generation_mw:.1f} MW backup / peaking generation."
            )
            constraints_violated.append("backup_generation_required")

        action_taken = "DEFICIT: " + "; ".join(action_parts) if action_parts else "DEFICIT: no action available"

    elif net_balance > 0:
        # ----------------------------------------------------------------
        # SURPLUS: prioritise absorbing before curtailing
        # ----------------------------------------------------------------
        surplus = net_balance
        remaining_surplus = surplus
        explanation_parts.append(
            f"Surplus of {surplus:.1f} MW (renewable {renewable_mw:.1f} + "
            f"conventional {conventional_mw:.1f} - demand {demand_mw:.1f})."
        )

        # Priority 1: BESS charging
        for i, bess in enumerate(bess_states):
            if remaining_surplus <= 0:
                break
            headroom_energy = max(0.0, (cfg.bess_soc_max - bess.current_soc) * bess.capacity_mwh)
            max_charge = min(bess.max_charge_mw, headroom_energy / cfg.interval_hours)
            charge = min(remaining_surplus, max_charge)
            if charge > 0:
                bess_action_mw += charge  # positive = charging
                energy_added = charge * cfg.interval_hours
                new_soc = bess.current_soc + energy_added / bess.capacity_mwh
                updated_bess[i].current_soc = min(cfg.bess_soc_max, new_soc)
                remaining_surplus -= charge
                action_parts.append(f"BESS {bess.asset_id} charge {charge:.1f} MW")
                explanation_parts.append(
                    f"BESS {bess.asset_id}: charging {charge:.1f} MW "
                    f"(SOC {bess.current_soc:.2f} -> {updated_bess[i].current_soc:.2f})."
                )

        # Priority 2: Flexible load shifting (increase load to absorb)
        if remaining_surplus > 0:
            flex_absorb = min(remaining_surplus, cfg.demand_response_capacity_mw * 0.5)
            if flex_absorb > 0:
                demand_response_mw = -flex_absorb  # negative = load increase
                remaining_surplus -= flex_absorb
                action_parts.append(f"Flexible load absorption {flex_absorb:.1f} MW")
                explanation_parts.append(
                    f"Shifting {flex_absorb:.1f} MW to flexible loads (EV charging, industrial)."
                )

        # Priority 3: Export
        if remaining_surplus > 0:
            export = min(remaining_surplus, cfg.export_capacity_mw)
            export_mw = export
            remaining_surplus -= export
            action_parts.append(f"Export {export:.1f} MW")
            explanation_parts.append(f"Exporting {export:.1f} MW to interconnected grid.")

        # Priority 4: Curtailment (last resort)
        if remaining_surplus > 0:
            curtailment_mw = remaining_surplus
            action_parts.append(f"Curtail {curtailment_mw:.1f} MW")
            explanation_parts.append(
                f"Curtailing {curtailment_mw:.1f} MW renewable generation (no other absorption available)."
            )

        action_taken = "SURPLUS: " + "; ".join(action_parts) if action_parts else "SURPLUS: balanced"

    else:
        action_taken = "BALANCED"
        explanation_parts.append("Supply and demand are balanced. No action required.")

    # Recompute effective demand after DR
    effective_demand = demand_mw - demand_response_mw  # DR can increase or decrease
    effective_supply = renewable_mw + conventional_mw + backup_generation_mw + abs(min(0, bess_action_mw))
    reserve_margin = (available_capacity_mw - effective_demand) / available_capacity_mw if available_capacity_mw > 0 else 0.0

    # Aggregate BESS SOC
    new_total_soc = (
        sum(b.current_soc * b.capacity_mwh for b in updated_bess) / total_bess_capacity_mwh
        if total_bess_capacity_mwh > 0 else total_bess_soc
    )

    return IntervalResult(
        timestamp=timestamp,
        demand_mw=round(demand_mw, 2),
        renewable_mw=round(renewable_mw, 2),
        conventional_mw=round(conventional_mw, 2),
        available_capacity_mw=round(available_capacity_mw, 2),
        net_balance_mw=round(net_balance, 2),
        bess_action_mw=round(bess_action_mw, 2),
        demand_response_mw=round(demand_response_mw, 2),
        backup_generation_mw=round(backup_generation_mw, 2),
        export_mw=round(export_mw, 2),
        curtailment_mw=round(curtailment_mw, 2),
        reserve_margin=round(reserve_margin, 4),
        bess_soc_after=round(new_total_soc, 4),
        action_taken=action_taken,
        constraints_violated=constraints_violated,
        explanation=" ".join(explanation_parts),
    ), updated_bess


def run_optimisation(
    forecast_df: pd.DataFrame,
    bess_df: pd.DataFrame,
    cfg: Optional[OptimiserConfig] = None,
) -> pd.DataFrame:
    """
    Run optimisation over a full forecast horizon.

    Parameters
    ----------
    forecast_df : DataFrame with columns:
        timestamp, demand_mw, renewable_mw, conventional_mw, available_capacity_mw
    bess_df : DataFrame with latest BESS state per asset_id:
        asset_id, capacity_mwh, current_soc, max_charge_mw, max_discharge_mw
    cfg : OptimiserConfig

    Returns
    -------
    DataFrame of IntervalResult records.
    """
    if cfg is None:
        cfg = DEFAULT_OPT_CONFIG

    # Build initial BESS states
    bess_states: List[BESSState] = []
    latest_bess = bess_df.sort_values("timestamp").groupby("asset_id").last().reset_index()
    for _, row in latest_bess.iterrows():
        bess_states.append(BESSState(
            asset_id=row["asset_id"],
            capacity_mwh=float(row["capacity_mwh"]),
            current_soc=float(row["current_soc"]),
            max_charge_mw=float(row["max_charge_mw"]),
            max_discharge_mw=float(row["max_discharge_mw"]),
        ))

    results = []
    for _, row in forecast_df.iterrows():
        result, bess_states = optimise_interval(
            timestamp=row["timestamp"],
            demand_mw=float(row["demand_mw"]),
            renewable_mw=float(row.get("renewable_mw", 0.0)),
            conventional_mw=float(row.get("conventional_mw", 0.0)),
            available_capacity_mw=float(row.get("available_capacity_mw", 2600.0)),
            bess_states=bess_states,
            cfg=cfg,
        )
        results.append(result.__dict__)

    return pd.DataFrame(results)
