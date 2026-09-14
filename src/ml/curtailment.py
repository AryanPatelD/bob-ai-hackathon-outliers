"""
Curtailment Minimisation Module — Phase 2.

Calculates, for each interval and in aggregate:
  - renewable_surplus_mw       : how much renewable exceeds instantaneous need
  - potential_curtailment_mw   : what would be curtailed WITHOUT optimisation
  - bess_absorption_mw         : how much BESS charging absorbs
  - flexible_load_absorption_mw: how much flexible load absorbs
  - export_mw                  : how much is exported
  - unavoidable_curtailment_mw : residual after all absorption
  - avoided_curtailment_mw     : potential - unavoidable
  - curtailment_reduction_pct  : avoided / potential * 100

Every calculation step is returned for transparent auditability.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import numpy as np


def compute_curtailment_analysis(opt_results_df: pd.DataFrame) -> dict:
    """
    Derive curtailment metrics from the optimiser output DataFrame.

    The optimiser already resolves surplus disposition — this module
    aggregates and explains the outcomes.

    Parameters
    ----------
    opt_results_df : output of run_optimisation()

    Returns
    -------
    dict with per-interval breakdown and aggregate summary
    """
    df = opt_results_df.copy()

    # Renewable surplus = max(0, net_balance)
    df["renewable_surplus_mw"] = df["net_balance_mw"].clip(lower=0)

    # Without optimisation: all surplus would be curtailed
    df["potential_curtailment_mw"] = df["renewable_surplus_mw"]

    # BESS absorption: positive bess_action_mw means charging (absorbing surplus)
    df["bess_absorption_mw"] = df["bess_action_mw"].clip(lower=0)

    # Flexible load absorption: negative demand_response_mw means load was added
    df["flexible_load_absorption_mw"] = (-df["demand_response_mw"]).clip(lower=0)

    # Export is already tracked
    # curtailment_mw is already tracked (unavoidable)
    df["unavoidable_curtailment_mw"] = df["curtailment_mw"]

    df["avoided_curtailment_mw"] = (
        df["potential_curtailment_mw"] - df["unavoidable_curtailment_mw"]
    ).clip(lower=0)

    # Aggregate summary
    total_potential = float(df["potential_curtailment_mw"].sum())
    total_unavoidable = float(df["unavoidable_curtailment_mw"].sum())
    total_avoided = float(df["avoided_curtailment_mw"].sum())
    total_bess_abs = float(df["bess_absorption_mw"].sum())
    total_flex_abs = float(df["flexible_load_absorption_mw"].sum())
    total_export = float(df["export_mw"].sum())
    reduction_pct = (total_avoided / total_potential * 100) if total_potential > 0 else 100.0

    explanation_parts = []
    if total_potential == 0:
        explanation_parts.append("No renewable surplus detected in this window — curtailment is not applicable.")
    else:
        explanation_parts.append(
            f"Total potential curtailment (without optimisation): {total_potential:.1f} MWh."
        )
        if total_bess_abs > 0:
            explanation_parts.append(
                f"BESS charging absorbed {total_bess_abs:.1f} MWh ({total_bess_abs/total_potential*100:.1f}%)."
            )
        if total_flex_abs > 0:
            explanation_parts.append(
                f"Flexible load shifting absorbed {total_flex_abs:.1f} MWh ({total_flex_abs/total_potential*100:.1f}%)."
            )
        if total_export > 0:
            explanation_parts.append(
                f"Grid export absorbed {total_export:.1f} MWh ({total_export/total_potential*100:.1f}%)."
            )
        if total_unavoidable > 0:
            explanation_parts.append(
                f"Unavoidable curtailment: {total_unavoidable:.1f} MWh — "
                f"no further absorption capacity available."
            )
        else:
            explanation_parts.append("All surplus renewable energy was absorbed — zero curtailment achieved.")

    numeric_cols = ["renewable_surplus_mw", "potential_curtailment_mw",
                    "bess_absorption_mw", "flexible_load_absorption_mw", "export_mw",
                    "unavoidable_curtailment_mw", "avoided_curtailment_mw"]
    out_df = df[["timestamp"] + numeric_cols].copy()
    out_df[numeric_cols] = out_df[numeric_cols].round(3)
    interval_records = out_df.to_dict(orient="records")

    return {
        "summary": {
            "total_potential_curtailment_mwh": round(total_potential, 2),
            "total_bess_absorption_mwh": round(total_bess_abs, 2),
            "total_flexible_load_absorption_mwh": round(total_flex_abs, 2),
            "total_export_mwh": round(total_export, 2),
            "total_unavoidable_curtailment_mwh": round(total_unavoidable, 2),
            "total_avoided_curtailment_mwh": round(total_avoided, 2),
            "curtailment_reduction_pct": round(reduction_pct, 1),
        },
        "explanation": " ".join(explanation_parts),
        "intervals": interval_records,
    }
