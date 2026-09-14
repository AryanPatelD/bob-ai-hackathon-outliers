#!/usr/bin/env node
/**
 * GridWise AI — IBM Bob MCP Server
 *
 * Exposes GridWise AI backend analytics to IBM Bob as MCP tools.
 * Bob retrieves COMPUTED results from the backend API — it never fabricates
 * numerical values or telemetry.
 *
 * All tools call the GridWise FastAPI backend (default: http://localhost:8000).
 *
 * Available tools:
 *   get_grid_status         — current grid state
 *   get_load_forecast       — 24h load forecast
 *   get_spike_risks         — demand spike risk assessment
 *   get_renewable_forecast  — expected generation for all assets
 *   get_asset_anomalies     — anomaly detection results
 *   diagnose_asset          — root cause analysis for a specific asset
 *   get_optimisation_plan   — grid optimisation plan (scenario or live)
 *   get_curtailment_plan    — curtailment minimisation analysis
 *   generate_operator_brief — full structured operator brief
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const API_BASE = process.env.GRIDWISE_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// HTTP helper — calls the GridWise backend
// ---------------------------------------------------------------------------

async function apiGet(path: string): Promise<unknown> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`GridWise API error ${res.status} for ${url}: ${await res.text()}`);
  }
  return res.json();
}

async function apiPost(path: string, body: unknown): Promise<unknown> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`GridWise API error ${res.status} for POST ${url}: ${await res.text()}`);
  }
  return res.json();
}

function formatJson(data: unknown): string {
  return JSON.stringify(data, null, 2);
}

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

const server = new McpServer({
  name: "gridwise-bob-mcp",
  version: "1.0.0",
});

// ---------------------------------------------------------------------------
// Tool: get_grid_status
// ---------------------------------------------------------------------------

server.tool(
  "get_grid_status",
  "Get the current grid status including load, capacity, reserve margin, and overall risk level. Data source: computed from live synthetic grid data.",
  {},
  async () => {
    try {
      const data = await apiGet("/grid/risk?horizon_hours=1");
      const risk = data as { risk_summary: Array<Record<string, unknown>>; thresholds: Record<string, unknown> };
      const latest = risk.risk_summary?.[0];
      if (!latest) throw new Error("No grid status data available");

      const text = [
        "=== GRID STATUS (COMPUTED FROM BACKEND) ===",
        `Data Type: RULE-BASED OPTIMISATION RESULT — not raw telemetry`,
        `Timestamp: ${latest.timestamp}`,
        `Forecast Load: ${latest.forecast_load_mw} MW`,
        `Available Capacity: ${latest.available_capacity_mw} MW`,
        `Reserve Margin: ${((latest.reserve_margin as number) * 100).toFixed(1)}%`,
        `Spike Risk Score: ${latest.spike_risk_score}`,
        `Risk Level: ${latest.risk_level}`,
        ``,
        `Thresholds: CRITICAL <${(risk.thresholds.critical_reserve_margin as number) * 100}% | HIGH <${(risk.thresholds.high_reserve_margin as number) * 100}% | MEDIUM <${(risk.thresholds.medium_reserve_margin as number) * 100}%`,
      ].join("\n");

      return { content: [{ type: "text", text }] };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error fetching grid status: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Tool: get_load_forecast
// ---------------------------------------------------------------------------

server.tool(
  "get_load_forecast",
  "Get the 24-hour electricity demand forecast from the XGBoost model. Returns hourly forecast load in MW with model accuracy metrics. Data source: ML model prediction.",
  {
    horizon_hours: z.number().min(1).max(168).optional().describe("Forecast horizon in hours (default 24)"),
  },
  async ({ horizon_hours = 24 }) => {
    try {
      const data = await apiGet(`/forecast/load?horizon_hours=${horizon_hours}`) as Record<string, unknown>;
      const fc = data.forecast as Array<Record<string, unknown>>;
      const loads = fc.map(p => p.forecast_load_mw as number);
      const peak = Math.max(...loads);
      const avg = loads.reduce((a, b) => a + b, 0) / loads.length;
      const peakTs = fc[loads.indexOf(peak)]?.timestamp;

      const lines = [
        `=== LOAD FORECAST (ML MODEL PREDICTION) ===`,
        `Data Type: XGBoost model output — not measured telemetry`,
        `Horizon: ${horizon_hours} hours`,
        `Peak Forecast: ${peak.toFixed(1)} MW at ${peakTs}`,
        `Average Forecast: ${avg.toFixed(1)} MW`,
        ``,
        `Model Metrics (test set):`,
      ];

      const metrics = data.model_metrics as Record<string, Record<string, number>> | null;
      if (metrics?.test) {
        lines.push(`  MAE: ${metrics.test.mae.toFixed(1)} MW | RMSE: ${metrics.test.rmse.toFixed(1)} MW | R²: ${metrics.test.r2.toFixed(3)}`);
      }

      lines.push(``, `Hourly Forecast:`);
      fc.slice(0, horizon_hours).forEach(p => {
        lines.push(`  ${p.timestamp}: ${(p.forecast_load_mw as number).toFixed(1)} MW`);
      });

      return { content: [{ type: "text", text: lines.join("\n") }] };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error fetching load forecast: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Tool: get_spike_risks
// ---------------------------------------------------------------------------

server.tool(
  "get_spike_risks",
  "Get demand spike risk assessment for the next N hours. Returns risk level (LOW/MEDIUM/HIGH/CRITICAL) and reserve margin per hour. Data source: rule-based analysis on forecast.",
  {
    horizon_hours: z.number().min(1).max(168).optional().describe("Hours to look ahead (default 6)"),
  },
  async ({ horizon_hours = 6 }) => {
    try {
      const data = await apiGet(`/grid/risk?horizon_hours=${horizon_hours}`) as Record<string, unknown>;
      const risks = data.risk_summary as Array<Record<string, unknown>>;

      const critical = risks.filter(r => r.risk_level === "CRITICAL");
      const high = risks.filter(r => r.risk_level === "HIGH");
      const medium = risks.filter(r => r.risk_level === "MEDIUM");

      const lines = [
        `=== DEMAND SPIKE RISKS (RULE-BASED ANALYSIS) ===`,
        `Data Type: Deterministic threshold rules applied to forecast`,
        `Horizon: ${horizon_hours} hours`,
        `Critical Periods: ${critical.length}`,
        `High Risk Periods: ${high.length}`,
        `Medium Risk Periods: ${medium.length}`,
        ``,
      ];

      if (critical.length > 0) {
        lines.push(`CRITICAL PERIODS:`);
        critical.forEach(r => {
          lines.push(`  ${r.timestamp}: Load ${r.forecast_load_mw} MW, Reserve ${((r.reserve_margin as number) * 100).toFixed(1)}%, Score ${r.spike_risk_score}`);
        });
        lines.push(``);
      }

      lines.push(`All Risk Levels:`);
      risks.forEach(r => {
        lines.push(`  ${r.timestamp}: [${r.risk_level}] ${r.forecast_load_mw} MW, margin ${((r.reserve_margin as number) * 100).toFixed(1)}%`);
      });

      return { content: [{ type: "text", text: lines.join("\n") }] };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error fetching spike risks: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Tool: get_renewable_forecast
// ---------------------------------------------------------------------------

server.tool(
  "get_renewable_forecast",
  "Get expected renewable generation for all solar and wind assets based on physics-based models. Returns expected MW and capacity factor per asset. Data source: physics model prediction.",
  {
    asset_type: z.enum(["solar", "wind", "all"]).optional().describe("Filter by asset type (default: all)"),
  },
  async ({ asset_type = "all" }) => {
    try {
      const query = asset_type !== "all" ? `?asset_type=${asset_type}` : "";
      const data = await apiGet(`/forecast/renewables${query}`) as Record<string, unknown>;
      const forecasts = data.forecasts as Array<Record<string, unknown>>;

      // Group by asset
      const byAsset: Record<string, Array<Record<string, unknown>>> = {};
      forecasts.forEach(f => {
        const id = f.asset_id as string;
        if (!byAsset[id]) byAsset[id] = [];
        byAsset[id].push(f);
      });

      const lines = [
        `=== RENEWABLE FORECAST (PHYSICS MODEL) ===`,
        `Data Type: Physics-based deterministic model output`,
        `Assets covered: ${Object.keys(byAsset).length}`,
        ``,
      ];

      for (const [assetId, pts] of Object.entries(byAsset)) {
        const totalMW = pts.reduce((s, p) => s + (p.expected_generation_mw as number), 0);
        const avgCF = pts.reduce((s, p) => s + (p.capacity_factor as number), 0) / pts.length;
        const sample = pts[0];
        lines.push(`${sample.asset_id} (${sample.asset_type}, ${sample.capacity_mw} MW):`);
        lines.push(`  Total expected: ${totalMW.toFixed(1)} MWh over ${pts.length}h`);
        lines.push(`  Average CF: ${(avgCF * 100).toFixed(1)}%`);
      }

      return { content: [{ type: "text", text: lines.join("\n") }] };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error fetching renewable forecast: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Tool: get_asset_anomalies
// ---------------------------------------------------------------------------

server.tool(
  "get_asset_anomalies",
  "Detect underperforming renewable assets by comparing actual vs expected generation. Returns severity (NORMAL/WARNING/CRITICAL) and performance ratios. Data source: deterministic rule-based anomaly detection.",
  {
    hours: z.number().min(1).max(168).optional().describe("Lookback window in hours (default 24)"),
    severity: z.enum(["WARNING", "CRITICAL"]).optional().describe("Filter by severity"),
  },
  async ({ hours = 24, severity }) => {
    try {
      const q = severity ? `?hours=${hours}&severity=${severity}` : `?hours=${hours}`;
      const data = await apiGet(`/assets/anomalies${q}`) as Record<string, unknown>;
      const anomalies = data.anomalies as Array<Record<string, unknown>>;

      const lines = [
        `=== ASSET ANOMALIES (RULE-BASED DETECTION) ===`,
        `Data Type: Deterministic threshold comparison — actual vs expected`,
        `Lookback: ${hours} hours`,
        `Total assets: ${data.total_assets}`,
        `In WARNING: ${data.assets_in_warning}`,
        `In CRITICAL: ${data.assets_in_critical}`,
        ``,
      ];

      if (anomalies.length === 0) {
        lines.push("No anomalies detected — all assets performing within expected range.");
      } else {
        anomalies.forEach(a => {
          lines.push(
            `[${a.severity}] ${a.asset_name} (${a.asset_type}): ` +
            `actual ${a.actual_generation_mw} MW vs expected ${a.expected_generation_mw} MW | ` +
            `perf ratio ${a.performance_ratio} | dev ${a.absolute_deviation_mw} MW`
          );
        });
      }

      return { content: [{ type: "text", text: lines.join("\n") }] };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error fetching anomalies: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Tool: diagnose_asset
// ---------------------------------------------------------------------------

server.tool(
  "diagnose_asset",
  "Run root cause analysis for a specific renewable asset. Separates weather-explained losses from unexplained losses and returns probable causes with confidence scores. Data source: evidence-based heuristic rules.",
  {
    asset_id: z.string().describe("Asset ID (e.g. SOL-01, WIN-02)"),
  },
  async ({ asset_id }) => {
    try {
      const data = await apiGet(`/assets/${asset_id}/diagnosis`) as Record<string, unknown>;

      const lines = [
        `=== ASSET DIAGNOSIS: ${data.asset_id} — ${data.asset_name} (${data.asset_type}) ===`,
        `Data Type: Evidence-based heuristic RCA — not a mechanical certainty`,
        `Timestamp: ${data.timestamp}`,
        `Expected Generation: ${data.expected_generation_mw} MW`,
        `Actual Generation:   ${data.actual_generation_mw} MW`,
        `Deviation: ${data.deviation_mw} MW (${((data.deviation_pct as number) * 100).toFixed(1)}%)`,
        `Weather-Explained Loss: ${data.weather_explained_loss_mw} MW`,
        `Unexplained Loss:       ${data.unexplained_loss_mw} MW`,
        `Overall Confidence: ${((data.overall_confidence as number) * 100).toFixed(0)}%`,
        `Inspection Recommended: ${data.recommended_inspection ? "YES" : "NO"}`,
        ``,
        `Probable Causes:`,
      ];

      const causes = data.probable_causes as Array<Record<string, unknown>>;
      if (causes.length === 0) {
        lines.push("  No anomaly detected — generation within expected range.");
      } else {
        causes.forEach(c => {
          lines.push(`  [${((c.confidence as number) * 100).toFixed(0)}%] ${c.cause}: ${c.description}`);
        });
      }

      if (data.recommended_inspection) {
        lines.push(``, `INSPECTION NOTES: ${data.inspection_notes}`);
      }

      return { content: [{ type: "text", text: lines.join("\n") }] };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error diagnosing asset ${asset_id}: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Tool: get_optimisation_plan
// ---------------------------------------------------------------------------

server.tool(
  "get_optimisation_plan",
  "Get the grid optimisation plan showing BESS dispatch, demand response, and backup generation recommendations. Optionally run a demo scenario (A/B/C). Data source: deterministic priority-order dispatch optimiser.",
  {
    scenario: z.enum(["A", "B", "C"]).optional().describe("Demo scenario: A=normal, B=demand spike, C=surplus+anomaly"),
    horizon_hours: z.number().min(1).max(72).optional().describe("Forecast horizon (default 24)"),
  },
  async ({ scenario, horizon_hours = 24 }) => {
    try {
      const path = scenario
        ? `/optimisation/plan?scenario=${scenario}`
        : `/optimisation/plan?horizon_hours=${horizon_hours}`;
      const data = await apiGet(path) as Record<string, unknown>;

      const results = data.results as Array<Record<string, unknown>>;
      const summary = data.summary as Record<string, unknown>;

      const lines = [
        `=== OPTIMISATION PLAN (DETERMINISTIC DISPATCH) ===`,
        `Data Type: Rule-based priority-order dispatch — not ML prediction`,
        scenario ? `Scenario: ${scenario} — ${summary?.scenario_name ?? ""}` : `Horizon: ${horizon_hours}h`,
        ``,
        `SUMMARY:`,
        `  Deficit periods: ${summary?.deficit_periods ?? 0}`,
        `  Surplus periods: ${summary?.surplus_periods ?? 0}`,
        `  BESS total discharge: ${summary?.total_bess_discharge_mwh ?? 0} MWh`,
        `  BESS total charge: ${summary?.total_bess_charge_mwh ?? 0} MWh`,
        `  Demand response activations: ${summary?.total_dr_intervals ?? 0}`,
        `  Backup generation required: ${summary?.backup_required ? "YES" : "NO"}`,
        `  Total curtailment: ${summary?.total_curtailment_mwh ?? 0} MWh`,
        ``,
        `INTERVAL ACTIONS (first 12h):`,
      ];

      results.slice(0, 12).forEach(r => {
        lines.push(`  ${r.timestamp}: net ${r.net_balance_mw} MW | ${r.action_taken}`);
      });

      return { content: [{ type: "text", text: lines.join("\n") }] };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error fetching optimisation plan: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Tool: get_curtailment_plan
// ---------------------------------------------------------------------------

server.tool(
  "get_curtailment_plan",
  "Get the renewable curtailment minimisation analysis showing how much surplus energy was absorbed vs curtailed. Data source: derived from optimisation results.",
  {
    scenario: z.enum(["A", "B", "C"]).optional().describe("Demo scenario to analyse"),
  },
  async ({ scenario }) => {
    try {
      const path = scenario ? `/optimisation/curtailment?scenario=${scenario}` : `/optimisation/curtailment`;
      const data = await apiGet(path) as Record<string, unknown>;
      const summary = data.summary as Record<string, unknown>;

      const lines = [
        `=== CURTAILMENT MINIMISATION PLAN (DERIVED FROM OPTIMISATION) ===`,
        `Data Type: Calculated from dispatch optimiser — transparent arithmetic`,
        ``,
        `Potential Curtailment (without opt): ${summary.total_potential_curtailment_mwh} MWh`,
        `BESS Absorption:                     ${summary.total_bess_absorption_mwh} MWh`,
        `Flexible Load Absorption:            ${summary.total_flexible_load_absorption_mwh} MWh`,
        `Grid Export:                         ${summary.total_export_mwh} MWh`,
        `Unavoidable Curtailment:             ${summary.total_unavoidable_curtailment_mwh} MWh`,
        `Avoided Curtailment:                 ${summary.total_avoided_curtailment_mwh} MWh`,
        `Curtailment Reduction:               ${summary.curtailment_reduction_pct}%`,
        ``,
        `Explanation: ${data.explanation}`,
      ];

      return { content: [{ type: "text", text: lines.join("\n") }] };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error fetching curtailment plan: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Tool: generate_operator_brief
// ---------------------------------------------------------------------------

server.tool(
  "generate_operator_brief",
  "Generate a complete structured operator optimisation brief covering grid status, forecasts, risks, anomalies, root causes, BESS plan, demand response plan, and curtailment plan. Data source: aggregated from all GridWise backend services.",
  {
    scenario: z.enum(["A", "B", "C"]).optional().describe("Optional demo scenario (default: live data)"),
  },
  async ({ scenario }) => {
    try {
      const path = scenario ? `/brief?scenario=${scenario}` : `/brief`;
      const data = await apiGet(path) as Record<string, unknown>;

      const lines = [
        `=== OPERATOR BRIEF (AGGREGATED FROM ALL GRIDWISE BACKEND SERVICES) ===`,
        `Data Types Used: MEASURED | MODEL PREDICTIONS | RULE/OPTIMISATION RESULTS | AI EXPLANATIONS`,
        `Generated: ${data.generated_at}`,
        ``,
        data.full_text as string,
      ];

      return { content: [{ type: "text", text: lines.join("\n") }] };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error generating operator brief: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("GridWise IBM Bob MCP server running on stdio");
  console.error(`Backend API: ${API_BASE}`);
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
