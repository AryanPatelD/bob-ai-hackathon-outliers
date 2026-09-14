# Solution Overview

## What We Built

GridWise AI is an end-to-end grid intelligence platform with five integrated layers:

### Layer 1: Data Foundation
Synthetic but realistic hourly time-series data for one calendar year (8,760 records) covering:
- Grid load and capacity
- Weather (temperature, humidity, cloud cover, irradiance, wind speed/direction)
- 5 renewable assets (3 solar, 2 wind) with realistic fault injection
- 2 BESS units with simulated SOC cycles

### Layer 2: Forecasting and Monitoring

**Load Forecasting** — XGBoost regression with chronological train/val/test splitting. Features include temporal encodings, lagged load values (1h–168h), and weather inputs. Achieves R²=0.90 on held-out test data.

**Renewable Expected Generation** — Physics-based deterministic models:
- Solar: GHI × cloud transmissivity factor × temperature derating coefficient
- Wind: Cubic power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s) with air density correction

Both models return intermediate calculation steps for complete auditability.

### Layer 3: Anomaly Detection and Diagnosis

**Anomaly Detection** — Compares actual vs expected generation using performance ratio and absolute deviation thresholds. Includes a critical safeguard: low expected generation (nighttime, calm wind) is not flagged as an anomaly, preventing false positives.

**Root Cause Analysis** — Evidence-based rule system that:
1. Calculates the total generation loss
2. Estimates weather-explained loss (cloud cover, low irradiance, high temperature, insufficient/excessive wind)
3. Calculates residual unexplained loss
4. Assigns confidence-scored cause candidates
5. Recommends physical inspection only when unexplained loss exceeds a configurable threshold

### Layer 4: Grid Optimisation

Priority-order dispatch engine that calculates `net_balance = renewable + conventional - demand` per interval and dispatches responses:

**Deficit**: BESS discharge → Demand response → Backup generation  
**Surplus**: BESS charging → Flexible load shifting → Export → Curtailment (last resort)

All BESS constraints enforced: SOC bounds [10%, 95%], max charge/discharge rates, energy capacity.

### Layer 5: IBM Bob Integration

A custom MCP server written in Node.js/TypeScript exposes nine tools to IBM Bob. Each tool calls the FastAPI backend and returns formatted, labelled results. Bob labels every piece of data by source type (MEASURED / ML PREDICTION / RULE-BASED / AI EXPLANATION) and never fabricates numerical values.

## How It All Connects

```
Weather + Grid Data
        ↓
  Physics Models ──→ Expected Generation
        ↓                    ↓
  XGBoost Forecast    Anomaly Detection
        ↓                    ↓
  Spike Detection     Root Cause Analysis
        ↓                    ↓
        └────────────────────┘
                  ↓
        Grid Optimisation Engine
                  ↓
        Curtailment Minimiser
                  ↓
        Operator Brief Generator
                  ↓
    FastAPI (REST) + IBM Bob (MCP)
                  ↓
         Operator Dashboard
```

## Design Decisions

1. **Explainability over black-box accuracy**: Every model returns its intermediate steps. Operators can see exactly why a recommendation was made.

2. **Conservative anomaly detection**: The system would rather miss a marginal warning than generate false alarms during poor weather. Trust is built through precision, not recall.

3. **Deterministic optimisation**: The dispatch engine uses configurable priority rules rather than a learned policy. This makes behaviour predictable and auditable — critical for grid operations.

4. **Bob as a structured query interface**: Bob calls nine specific backend tools. It cannot invent numbers. Every numerical claim in a Bob response traces to a computed backend value.

5. **Synthetic data clearly labelled**: Every API response and Bob tool output includes a `data_type` field identifying whether the value is measured, predicted, rule-based, or AI-generated. The dashboard displays these labels prominently.
