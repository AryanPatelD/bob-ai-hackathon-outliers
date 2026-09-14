# Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Python 3.14 tested |
| Node.js | 20+ | Required for MCP server |
| npm | 8+ | Included with Node.js |
| Git | Any | For cloning |

Check versions:
```bash
python --version    # must be 3.11+
node --version      # must be v20+
npm --version
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-org/bob-ai-hackathon-outliers.git
cd bob-ai-hackathon-outliers
```

---

## 2. Environment Variables

```bash
cp src/.env.example src/.env
```

The default values in `.env.example` work without modification for a local setup.
No API keys or secrets are required — GridWise AI uses only synthetic data.

Key variables (all have safe defaults):
```
DATA_DIR=src/data/synthetic
MODEL_PATH=src/ml/saved_models/load_forecast_xgb.json
SPIKE_CRITICAL_RESERVE=0.05
ANOMALY_MIN_EXPECTED_MW=2.0
RCA_INSPECTION_THRESHOLD_MW=3.0
```

---

## 3. Install Python Dependencies

```bash
pip install -r src/requirements.txt
```

Core packages installed:
- `fastapi` — REST API framework
- `uvicorn` — ASGI server
- `xgboost` — load forecasting model
- `scikit-learn` — metrics
- `pandas`, `numpy`, `pyarrow` — data processing
- `pydantic` — schema validation
- `pytest` — testing

---

## 4. Generate Synthetic Data

```bash
python -m src.data.synthetic_generator
```

This generates 4 parquet files in `src/data/synthetic/`:
- `weather.parquet` — 8,760 hourly weather records
- `grid.parquet` — 8,760 hourly grid load records
- `assets.parquet` — 43,800 renewable asset records (5 assets × 8,760h)
- `bess.parquet` — 17,520 BESS records (2 units × 8,760h)

> WARNING: All data is SYNTHETIC — not real telemetry.

---

## 5. Run Tests

```bash
pytest src/tests/ -v
```

Expected result: **62 passed, 0 failed, 0 warnings**

Tests cover:
- Load forecasting pipeline (leakage prevention, chronological split, feature engineering)
- Anomaly detection (normal/warning/critical, nighttime suppression)
- Root cause analysis (weather vs unexplained, inspection triggers)
- Grid optimisation (deficit, surplus, BESS constraints, energy conservation)
- Curtailment analysis (accounting correctness)
- Demo scenarios A, B, C

---

## 6. Start the Backend API

```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

On first request to any forecast endpoint, the XGBoost model will train automatically (~30-60 seconds). Subsequent requests use the cached model at `src/ml/saved_models/load_forecast_xgb.json`.

To pre-train the model before starting (optional):
```bash
python -c "import sys; sys.path.insert(0,'.'); from src.services.data_store import get_or_train_model; get_or_train_model()"
```

---

## 7. Access the Dashboard and API

| URL | Description |
|---|---|
| `http://localhost:8000/dashboard` | Operator Dashboard (7 pages) |
| `http://localhost:8000/docs` | Interactive API Documentation (Swagger UI) |
| `http://localhost:8000/redoc` | ReDoc API Documentation |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/demo` | Hackathon demo scenario (Scenario D) |

---

## 8. IBM Bob MCP Server Setup

The MCP server is registered at `.bob/mcp.json` and connects automatically when the workspace is open in Bob.

**Build the MCP server** (required once after cloning):
```bash
cd src/bob_mcp
npm install
npx tsc
cd ../..
```

**Verify the build**:
```bash
# Should print: True
Test-Path src/bob_mcp/build/index.js   # PowerShell
# OR
test -f src/bob_mcp/build/index.js     # bash
```

The `.bob/mcp.json` configuration points to the built server:
```json
{
  "mcpServers": {
    "gridwise-bob-mcp": {
      "command": "node",
      "args": ["src/bob_mcp/build/index.js"],
      "env": { "GRIDWISE_API_URL": "http://localhost:8000" }
    }
  }
}
```

**Important**: The MCP server requires the FastAPI backend to be running (`uvicorn`) before Bob can call the GridWise tools.

---

## 9. Verification Checklist

Run this to verify the complete pipeline:

```bash
python src/utils/audit.py
```

Expected output:
```
AUDIT RESULTS: 10 PASSED  0 FAILED
  PASS  1. Data Loading
  PASS  2. Load Forecasting
  PASS  3. Spike Detection
  PASS  4. Renewable Forecast
  PASS  5. Anomaly Detection
  PASS  6. Root Cause Analysis
  PASS  7. Grid Optimisation
  PASS  8. Curtailment Minimisation
  PASS  9. Operator Brief
  PASS  10. API Route Registration
```

---

## 10. Demo Walkthrough

### Scenario D — Hackathon Demo (Spike + Inverter Fault)

```bash
curl http://localhost:8000/demo
```

Or open the dashboard at `http://localhost:8000/dashboard`:
1. Go to **Grid Optimisation** → select **Scenario B**
2. Go to **Asset Performance** → observe SOL-02 CRITICAL
3. Go to **Asset Diagnosis** → select SOL-02 → click Diagnose
4. Go to **AI Operator Advisor** → type: *"Generate the operator optimisation brief"*

---

## 11. Troubleshooting

### "No matching distribution found for numpy==..."
The `requirements.txt` uses flexible version bounds (`>=`). If a version conflict occurs:
```bash
pip install fastapi uvicorn pydantic xgboost scikit-learn pandas numpy pyarrow python-dotenv httpx pytest pytest-asyncio
```

### "Model not found" on first API call
This is expected — the model trains automatically. Wait ~60 seconds on the first forecast request.

### MCP server not connecting in Bob
1. Ensure the API is running: `http://localhost:8000/health`
2. Verify the build: `node src/bob_mcp/build/index.js` (should start silently)
3. Check `.bob/mcp.json` path matches your workspace root

### Dashboard shows "API Offline"
The dashboard cannot connect to `http://localhost:8000`. Start the uvicorn server first.

### UnicodeEncodeError on Windows
This was fixed — all non-ASCII characters in print statements were replaced with ASCII equivalents.

---

## Complete Setup (Single Script)

```bash
# Clone
git clone https://github.com/your-org/bob-ai-hackathon-outliers.git
cd bob-ai-hackathon-outliers

# Python setup
pip install -r src/requirements.txt
python -m src.data.synthetic_generator
pytest src/tests/ -v

# MCP server build
cd src/bob_mcp && npm install && npx tsc && cd ../..

# Start API
uvicorn src.backend.main:app --port 8000 --reload
# Open: http://localhost:8000/dashboard
```
