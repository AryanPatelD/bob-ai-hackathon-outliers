# GridWise AI — Phase 1: Grid Load Optimisation & Renewable Energy Advisor

> **⚠️ All data in this project is SYNTHETIC / DEMO data.**
> It is NOT real grid or weather telemetry. It is generated programmatically
> for demonstration and testing purposes only.

---

## Overview

GridWise AI is a data, ML, and analytics foundation for grid operators to:

- **Forecast** electricity demand 24 hours ahead (XGBoost)
- **Detect** demand spikes and compute risk levels (LOW / MEDIUM / HIGH / CRITICAL)
- **Estimate** expected renewable generation (physics-based solar + wind models)
- **Detect anomalies** in renewable asset performance (actual vs expected)
- **Diagnose root causes** for underperforming assets (weather-explained vs unexplained)

---

## Project Structure

```
src/
├── backend/
│   ├── __init__.py
│   └── main.py              ← FastAPI application + all endpoints
├── ml/
│   ├── __init__.py
│   ├── load_forecasting.py  ← XGBoost load forecast pipeline
│   ├── spike_detection.py   ← Demand spike / risk classification
│   ├── renewable_generation.py  ← Solar + wind physics models
│   ├── anomaly_detection.py ← Asset anomaly detection
│   └── root_cause_analysis.py   ← Evidence-based RCA
├── data/
│   ├── __init__.py
│   ├── synthetic_generator.py  ← Synthetic dataset generator
│   └── synthetic/           ← Auto-generated parquet files (gitignored)
├── services/
│   ├── __init__.py
│   └── data_store.py        ← Data access + model caching layer
├── models/
│   ├── __init__.py
│   └── schemas.py           ← Pydantic models / API schemas
├── utils/
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_gridwise.py     ← Full test suite
├── requirements.txt
├── .env.example
└── README.md                ← This file
```

---

## Quick Start

### 1. Install dependencies

```bash
cd src
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python -m src.data.synthetic_generator
```

This generates ~8,760 hourly records (1 year) for:
- Weather (temperature, humidity, cloud cover, irradiance, wind speed/direction)
- Grid load + available capacity + conventional generation
- 5 renewable assets (3 solar, 2 wind) with injected faults
- 2 BESS units with simulated SOC

Saved as parquet files in `src/data/synthetic/`.

### 3. Train the model

The model trains automatically on first API call. To pre-train:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.services.data_store import get_or_train_model
get_or_train_model()
"
```

### 4. Run the API

```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation: http://localhost:8000/docs

### 5. Run tests

```bash
pytest src/tests/test_gridwise.py -v
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/assets` | All renewable assets (latest status) |
| GET | `/assets/anomalies` | Anomaly detection (default 24h lookback) |
| GET | `/assets/{id}/diagnosis` | Root cause analysis for a single asset |
| GET | `/forecast/load` | 24-hour load forecast (XGBoost) |
| GET | `/forecast/renewables` | Expected generation for all assets |
| GET | `/grid/risk` | Demand spike risk assessment |
| GET | `/docs` | OpenAPI interactive documentation |
| GET | `/redoc` | ReDoc documentation |

### Example calls

```bash
# Health check
curl http://localhost:8000/health

# Get all assets
curl http://localhost:8000/assets

# Get anomalies (last 48 hours, CRITICAL only)
curl "http://localhost:8000/assets/anomalies?hours=48&severity=CRITICAL"

# Root cause diagnosis for asset SOL-01
curl http://localhost:8000/assets/SOL-01/diagnosis

# 24-hour load forecast
curl http://localhost:8000/forecast/load

# Wind renewable forecast only
curl "http://localhost:8000/forecast/renewables?asset_type=wind"

# Grid risk (72h, custom critical threshold)
curl "http://localhost:8000/grid/risk?horizon_hours=72&critical_reserve=0.08"
```

---

## ML Models

### Load Forecasting (XGBoost)
- **Algorithm**: XGBoost Regressor with early stopping
- **Features**: hour, day-of-week, month, is_weekend, day-of-year, temperature, humidity, load lags (1h/2h/3h/24h/48h/168h), rolling means (3h/6h/24h)
- **Split**: chronological 70% train / 15% validation / 15% test
- **Target**: `grid_load_mw`
- **Leakage prevention**: lags use `shift(n)` — strictly backward-looking

### Solar Expected Generation
Physics model: GHI insolation ratio × cloud transmissivity × temperature derating
- Cloud transmissivity: `1 - 0.75 × (cloud_cover/100)³`
- Temperature derating: `-0.45%/°C` above 25°C (STC)

### Wind Expected Generation
Parametric power curve model:
- Cut-in: 3 m/s | Rated: 12 m/s | Cut-out: 25 m/s
- Cubic ramp from cut-in to rated
- Air density correction for temperature

### Anomaly Detection
Deterministic threshold comparison on performance ratio and absolute deviation.
Key safeguard: anomalies are **suppressed when expected generation < 2 MW**
(prevents nighttime / calm-weather false positives).

### Root Cause Analysis
Evidence-based rule system:
- Separates weather-explained losses from unexplained losses
- Returns confidence scores [0, 1] per candidate cause
- Never claims mechanical fault as certain without supporting evidence

---

## Assumptions

1. All data is synthetic — not intended for production use without real telemetry
2. Solar model assumes latitude ~35°N for irradiance calculations
3. Wind power curve uses generic IEC Class II parameters
4. Air density correction uses ISO 2533 standard atmosphere model
5. Grid capacity is treated as static (2,600 MW default) — real systems vary
6. BESS SOC is simulated with a simplified charge/discharge schedule
7. Fault injection affects ~5% of asset records to create detectable anomalies

## Known Limitations

1. No real-time data ingestion — data is static parquet files
2. Load forecast is univariate (load + weather) — no market or event signals
3. Wind direction does not affect generation in the current model (turbines yaw)
4. No uncertainty quantification beyond train/val/test metric reporting
5. RCA confidence scores are heuristic — not calibrated probabilities
6. BESS optimisation dispatch is not implemented (Phase 2)
7. No model drift detection or retraining triggers
8. No authentication on the API (Phase 2)

---

## Configuration

All thresholds are configurable via environment variables.
Copy `src/.env.example` to `src/.env` and adjust as needed.

Key variables:
- `SPIKE_CRITICAL_RESERVE` — reserve margin below which risk is CRITICAL (default 5%)
- `ANOMALY_CRITICAL_PERF_RATIO` — performance ratio threshold for CRITICAL (default 0.50)
- `ANOMALY_MIN_EXPECTED_MW` — suppress anomaly alerts below this expected output (default 2 MW)
- `RCA_INSPECTION_THRESHOLD_MW` — unexplained loss that triggers inspection recommendation (default 3 MW)

---

## Running Everything (Single Command Sequence)

```bash
# From workspace root
cd src
pip install -r requirements.txt
python -m src.data.synthetic_generator           # Generate data (~5s)
pytest src/tests/test_gridwise.py -v             # Run tests
uvicorn src.backend.main:app --port 8000 --reload  # Start API
```

The first API call that involves load forecasting will train the XGBoost model
automatically (~30–60 seconds). Subsequent calls use the cached model.
