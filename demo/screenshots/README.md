# Screenshots

This directory contains HTML screenshot mockups of the GridWise AI operator dashboard.

> NOTE: These are HTML mockups showing the dashboard UI with representative data.
> For live screenshots, run the application and use a browser screenshot tool.

## Files

| File | Description |
|---|---|
| `screenshot-01-overview.html` | Overview dashboard — grid status, active anomalies, curtailment avoided |
| `screenshot-02-rca.html` | Asset Diagnosis — SOL-02 inverter fault, 49.8 MW unexplained loss |
| `screenshot-03-optimisation-bob.html` | Grid Optimisation (Scenario B) + Bob generating operator brief |

## To Generate Live Screenshots

1. Start the API: `uvicorn src.backend.main:app --port 8000 --reload`
2. Open `http://localhost:8000/dashboard` in Chrome/Firefox
3. Navigate to each page and use browser DevTools > Screenshot

## Data Note

All values shown in screenshots are SYNTHETIC/DEMO data — not real grid telemetry.
