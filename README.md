# MDS - Traffic Risk Dashboard Demo

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard_app.py
```

## What this demo includes

- KPI overview for accident/risk indicators
- GPS-based risk map with high/medium/low levels
- Decision recommendation table by threshold
- High-risk time analysis and area comparison
- Sidebar controls for city/month filters and threshold tuning

## Data

You can run with built-in synthetic demo data immediately.

If you upload your own CSV, include at least these columns:

- `pred_prob`
- `city`
- `hour`
- `lat`
- `lon`

Optional columns used for richer display:
- `month`, `district`, `accident_type`, `deaths`, `injuries`
