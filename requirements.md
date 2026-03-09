# Week Task Requirements (Model B Dynamic Features)

## Goal
Build a unified monthly dynamic table keyed by `bgrp_id, month` for Model B (ST-Encoder + LSTM baseline), and prepare model-ready artifacts reusable for Model C.

## Required Inputs
- `data/raw/Automated_Traffic_Volume_Counts_20260209.csv`
- `data/raw/Traffic_Volume_Counts_(Historical)_20260209.csv` (optional but supported)
- `data/raw/Active_Cabaret_and_Catering_Licenses_20260209.csv` (optional but supported)
- `data/raw/weather_monthly.csv` (optional, schema-flexible)
- `data/raw/MapPLUTO25v4_unclipped.gdb` + layer `MapPLUTO_25v4_unclipped` (for BBL crosswalk)
- `data/raw/nyc_bgrp.geojson` (recommended, true bgrp geometry)

## Fallback Behavior
- If `bgrp_geojson` is missing, pipeline falls back to `block_id` as `bgrp_id` (proxy only).
- `use_bbl_crosswalk` defaults to `false` for speed; enable it if you need BBL-key alignment from MapPLUTO.
- Monthly window is controlled by `monthly_start_month` / `monthly_end_month` in `config.yaml`.
- Join priority:
  1. key-based joins (`BBL`, then existing ids)
  2. spatial joins (within + nearest fallback)

## Run Commands (from repo root)
```powershell
python -m src.build_monthly_dynamic_features --config config.yaml
python -m src.build_model_inputs --config config.yaml
```

## Outputs
- `data/processed/monthly_dynamic_with_activity.csv`
- `data/processed/monthly_dynamic.parquet`
- `data/processed/monthly_dynamic_quality.csv`
- `data/processed/scaler.pkl`
- `data/processed/model_inputs_monthly.csv`
- `data/processed/model_inputs_monthly_tensors.npz`

## Quality Checks Required
- Unique key check: one row per `bgrp_id, month`
- Missingness rate per feature column
- Temporal coverage (month count)
- Spatial coverage (bgrp count)

## Python Dependencies
Current repo dependencies from `requirements.txt` are sufficient for CSV flow.
For parquet I/O, install one of:
- `pyarrow` (recommended)
- `fastparquet`
