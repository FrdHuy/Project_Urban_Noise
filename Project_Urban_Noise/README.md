# Project_Urban_Noise

Data processing pipeline for the SYSEN 5900 urban noise project. Transforms raw NYC geospatial and traffic data into model-ready features at the block-group level.

---

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml   # then edit paths as needed
```

---

## Pipeline

Run all commands from this directory (`Project_Urban_Noise/`).

| Step | Command | Output |
|------|---------|--------|
| 1. Block geometry | `python -m src.build_blocks --config config.yaml` | `data/processed/blocks.geojson` |
| 2. Static features | `python -m src.compute_block_features --config config.yaml` | `data/processed/block_features.csv` |
| 3. Dynamic features | `python -m src.build_monthly_dynamic_features --config config.yaml` | `data/processed/monthly_dynamic_with_activity.csv` |
| 4. Traffic calibration | `python -m src.traffic_interpolation_quantile` | `data/processed/traffic_combined_calibrated.csv` |
| 5. Model inputs | `python -m src.build_model_inputs --config config.yaml` | `data/processed/model_inputs_monthly.csv` |

---

## Source Modules (`src/`)

| Module | Description |
|--------|-------------|
| `common.py` | Shared utilities (config loading, logging, column detection) |
| `build_blocks.py` | Dissolve MapPLUTO parcels into block geometries |
| `compute_block_features.py` | Building density, FAR, height metrics per block |
| `build_monthly_dynamic_features.py` | Monthly traffic + activity features per block group |
| `traffic_interpolation_quantile.py` | Traffic gap-fill (year-month mean) + quantile calibration |
| `compute_dynamic_features.py` | Weekly dynamic feature alignment |
| `build_nsi_input.py` | NSI model input assembly |
| `build_model_inputs.py` | Scale and export final model input tensors |
| `plot_heatmaps.py` | Heatmaps for static features |
| `plot_monthly_dynamic_visuals.py` | Monthly feature visualizations |
| `traffic_eda.py` | Traffic exploratory analysis |

---

## Data

All data lives in `data/` (not committed to git).

```
data/
├── raw/          # Original source files — never modify
└── processed/    # Pipeline outputs — can be regenerated
```

Key raw inputs: `MapPLUTO25v4_unclipped.gdb`, `Automated_Traffic_Volume_Counts_*.csv`, `Traffic_Volume_Counts_(Historical)_*.csv`, `nyc_bgrp.geojson`

---

## Weekly Reports

Progress reports are in `docs/reports/` named `YYYY-MM-DD_<task>.md`.
