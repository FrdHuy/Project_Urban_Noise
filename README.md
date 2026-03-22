# SYSEN 5900 — Urban Noise Project

NYC urban noise complaint prediction. The project links traffic, building, and activity data to 311 noise complaints at the block-group level, and trains temporal forecasting models (TFT, G-Transformer, LSTM).

---

## Repository Structure

```
SYSEN 5900/
│
├── Project_Urban_Noise/          # Main analysis pipeline
│   ├── src/                      # Python modules
│   │   ├── build_blocks.py               # Block geometry from MapPLUTO
│   │   ├── compute_block_features.py     # Static urban form features
│   │   ├── build_monthly_dynamic_features.py  # Monthly traffic + activity features
│   │   ├── build_model_inputs.py         # Assemble & scale model input tensors
│   │   ├── build_nsi_input.py            # NSI model input preparation
│   │   ├── compute_dynamic_features.py   # Weekly dynamic feature alignment
│   │   ├── traffic_interpolation_quantile.py  # Traffic gap-fill + quantile calibration
│   │   ├── plot_heatmaps.py              # Static feature heatmaps
│   │   ├── plot_monthly_dynamic_visuals.py    # Monthly feature plots
│   │   └── traffic_eda.py                # Traffic exploratory analysis
│   │
│   ├── docs/                     # Documentation
│   │   ├── reports/              # Weekly progress reports (YYYY-MM-DD_<task>.md)
│   │   └── *.md                  # Analysis notes and figures documentation
│   │
│   ├── data/                     # Not committed — local only
│   │   ├── raw/                  # Raw input files (CSV, GDB, GeoJSON)
│   │   └── processed/            # Pipeline outputs (CSV, Parquet, GeoJSON)
│   │
│   ├── figures/                  # Generated plots — not committed
│   ├── config.yaml               # Runtime config (copy from config.example.yaml)
│   ├── config.example.yaml       # Config template
│   ├── requirements.txt          # Python dependencies
│   └── README.md                 # Module-level documentation
│
├── noise_complaint_data/         # Early EDA scripts
│   ├── 311_requests.py
│   ├── merge_noise_csv.py
│   ├── season_decompose_regression.py
│   └── ts_with_exog_yearly.py
│
├── classify_descriptor.ipynb     # Noise descriptor classification notebook
└── AGENTS.md                     # AI agent task rules
```

---

## Data Inputs (local only, never committed)

| File | Description |
|------|-------------|
| `data/raw/MapPLUTO25v4_unclipped.gdb` | NYC parcel data (main spatial input) |
| `data/raw/Automated_Traffic_Volume_Counts_*.csv` | 15-min traffic counts, 2000–2025 |
| `data/raw/Traffic_Volume_Counts_(Historical)_*.csv` | Daily traffic, 2012–2021 |
| `data/raw/Active_Cabaret_and_Catering_Licenses_*.csv` | Event/activity data |
| `data/raw/nyc_bgrp.geojson` | Block group boundaries |

---

## Pipeline (run from `Project_Urban_Noise/`)

```bash
# 1. Block geometries
python -m src.build_blocks --config config.yaml

# 2. Static urban form features
python -m src.compute_block_features --config config.yaml

# 3. Monthly dynamic features (traffic + activity)
python -m src.build_monthly_dynamic_features --config config.yaml

# 4. Traffic gap-fill + quantile calibration
python -m src.traffic_interpolation_quantile

# 5. Assemble model inputs
python -m src.build_model_inputs --config config.yaml
```

---

## Dependencies

```bash
pip install -r Project_Urban_Noise/requirements.txt
```

Key packages: `geopandas`, `pandas`, `numpy`, `shapely`, `scipy`, `statsmodels`, `pyarrow`, `mapclassify`
