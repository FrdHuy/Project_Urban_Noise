# Project_Urban_Noise

Compute NYC **block-level** urban form features from MapPLUTO + building footprint/3D metrics.

## Inputs (local, not committed)
Put these under `data/raw/`:
- `MapPLUTO25v4_unclipped.gdb/`
- `bldg_3d_metrics.csv`
- `BUILDING_20260209.csv`
- `Centerline_20260209.csv` (optional, for street H/W)

## Outputs
- `data/processed/blocks.geojson`
- `data/processed/block_features.geojson`
- `data/processed/block_features.csv`
- `figures/heatmap_building_density.png`
- `figures/heatmap_height_heterogeneity.png`

## Run
1. Copy `config.example.yaml` to `config.yaml` and edit paths if needed.
2. `python -m src.build_blocks --config config.yaml`
3. `python -m src.compute_block_features --config config.yaml`
4. `python -m src.plot_heatmaps --config config.yaml`
