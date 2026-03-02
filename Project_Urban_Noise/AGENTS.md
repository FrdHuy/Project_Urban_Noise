@"
# AGENTS.md

## Project rules
- Never commit raw datasets (data/raw/*, *.gdb, large CSV/ZIP). Keep them local and gitignored.
- Scripts must be runnable from repo root with `--config config.yaml`.
- Use EPSG:2263 for area/length calculations; validate CRS before computing.
- Prefer key-based joins (Borough+Block, BBL) over spatial joins; spatial join only as fallback.

## Team lead task (primary)
Compute block-level urban form features and export for integration:
- street height-to-width ratio (H/W) (use Centerline width if available; otherwise output proxy and document)
- building density
- FAR
- sky openness proxy
- height heterogeneity
  Also generate heatmaps for building density and height heterogeneity.
  "@ | Out-File -Encoding utf8 AGENTS.md