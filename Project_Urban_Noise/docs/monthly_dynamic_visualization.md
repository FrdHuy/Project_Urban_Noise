# Monthly Dynamic Data Visualization

This note explains the four figures created from the monthly dynamic outputs. The goal is to show temporal traffic patterns, nonzero traffic coverage, the behavior of the unified monthly feature table, and the quality profile of the final dataset.

## Files Used

- `data/processed/traffic_monthly_summary.csv`
- `data/processed/traffic_monthly_nonzero.csv`
- `data/processed/monthly_dynamic_with_activity.csv`
- `data/processed/monthly_dynamic_quality.csv`

## Figures

### 1. Monthly Traffic Summary Across NYC

Figure: `figures/monthly_traffic_summary.png`

This figure shows three monthly trends:

- `traffic_volume_sum`: total traffic volume aggregated across all observed BGRP-month records
- `traffic_obs_count`: total number of traffic observations in each month
- `bgrp_nonzero_traffic_volume`: number of BGRPs with nonzero traffic volume in each month

This is the best high-level summary of the traffic pipeline because it shows both signal magnitude and coverage over time.

### 2. Distribution of Nonzero Traffic Observations

Figure: `figures/monthly_nonzero_traffic_distribution.png`

This figure summarizes `traffic_monthly_nonzero.csv` at the yearly level:

- top panel: count of active BGRP-month observations with nonzero traffic
- bottom panel: median and 90th percentile of nonzero traffic volume by year

This avoids an unreadable full `bgrp_id x month` heatmap while still showing how the nonzero observations are distributed over time.

### 3. Temporal Trends in Monthly Dynamic Features

Figure: `figures/monthly_dynamic_feature_trends.png`

This figure uses the unified monthly table keyed by `bgrp_id, month` and aggregates it to citywide monthly totals.

Top panel:

- `traffic_volume_sum`
- `traffic_obs_count`
- `traffic_hist_daily_sum`

Bottom panel:

- `event_new_count`
- `event_active_count`

This figure shows how traffic-related and activity-related features evolve over time after integration into a single monthly table.

### 4. Monthly Dynamic Data Quality Check

Figure: `figures/monthly_dynamic_quality_check.png`

This figure visualizes:

- `missing_rate` by feature
- `zero_rate` by feature

The rates are high for several traffic columns because the final monthly table is built on the full `bgrp_id x month` grid. Most BGRP-month combinations have no direct observation, so sparsity is expected and should not automatically be interpreted as data corruption.

## Interpretation Notes

- The monthly dynamic table is intentionally wide in coverage and sparse in observed traffic values.
- `traffic_*` columns are mostly zero or missing because traffic observations exist for a limited subset of BGRPs and months.
- `event_active_count` is denser than `event_new_count` because license activity persists across multiple months.
- The quality chart should be discussed together with the design choice of constructing a full monthly panel.

## How To Regenerate

Run from the repository root:

```powershell
python -m src.plot_monthly_dynamic_visuals --config config.yaml
```
