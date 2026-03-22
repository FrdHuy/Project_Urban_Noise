"""
Traffic Data Interpolation + Quantile Curve Fitting
====================================================
Week of 2026-03-16 task (Fayer):

Step 1 – Temporal interpolation of Automated Traffic Volume Counts
  For each (SegmentID, Direction, year-month) cell that has no observed data,
  fill with the cross-segment mean for that (year, month).

Step 2 – Quantile curve calibration using Historical Traffic Volume Counts
  The Historical dataset is spatially more complete but has no dense time series.
  For each month we compute the empirical CDF of the Historical data and use it
  to remap the Automated values so they sit at the *same quantile position* in
  the Historical distribution (CDF matching / quantile normalization).

Outputs (saved under data/processed/):
  traffic_auto_monthly.csv        – Automated data aggregated to monthly totals
  traffic_hist_monthly.csv        – Historical data aggregated to monthly totals
  traffic_auto_interpolated.csv   – After Step-1 gap-fill
  traffic_combined_calibrated.csv – After Step-2 quantile curve
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]           # Project_Urban_Noise/
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

AUTO_CSV = RAW_DIR / "Automated_Traffic_Volume_Counts_20260209.csv"
HIST_CSV = RAW_DIR / "Traffic_Volume_Counts_(Historical)_20260209.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_start(year_series: pd.Series, month_series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        {"year": year_series, "month": month_series, "day": 1},
        errors="coerce",
    ).dt.to_period("M").dt.to_timestamp(how="start")


def _parse_hourly_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "am" in c.lower() or "pm" in c.lower()]


def _to_numeric_col(series: pd.Series) -> pd.Series:
    """Parse a column that may contain comma-formatted numbers."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


# ---------------------------------------------------------------------------
# Step 0 – Load & aggregate raw data to monthly per-SegmentID totals
# ---------------------------------------------------------------------------

def load_automated_monthly(path: Path) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        SegmentID, Direction, month, vol_sum, obs_count
    """
    LOG.info("Loading Automated Traffic: %s", path)
    usecols = ["Yr", "M", "D", "Vol", "SegmentID", "Direction"]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)

    raw["Vol"] = pd.to_numeric(raw["Vol"], errors="coerce")
    raw["SegmentID"] = pd.to_numeric(raw["SegmentID"], errors="coerce")
    raw["month"] = _month_start(raw["Yr"], raw["M"])
    raw = raw.dropna(subset=["month", "SegmentID", "Vol"])

    agg = (
        raw.groupby(["SegmentID", "Direction", "month"], as_index=False)
        .agg(vol_sum=("Vol", "sum"), obs_count=("Vol", "size"))
    )
    LOG.info("Automated monthly rows: %s  unique segments: %s", len(agg), agg["SegmentID"].nunique())
    return agg


def load_historical_monthly(path: Path) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        SegmentID, Direction, month, hist_daily_total
    Historical data stores 24 hourly columns per observation day; we sum them
    to get a daily total and then sum again to get a monthly total per segment.
    """
    LOG.info("Loading Historical Traffic: %s", path)
    df = pd.read_csv(path, low_memory=False)

    hour_cols = _parse_hourly_cols(df)
    for c in hour_cols:
        df[c] = _to_numeric_col(df[c])

    df["daily_total"] = df[hour_cols].fillna(0.0).sum(axis=1)
    df["month"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
    df["SegmentID"] = pd.to_numeric(df["SegmentID"], errors="coerce")
    df = df.dropna(subset=["month", "SegmentID"])

    agg = (
        df.groupby(["SegmentID", "Direction", "month"], as_index=False)
        .agg(hist_daily_total=("daily_total", "sum"), hist_obs_days=("daily_total", "size"))
    )
    LOG.info("Historical monthly rows: %s  unique segments: %s", len(agg), agg["SegmentID"].nunique())
    return agg


# ---------------------------------------------------------------------------
# Step 1 – Temporal interpolation by year-month mean
# ---------------------------------------------------------------------------

def interpolate_by_year_month_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (SegmentID, Direction) pair, build a complete grid of months
    between its first and last observed month.  Fill missing cells with the
    cross-segment mean for that (year, month) combination.

    Input / output columns: SegmentID, Direction, month, vol_sum, obs_count
    The interpolated flag marks synthetically filled rows.
    """
    LOG.info("Step 1 – interpolating gaps by year-month mean …")

    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])

    # --- compute year-month means across all segments (reference table) ---
    df["year"] = df["month"].dt.year
    df["month_of_year"] = df["month"].dt.month
    ym_mean = (
        df.groupby(["year", "month_of_year"])["vol_sum"]
        .mean()
        .rename("ym_mean")
        .reset_index()
    )

    # --- build full (SegmentID, Direction, month) grid ---
    seg_range = (
        df.groupby(["SegmentID", "Direction"])["month"]
        .agg(["min", "max"])
        .reset_index()
    )
    records = []
    for _, row in seg_range.iterrows():
        months = pd.date_range(row["min"], row["max"], freq="MS")
        for m in months:
            records.append(
                {"SegmentID": row["SegmentID"], "Direction": row["Direction"], "month": m}
            )
    grid = pd.DataFrame(records)
    grid["year"] = grid["month"].dt.year
    grid["month_of_year"] = grid["month"].dt.month

    # --- merge observed data onto grid ---
    merged = grid.merge(
        df[["SegmentID", "Direction", "month", "vol_sum", "obs_count"]],
        on=["SegmentID", "Direction", "month"],
        how="left",
    )

    # --- merge year-month means ---
    merged = merged.merge(ym_mean, on=["year", "month_of_year"], how="left")

    # --- fill gaps ---
    gap_mask = merged["vol_sum"].isna()
    merged.loc[gap_mask, "vol_sum"] = merged.loc[gap_mask, "ym_mean"]
    merged.loc[gap_mask, "obs_count"] = 0
    merged["interpolated"] = gap_mask.astype(int)

    merged = merged.drop(columns=["year", "month_of_year", "ym_mean"])

    n_filled = gap_mask.sum()
    n_total = len(merged)
    LOG.info(
        "Interpolated %s / %s cells (%.1f%% gap-fill rate)",
        n_filled, n_total, 100.0 * n_filled / n_total if n_total else 0,
    )
    return merged


# ---------------------------------------------------------------------------
# Step 2 – Quantile curve calibration
# ---------------------------------------------------------------------------

def quantile_curve_calibrate(
    auto_df: pd.DataFrame,
    hist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remap automated traffic values so that each observation sits at the *same
    quantile position* in the Historical distribution (CDF matching).

    Strategy:
      - For the overlapping set of months, build the empirical CDF of the
        Historical `hist_daily_total`.
      - For each Automated row, compute its rank in the Automated distribution
        for that month → look up the corresponding Historical quantile value.
      - Store the calibrated value as `vol_calibrated`.

    For months with no Historical data we fall back to the raw `vol_sum`.
    """
    LOG.info("Step 2 – quantile curve calibration …")

    auto = auto_df.copy()
    auto["month"] = pd.to_datetime(auto["month"])

    # Aggregate Historical to month-level distribution (ignore Direction split
    # since Historical coverage is sparser)
    hist = hist_df.copy()
    hist["month"] = pd.to_datetime(hist["month"])
    hist_month = (
        hist.groupby(["SegmentID", "month"])["hist_daily_total"]
        .sum()
        .reset_index()
        .rename(columns={"hist_daily_total": "h_vol"})
    )

    # For each month: build sorted hist distribution → interpolation function
    auto["vol_calibrated"] = np.nan

    months_with_hist = hist_month["month"].unique()

    for month in sorted(months_with_hist):
        h_vals = hist_month.loc[hist_month["month"] == month, "h_vol"].dropna().values
        if len(h_vals) < 2:
            continue

        h_sorted = np.sort(h_vals)
        h_quantiles = np.linspace(0, 1, len(h_sorted))

        # Automated rows for this month
        mask = auto["month"] == month
        a_vals = auto.loc[mask, "vol_sum"].values
        if len(a_vals) == 0:
            continue

        # Rank each automated value within its own monthly distribution
        # → map to Historical quantile value
        a_sorted_idx = np.argsort(a_vals)
        a_ranks = np.empty(len(a_vals))
        a_ranks[a_sorted_idx] = np.linspace(0, 1, len(a_vals))

        # Interpolate: quantile → hist value
        calibrated = np.interp(a_ranks, h_quantiles, h_sorted)
        auto.loc[mask, "vol_calibrated"] = calibrated

    # For months without Historical data keep raw vol_sum
    no_cal_mask = auto["vol_calibrated"].isna()
    auto.loc[no_cal_mask, "vol_calibrated"] = auto.loc[no_cal_mask, "vol_sum"]

    n_calibrated = (~no_cal_mask).sum()
    LOG.info(
        "Calibrated %s / %s rows using Historical distribution (%.1f%%)",
        n_calibrated, len(auto), 100.0 * n_calibrated / len(auto) if len(auto) else 0,
    )
    return auto


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(auto_raw: pd.DataFrame, auto_interp: pd.DataFrame, calibrated: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)

    print("\n[Raw Automated Monthly]")
    print(f"  Rows: {len(auto_raw):,}")
    print(f"  Unique segments: {auto_raw['SegmentID'].nunique():,}")
    print(f"  Month range: {auto_raw['month'].min()} → {auto_raw['month'].max()}")
    print(f"  vol_sum stats:\n{auto_raw['vol_sum'].describe().to_string()}")

    print("\n[After Step-1 Interpolation]")
    print(f"  Total rows (grid): {len(auto_interp):,}")
    n_interp = auto_interp['interpolated'].sum()
    print(f"  Interpolated cells: {n_interp:,} ({100*n_interp/len(auto_interp):.1f}%)")
    print(f"  vol_sum stats (all):\n{auto_interp['vol_sum'].describe().to_string()}")

    print("\n[After Step-2 Quantile Calibration]")
    cal_mask = ~calibrated['vol_calibrated'].isna()
    print(f"  Rows with calibration: {cal_mask.sum():,}")
    print(f"  vol_calibrated stats:\n{calibrated['vol_calibrated'].describe().to_string()}")

    # Quantile comparison: before vs after for calibrated months
    overlap = calibrated.dropna(subset=['vol_calibrated'])
    print("\n[Quantile comparison (calibrated months)]")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        v_raw = overlap['vol_sum'].quantile(q)
        v_cal = overlap['vol_calibrated'].quantile(q)
        print(f"  P{int(q*100):02d}: raw={v_raw:,.0f}  calibrated={v_cal:,.0f}  ratio={v_cal/v_raw:.3f}" if v_raw else "  P{} raw=0".format(int(q*100)))

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load raw data ---
    auto_monthly = load_automated_monthly(AUTO_CSV)
    hist_monthly = load_historical_monthly(HIST_CSV)

    # --- Save intermediate aggregations ---
    auto_monthly.to_csv(PROC_DIR / "traffic_auto_monthly.csv", index=False)
    hist_monthly.to_csv(PROC_DIR / "traffic_hist_monthly.csv", index=False)
    LOG.info("Saved raw monthly aggregations.")

    # --- Step 1: interpolate ---
    auto_interpolated = interpolate_by_year_month_mean(auto_monthly)
    auto_interpolated.to_csv(PROC_DIR / "traffic_auto_interpolated.csv", index=False)
    LOG.info("Saved interpolated data → traffic_auto_interpolated.csv")

    # --- Step 2: quantile curve ---
    calibrated = quantile_curve_calibrate(auto_interpolated, hist_monthly)
    calibrated.to_csv(PROC_DIR / "traffic_combined_calibrated.csv", index=False)
    LOG.info("Saved calibrated data → traffic_combined_calibrated.csv")

    # --- Diagnostics ---
    print_diagnostics(auto_monthly, auto_interpolated, calibrated)

    print("\nDone. Output files in:", PROC_DIR)


if __name__ == "__main__":
    main()
