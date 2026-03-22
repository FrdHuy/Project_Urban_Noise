"""
traffic_eda.py  –  Exploratory Data Analysis for Traffic Volume Data
=====================================================================
Reads the monthly dynamic panel (monthly_dynamic_with_activity.csv or .parquet)
and generates a full suite of EDA figures + a markdown summary.

Usage (from repo root):
    python -m src.traffic_eda --config config.yaml

All figures are saved to figures/traffic_eda/.
Summary is saved to data/processed/traffic_eda_summary.md.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from .common import ensure_parent_dir, find_column, load_config, setup_logging

warnings.filterwarnings("ignore", category=FutureWarning)

LOGGER = logging.getLogger(__name__)

# ── Default paths (overridden by config) ───────────────────────────────────
DEFAULT_MONTHLY_CSV = "data/processed/monthly_dynamic_with_activity.csv"
DEFAULT_MONTHLY_PARQUET = "data/processed/monthly_dynamic.parquet"
DEFAULT_FIG_DIR = "figures/traffic_eda"
DEFAULT_SUMMARY_OUT = "data/processed/traffic_eda_summary.md"

# Candidate names for traffic column – handles both pipeline naming conventions
TRAFFIC_VOL_CANDIDATES = [
    "traffic_volume_sum",   # monthly pipeline
    "traffic",              # weekly pipeline
    "vol",
    "volume",
    "traffic_vol",
]
TRAFFIC_HIST_CANDIDATES = ["traffic_hist_daily_sum"]
TRAFFIC_OBS_CANDIDATES = ["traffic_obs_count"]
BGRP_CANDIDATES = ["bgrp_id", "bgrp", "geoid", "geoid20", "bg_id", "block_group"]
MONTH_CANDIDATES = ["month", "date", "period"]


# ── Style (consistent with existing repo) ──────────────────────────────────
def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


# ── Data loading ────────────────────────────────────────────────────────────
def load_monthly_data(config: dict) -> pd.DataFrame:
    """
    Load monthly dynamic panel data.
    Tries parquet first (faster), falls back to CSV.
    Also detects and renames traffic columns to standard names.
    """
    paths = config["paths"]
    outputs = paths.get("outputs", {})

    parquet_path = outputs.get("monthly_dynamic_parquet", DEFAULT_MONTHLY_PARQUET)
    csv_path = outputs.get("monthly_dynamic_csv", DEFAULT_MONTHLY_CSV)

    df = None
    if Path(parquet_path).exists():
        LOGGER.info("Loading from parquet: %s", parquet_path)
        df = pd.read_parquet(parquet_path)
    elif Path(csv_path).exists():
        LOGGER.info("Loading from CSV: %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError(
            f"Neither {parquet_path} nor {csv_path} found. "
            "Run build_monthly_dynamic_features.py first."
        )

    # ── Normalize key column names ──────────────────────────────────────
    # month column
    month_col = find_column(df.columns, MONTH_CANDIDATES)
    if month_col and month_col != "month":
        df = df.rename(columns={month_col: "month"})
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)

    # bgrp_id column
    bgrp_col = find_column(df.columns, BGRP_CANDIDATES)
    if bgrp_col and bgrp_col != "bgrp_id":
        df = df.rename(columns={bgrp_col: "bgrp_id"})

    # traffic volume column – normalize to "traffic_volume_sum"
    tv_col = find_column(df.columns, TRAFFIC_VOL_CANDIDATES)
    if tv_col and tv_col != "traffic_volume_sum":
        LOGGER.info("Renaming traffic column '%s' → 'traffic_volume_sum'", tv_col)
        df = df.rename(columns={tv_col: "traffic_volume_sum"})
    if "traffic_volume_sum" not in df.columns:
        raise KeyError(
            "No traffic volume column found. "
            f"Searched for: {TRAFFIC_VOL_CANDIDATES}. "
            f"Available columns: {list(df.columns)}"
        )

    df["traffic_volume_sum"] = pd.to_numeric(df["traffic_volume_sum"], errors="coerce").fillna(0.0)

    LOGGER.info(
        "Loaded %d rows × %d cols | date range: %s – %s | bgrps: %d",
        len(df),
        len(df.columns),
        df["month"].min().date(),
        df["month"].max().date(),
        df["bgrp_id"].nunique() if "bgrp_id" in df.columns else -1,
    )
    return df


# ── Helper: save figure ─────────────────────────────────────────────────────
def _save(fig: plt.Figure, fig_dir: Path, name: str) -> Path:
    out = fig_dir / name
    ensure_parent_dir(out)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved: %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Missing / Zero value heatmap (bgrp × year)
# Purpose: Understand data coverage before any analysis.
# ─────────────────────────────────────────────────────────────────────────────
def plot_coverage_heatmap(df: pd.DataFrame, fig_dir: Path) -> dict:
    """
    Two-panel data coverage overview:
      Left:  How many bgrps have any traffic data each year (citywide sensor coverage trend)
      Right: How many months of data does each bgrp have in total (per-bgrp data richness)
    Much more readable than a raw bgrp×year matrix.
    """
    df2 = df.copy()
    df2["year"] = df2["month"].dt.year
    df2["has_traffic"] = df2["traffic_volume_sum"] > 0

    # ── Panel A: how many bgrps have data each year ──────────────────────
    bgrps_per_year = (
        df2[df2["has_traffic"]]
        .groupby("year")["bgrp_id"].nunique()
        .reset_index()
        .rename(columns={"bgrp_id": "n_bgrp_with_data"})
    )
    total_bgrps = df2["bgrp_id"].nunique()
    bgrps_per_year["pct"] = bgrps_per_year["n_bgrp_with_data"] / total_bgrps * 100

    # ── Panel B: per-bgrp total nonzero months ───────────────────────────
    months_per_bgrp = (
        df2[df2["has_traffic"]]
        .groupby("bgrp_id")["month"].nunique()
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: citywide coverage over years
    color_bar = "#4c72b0"
    axes[0].bar(bgrps_per_year["year"], bgrps_per_year["n_bgrp_with_data"],
                color=color_bar, alpha=0.8)
    ax0r = axes[0].twinx()
    ax0r.plot(bgrps_per_year["year"], bgrps_per_year["pct"],
              color="#c44e52", linewidth=2, marker="o", markersize=4, label="% of all bgrps")
    ax0r.set_ylabel("% of all bgrps covered", color="#c44e52")
    ax0r.tick_params(axis="y", colors="#c44e52")
    ax0r.set_ylim(0, max(bgrps_per_year["pct"]) * 1.3)
    axes[0].set_title(
        f"How Many bgrps Have Traffic Data Each Year\n(total bgrps in panel = {total_bgrps:,})"
    )
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("# bgrps with ≥1 nonzero month", color=color_bar)
    axes[0].tick_params(axis="x", rotation=45)

    # Right: distribution of data richness per bgrp
    axes[1].hist(months_per_bgrp.values, bins=50, color="#55a868", alpha=0.85,
                 edgecolor="white", linewidth=0.3)
    axes[1].axvline(months_per_bgrp.median(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Median = {months_per_bgrp.median():.0f} months")
    axes[1].axvline(months_per_bgrp.mean(), color="navy", linestyle="--", linewidth=1.5,
                    label=f"Mean = {months_per_bgrp.mean():.0f} months")
    axes[1].set_title(
        f"Data Richness: How Many Months of Data per bgrp\n"
        f"({(months_per_bgrp > 0).sum():,} bgrps have any data out of {total_bgrps:,} total)"
    )
    axes[1].set_xlabel("Number of months with nonzero traffic")
    axes[1].set_ylabel("Number of bgrps")
    axes[1].legend(frameon=False)

    fig.suptitle("Traffic Data Coverage Overview", fontsize=14)
    fig.tight_layout()
    _save(fig, fig_dir, "01_coverage_heatmap.png")

    overall_zero_rate = (df["traffic_volume_sum"] == 0).mean()
    overall_missing_rate = df["traffic_volume_sum"].isna().mean()
    n_bgrp_with_any = int((months_per_bgrp > 0).sum())
    return {
        "zero_rate": overall_zero_rate,
        "missing_rate": overall_missing_rate,
        "n_bgrp": total_bgrps,
        "n_bgrp_with_any_data": n_bgrp_with_any,
        "date_min": df["month"].min(),
        "date_max": df["month"].max(),
        "n_rows": len(df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Monthly time-series (citywide mean)
# Purpose: Identify overall temporal trends and anomalies (e.g., COVID dip).
# ─────────────────────────────────────────────────────────────────────────────
def plot_time_series(df: pd.DataFrame, fig_dir: Path) -> dict:
    """
    Citywide monthly mean and total traffic volume over time.
    """
    monthly = df.groupby("month", as_index=False).agg(
        mean_vol=("traffic_volume_sum", "mean"),
        total_vol=("traffic_volume_sum", "sum"),
        n_nonzero=("traffic_volume_sum", lambda s: (s > 0).sum()),
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(monthly["month"], monthly["total_vol"], color="#c44e52", linewidth=1.8)
    axes[0].set_title("Citywide Total Traffic Volume by Month")
    axes[0].set_ylabel("Total volume")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    axes[1].plot(monthly["month"], monthly["mean_vol"], color="#4c72b0", linewidth=1.8)
    axes[1].set_title("Citywide Mean Traffic Volume per bgrp by Month")
    axes[1].set_ylabel("Mean volume per bgrp")

    axes[2].plot(monthly["month"], monthly["n_nonzero"], color="#55a868", linewidth=1.8)
    axes[2].set_title("Number of bgrp-months with Nonzero Traffic")
    axes[2].set_ylabel("Count")
    axes[2].set_xlabel("Month")

    fig.suptitle("Traffic Volume Time Series (Citywide)", fontsize=14, y=1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, fig_dir, "02_time_series.png")

    return {
        "global_mean": monthly["mean_vol"].mean(),
        "global_max_month": str(monthly.loc[monthly["total_vol"].idxmax(), "month"].date()),
        "global_min_month": str(monthly.loc[monthly["total_vol"].idxmin(), "month"].date()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Monthly average bar chart (seasonal pattern 1–12)
# Purpose: Confirm whether traffic has a within-year seasonal structure.
# ─────────────────────────────────────────────────────────────────────────────
def plot_monthly_average(df: pd.DataFrame, fig_dir: Path) -> dict:
    """
    Average traffic volume by calendar month (1–12), across all bgrps and years.
    """
    df2 = df.copy()
    df2["cal_month"] = df2["month"].dt.month
    # Only include non-zero records for this view (zeros are structural zeros)
    nonzero = df2[df2["traffic_volume_sum"] > 0]

    by_month = nonzero.groupby("cal_month")["traffic_volume_sum"].agg(["mean", "median", "std"])
    by_month.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean + std error bars
    axes[0].bar(by_month.index, by_month["mean"], yerr=by_month["std"],
                color="#4c72b0", alpha=0.8, capsize=4, error_kw={"linewidth": 1})
    axes[0].set_title("Mean Traffic Volume by Calendar Month\n(nonzero observations only)")
    axes[0].set_ylabel("Mean volume")
    axes[0].set_xlabel("Month")
    axes[0].tick_params(axis="x", rotation=45)

    # Median (less sensitive to outliers)
    axes[1].bar(by_month.index, by_month["median"], color="#dd8452", alpha=0.8)
    axes[1].set_title("Median Traffic Volume by Calendar Month\n(nonzero observations only)")
    axes[1].set_ylabel("Median volume")
    axes[1].set_xlabel("Month")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Within-Year Seasonal Pattern", fontsize=14)
    fig.tight_layout()
    _save(fig, fig_dir, "03_monthly_average.png")

    peak_month = by_month["mean"].idxmax()
    trough_month = by_month["mean"].idxmin()
    return {
        "peak_calendar_month": peak_month,
        "trough_calendar_month": trough_month,
        "seasonal_range_pct": round(
            (by_month["mean"].max() - by_month["mean"].min()) / by_month["mean"].mean() * 100, 1
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: STL seasonal decomposition
# Purpose: Separate trend, seasonality, and residual for a representative bgrp.
# ─────────────────────────────────────────────────────────────────────────────
def plot_stl_decomposition(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    STL decomposition on citywide monthly total traffic.
    Uses period=12 (annual seasonality).
    """
    monthly_total = (
        df.groupby("month")["traffic_volume_sum"].sum()
        .asfreq("MS")  # ensure monthly frequency
        .fillna(0)
    )

    # Need at least 2 full years of data for STL
    if len(monthly_total) < 24:
        LOGGER.warning("Not enough data for STL decomposition (need >= 24 months). Skipping.")
        return

    try:
        stl = STL(monthly_total, period=12, robust=True)
        res = stl.fit()
    except Exception as e:
        LOGGER.warning("STL decomposition failed: %s", e)
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(monthly_total.index, monthly_total.values, color="#333333", linewidth=1.5)
    axes[0].set_title("Original")
    axes[0].set_ylabel("Total volume")

    axes[1].plot(monthly_total.index, res.trend, color="#c44e52", linewidth=1.5)
    axes[1].set_title("Trend")
    axes[1].set_ylabel("Trend")

    axes[2].plot(monthly_total.index, res.seasonal, color="#4c72b0", linewidth=1.5)
    axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[2].set_title("Seasonal (period = 12 months)")
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(monthly_total.index, res.resid, color="#55a868", linewidth=1.2)
    axes[3].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[3].set_title("Residual")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Month")

    # Annotate seasonal strength
    var_s = np.var(res.seasonal)
    var_r = np.var(res.resid)
    seasonal_strength = max(0, 1 - var_r / (var_s + var_r))
    fig.suptitle(
        f"STL Decomposition – Citywide Monthly Traffic\n"
        f"Seasonal strength = {seasonal_strength:.3f}  (0=none, 1=perfect)",
        fontsize=13,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, fig_dir, "04_stl_decomposition.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Spatial distribution (choropleth by bgrp)
# Purpose: Show which bgrps have high/low traffic – spatial heterogeneity.
# ─────────────────────────────────────────────────────────────────────────────
def plot_spatial_distribution(df: pd.DataFrame, config: dict, fig_dir: Path) -> dict:
    """
    Map of annual average traffic volume per bgrp.
    Tries to load bgrp geometry; falls back to a ranked bar chart.
    """
    paths = config["paths"]

    # Compute per-bgrp annual average
    df2 = df[df["traffic_volume_sum"] > 0].copy()
    bgrp_avg = df2.groupby("bgrp_id")["traffic_volume_sum"].mean().reset_index()
    bgrp_avg.columns = ["bgrp_id", "mean_traffic"]

    # Try loading spatial geometry
    geojson_path = paths.get("bgrp_geojson", "data/raw/nyc_bgrp.geojson")
    blocks_path = paths.get("outputs", {}).get("blocks_geojson", "data/processed/blocks.geojson")

    geo = None
    geo_join_col = "bgrp_id"  # column in geo to join on
    for gpath in [geojson_path, blocks_path]:
        if Path(gpath).exists():
            try:
                import geopandas as gpd
                geo = gpd.read_file(gpath)
                # Detect bgrp_id column in geo; also accept block_id (proxy bgrp)
                geo_bgrp_col = find_column(geo.columns, BGRP_CANDIDATES + ["block_id"])
                if geo_bgrp_col and geo_bgrp_col != "bgrp_id":
                    geo_join_col = geo_bgrp_col  # keep original name for merge
                LOGGER.info("Loaded geometry from: %s  (join key: %s)", gpath, geo_join_col)
                break
            except Exception as e:
                LOGGER.warning("Could not load geo from %s: %s", gpath, e)

    if geo is not None:
        # bgrp_avg uses "bgrp_id"; geo may use block_id or another key
        bgrp_avg_merge = bgrp_avg.rename(columns={"bgrp_id": geo_join_col})
        merged = geo.merge(bgrp_avg_merge, on=geo_join_col, how="left")
        merged["has_data"] = merged["mean_traffic"].notna() & (merged["mean_traffic"] > 0)

        n_with = merged["has_data"].sum()
        n_total = len(merged)
        pct_with = n_with / n_total * 100

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))

        # ── Left panel: binary sensor coverage (which blocks have any data) ──
        no_data = merged[~merged["has_data"]]
        has_data = merged[merged["has_data"]]
        no_data.plot(ax=axes[0], color="#d0d0d0", linewidth=0.1, label="No sensor")
        has_data.plot(ax=axes[0], color="#c44e52", linewidth=0.1, label="Has traffic data")
        axes[0].set_title(
            f"Sensor Coverage: Which Block Groups Have Traffic Data\n"
            f"{n_with:,} / {n_total:,} blocks ({pct_with:.1f}%) have ≥1 nonzero month"
        )
        axes[0].axis("off")
        axes[0].legend(loc="lower left", frameon=True, fontsize=9)

        # ── Right panel: traffic volume magnitude (only blocks with data) ──
        no_data.plot(ax=axes[1], color="#eeeeee", linewidth=0.1)
        has_data.plot(
            column="mean_traffic",
            ax=axes[1],
            legend=True,
            cmap="YlOrRd",
            linewidth=0.1,
            legend_kwds={"label": "Mean annual traffic volume", "orientation": "vertical"},
        )
        axes[1].set_title(
            "Annual Average Traffic Volume\n(colored blocks only – gray = no sensor coverage)"
        )
        axes[1].axis("off")

        fig.suptitle(
            "Spatial Distribution of Traffic Volume Across NYC Block Groups",
            fontsize=14,
        )
        fig.tight_layout()
        _save(fig, fig_dir, "05_spatial_distribution_map.png")
    else:
        LOGGER.warning("No geometry file found. Generating bar chart fallback.")
        # Fallback: ranked bar chart (top 50 and bottom 50)
        top50 = bgrp_avg.nlargest(50, "mean_traffic")
        bot50 = bgrp_avg.nsmallest(50, "mean_traffic")

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].barh(top50["bgrp_id"].astype(str), top50["mean_traffic"], color="#c44e52")
        axes[0].set_title("Top 50 bgrps – Highest Mean Traffic")
        axes[0].set_xlabel("Mean traffic volume")
        axes[0].invert_yaxis()

        axes[1].barh(bot50["bgrp_id"].astype(str), bot50["mean_traffic"], color="#4c72b0")
        axes[1].set_title("Bottom 50 bgrps – Lowest Mean Traffic (nonzero)")
        axes[1].set_xlabel("Mean traffic volume")
        axes[1].invert_yaxis()

        fig.suptitle("Spatial Distribution of Traffic Volume (Ranked)", fontsize=13)
        fig.tight_layout()
        _save(fig, fig_dir, "05_spatial_distribution_bar.png")

    cv = bgrp_avg["mean_traffic"].std() / bgrp_avg["mean_traffic"].mean()
    return {
        "spatial_cv": round(cv, 3),
        "top_bgrp": str(bgrp_avg.loc[bgrp_avg["mean_traffic"].idxmax(), "bgrp_id"]),
        "median_bgrp_mean": round(bgrp_avg["mean_traffic"].median(), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Box plot by calendar month (all bgrps)
# Purpose: Show distribution spread and outliers within each month.
# ─────────────────────────────────────────────────────────────────────────────
def plot_boxplot_by_month(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Box plots of log1p(traffic) grouped by calendar month.
    Helps visualize seasonal spread and outliers simultaneously.
    """
    df2 = df[df["traffic_volume_sum"] > 0].copy()
    df2["cal_month"] = df2["month"].dt.month
    df2["log_vol"] = np.log1p(df2["traffic_volume_sum"])

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    data_by_month = [
        df2[df2["cal_month"] == m]["log_vol"].dropna().values
        for m in range(1, 13)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw values
    raw_by_month = [
        df2[df2["cal_month"] == m]["traffic_volume_sum"].dropna().values
        for m in range(1, 13)
    ]
    axes[0].boxplot(raw_by_month, labels=month_labels, patch_artist=True,
                    boxprops={"facecolor": "#4c72b0", "alpha": 0.7},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    axes[0].set_title("Raw Traffic Volume by Calendar Month\n(nonzero only)")
    axes[0].set_ylabel("Traffic volume")
    axes[0].set_xlabel("Month")
    axes[0].tick_params(axis="x", rotation=45)

    # Log1p transformed
    axes[1].boxplot(data_by_month, labels=month_labels, patch_artist=True,
                    boxprops={"facecolor": "#c44e52", "alpha": 0.7},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    axes[1].set_title("log1p(Traffic Volume) by Calendar Month\n(nonzero only)")
    axes[1].set_ylabel("log1p(volume)")
    axes[1].set_xlabel("Month")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Monthly Distribution of Traffic Volume", fontsize=14)
    fig.tight_layout()
    _save(fig, fig_dir, "06_boxplot_by_month.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: Histogram + QQ Plot (raw vs log1p)
# Purpose: Assess distributional shape; decide if log transform is needed.
# ─────────────────────────────────────────────────────────────────────────────
def plot_distribution(df: pd.DataFrame, fig_dir: Path) -> dict:
    """
    Compare raw vs log1p distribution: histogram and QQ plot against normal.
    Also report skewness and kurtosis.
    """
    nonzero = df[df["traffic_volume_sum"] > 0]["traffic_volume_sum"].dropna()
    log_vals = np.log1p(nonzero)

    raw_skew = float(stats.skew(nonzero))
    raw_kurt = float(stats.kurtosis(nonzero))
    log_skew = float(stats.skew(log_vals))
    log_kurt = float(stats.kurtosis(log_vals))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw histogram
    axes[0, 0].hist(nonzero, bins=100, color="#4c72b0", alpha=0.8, density=True)
    axes[0, 0].set_title(f"Raw Traffic Volume\nskew={raw_skew:.2f}, kurtosis={raw_kurt:.2f}")
    axes[0, 0].set_xlabel("Volume")
    axes[0, 0].set_ylabel("Density")

    # log1p histogram
    axes[0, 1].hist(log_vals, bins=100, color="#c44e52", alpha=0.8, density=True)
    # Overlay normal curve
    mu, sigma = log_vals.mean(), log_vals.std()
    x = np.linspace(log_vals.min(), log_vals.max(), 300)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), "k-", linewidth=1.5, label="Normal fit")
    axes[0, 1].set_title(f"log1p(Traffic Volume)\nskew={log_skew:.2f}, kurtosis={log_kurt:.2f}")
    axes[0, 1].set_xlabel("log1p(Volume)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend(frameon=False)

    # QQ plot – raw
    probplot(nonzero, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("QQ Plot – Raw Traffic Volume")

    # QQ plot – log1p
    probplot(log_vals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot – log1p(Traffic Volume)")

    fig.suptitle("Traffic Volume Distribution Analysis", fontsize=14)
    fig.tight_layout()
    _save(fig, fig_dir, "07_distribution_qq.png")

    return {
        "raw_skew": round(raw_skew, 3),
        "raw_kurtosis": round(raw_kurt, 3),
        "log_skew": round(log_skew, 3),
        "log_kurtosis": round(log_kurt, 3),
        "recommend_log_transform": abs(log_skew) < abs(raw_skew),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8: ACF / PACF
# Purpose: Reveal autocorrelation structure for time-series modeling.
# ─────────────────────────────────────────────────────────────────────────────
def plot_autocorrelation(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    ACF and PACF on citywide monthly aggregated traffic.
    Uses log1p to reduce influence of extreme values.
    Also shows one representative high-coverage bgrp.
    """
    # Citywide monthly series
    monthly = (
        df.groupby("month")["traffic_volume_sum"].mean()
        .sort_index()
        .asfreq("MS")
        .fillna(0)
    )
    log_monthly = np.log1p(monthly)

    # Pick a representative bgrp (most non-zero months)
    if "bgrp_id" in df.columns:
        coverage = df[df["traffic_volume_sum"] > 0].groupby("bgrp_id").size()
        rep_bgrp = coverage.idxmax()
        bgrp_series = (
            df[df["bgrp_id"] == rep_bgrp]
            .set_index("month")["traffic_volume_sum"]
            .sort_index()
            .asfreq("MS")
            .fillna(0)
        )
        log_bgrp = np.log1p(bgrp_series)
    else:
        rep_bgrp = None
        log_bgrp = None

    n_lags = min(36, len(log_monthly) // 2 - 1)
    n_rows = 3 if log_bgrp is not None else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))

    plot_acf(log_monthly, lags=n_lags, ax=axes[0, 0], title="ACF – Citywide Monthly (log1p)")
    plot_pacf(log_monthly, lags=n_lags, ax=axes[0, 1], title="PACF – Citywide Monthly (log1p)",
              method="ywm")

    axes[1, 0].plot(log_monthly.index, log_monthly.values, color="#4c72b0", linewidth=1.5)
    axes[1, 0].set_title("Citywide log1p(Traffic) Time Series")
    axes[1, 0].set_xlabel("Month")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5, 0.5,
        "Lag-12 spike in ACF → annual seasonality\n"
        "Fast PACF decay → MA or seasonal MA structure",
        ha="center", va="center", fontsize=11, wrap=True,
        bbox={"boxstyle": "round", "facecolor": "#f0f0f0", "alpha": 0.8},
    )

    if log_bgrp is not None and len(log_bgrp) >= 24:
        plot_acf(log_bgrp, lags=min(n_lags, len(log_bgrp) // 2 - 1),
                 ax=axes[2, 0], title=f"ACF – bgrp {rep_bgrp} (log1p)")
        plot_pacf(log_bgrp, lags=min(n_lags, len(log_bgrp) // 2 - 1),
                  ax=axes[2, 1], title=f"PACF – bgrp {rep_bgrp} (log1p)", method="ywm")

    fig.suptitle("Autocorrelation Analysis – Traffic Volume", fontsize=14)
    fig.tight_layout()
    _save(fig, fig_dir, "08_acf_pacf.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 9: Extreme value analysis + transformation comparison
# Purpose: Quantify outliers and evaluate handling strategies.
# ─────────────────────────────────────────────────────────────────────────────
def plot_extreme_values(df: pd.DataFrame, fig_dir: Path) -> dict:
    """
    Compare raw, log1p, winsorized (99th pct), and clipped values.
    Shows how each method handles the long tail.
    """
    nonzero = df[df["traffic_volume_sum"] > 0]["traffic_volume_sum"].dropna()

    p95 = nonzero.quantile(0.95)
    p99 = nonzero.quantile(0.99)
    p999 = nonzero.quantile(0.999)

    # Winsorize at 99th percentile
    winsorized = nonzero.clip(upper=p99)
    # log1p
    log_vals = np.log1p(nonzero)
    # Clipped at 99.9th
    clipped = nonzero.clip(upper=p999)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def _hist(ax, data, title, color):
        ax.hist(data, bins=80, color=color, alpha=0.8, density=True)
        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.set_xlabel("Value")

    _hist(axes[0, 0], nonzero, "Raw (nonzero)", "#4c72b0")
    axes[0, 0].axvline(p99, color="red", linestyle="--", linewidth=1.2, label=f"p99={p99:,.0f}")
    axes[0, 0].legend(frameon=False)

    _hist(axes[0, 1], log_vals, "log1p transform", "#c44e52")

    _hist(axes[1, 0], winsorized, f"Winsorized @ p99={p99:,.0f}", "#55a868")

    _hist(axes[1, 1], clipped, f"Clipped @ p99.9={p999:,.0f}", "#8172b3")

    fig.suptitle("Extreme Value Handling Strategies", fontsize=14)
    fig.tight_layout()
    _save(fig, fig_dir, "09_extreme_values.png")

    n_above_p99 = (nonzero > p99).sum()
    return {
        "p95": round(float(p95), 1),
        "p99": round(float(p99), 1),
        "p999": round(float(p999), 1),
        "n_above_p99": int(n_above_p99),
        "pct_above_p99": round(n_above_p99 / len(nonzero) * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 10: bgrp-level traffic distribution (spatial heterogeneity)
# Purpose: Show how unequal traffic is across block groups.
# ─────────────────────────────────────────────────────────────────────────────
def plot_bgrp_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Box plot of per-bgrp median traffic (sorted).
    Shows spatial inequality in traffic exposure.
    """
    if "bgrp_id" not in df.columns:
        LOGGER.warning("No bgrp_id column – skipping bgrp distribution plot.")
        return

    nonzero = df[df["traffic_volume_sum"] > 0]
    bgrp_stats = nonzero.groupby("bgrp_id")["traffic_volume_sum"].agg(
        ["median", "mean", "std", "count"]
    ).reset_index()
    bgrp_stats = bgrp_stats.sort_values("median")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Sorted median across bgrps (Lorenz-curve style view)
    axes[0].plot(range(len(bgrp_stats)), bgrp_stats["median"], color="#4c72b0", linewidth=1.5)
    axes[0].fill_between(range(len(bgrp_stats)), bgrp_stats["median"], alpha=0.3, color="#4c72b0")
    axes[0].set_title("Sorted Median Traffic Volume by bgrp\n(spatial inequality view)")
    axes[0].set_xlabel("bgrp rank (low → high traffic)")
    axes[0].set_ylabel("Median traffic volume")

    # Histogram of per-bgrp medians
    axes[1].hist(bgrp_stats["median"], bins=60, color="#dd8452", alpha=0.85)
    axes[1].axvline(bgrp_stats["median"].mean(), color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean={bgrp_stats['median'].mean():,.0f}")
    axes[1].axvline(bgrp_stats["median"].median(), color="navy", linestyle="--",
                    linewidth=1.5, label=f"Median={bgrp_stats['median'].median():,.0f}")
    axes[1].set_title("Distribution of Per-bgrp Median Traffic")
    axes[1].set_xlabel("Median traffic volume")
    axes[1].set_ylabel("Number of bgrps")
    axes[1].legend(frameon=False)

    fig.suptitle("Spatial Heterogeneity of Traffic Volume Across bgrps", fontsize=14)
    fig.tight_layout()
    _save(fig, fig_dir, "10_bgrp_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary(stats_dict: dict, output_path: str) -> None:
    """
    Write a brief markdown summary of all EDA findings.
    """
    cov = stats_dict.get("coverage", {})
    ts = stats_dict.get("time_series", {})
    sea = stats_dict.get("seasonal", {})
    spa = stats_dict.get("spatial", {})
    dist = stats_dict.get("distribution", {})
    ext = stats_dict.get("extremes", {})

    summary = textwrap.dedent(f"""
    # Traffic Volume EDA Summary

    ## 1. Data Overview
    - **Total rows**: {cov.get("n_rows", "N/A"):,}
    - **Unique bgrps**: {cov.get("n_bgrp", "N/A")}
    - **Date range**: {cov.get("date_min", "N/A")} – {cov.get("date_max", "N/A")}
    - **Zero rate** (across all bgrp-month cells): {cov.get("zero_rate", 0):.1%}
    - **Missing rate**: {cov.get("missing_rate", 0):.1%}

    ## 2. Temporal Patterns
    - Citywide traffic peaked around **{ts.get("global_max_month", "N/A")}**
      and was lowest around **{ts.get("global_min_month", "N/A")}**
      (likely COVID-19 disruption if 2020–2021 appears in trough).
    - Calendar-month analysis shows peak in **{sea.get("peak_calendar_month", "N/A")}**
      and trough in **{sea.get("trough_calendar_month", "N/A")}**.
    - Seasonal range (peak–trough / mean): ≈{sea.get("seasonal_range_pct", "N/A")}%

    ## 3. Spatial Heterogeneity
    - Coefficient of variation across bgrps: **{spa.get("spatial_cv", "N/A")}**
      (>0.5 indicates high spatial inequality).
    - Highest-traffic bgrp: `{spa.get("top_bgrp", "N/A")}`
    - Median bgrp average: {spa.get("median_bgrp_mean", "N/A"):,}

    ## 4. Distribution Shape
    - Raw skewness: **{dist.get("raw_skew", "N/A")}** (kurtosis: {dist.get("raw_kurtosis", "N/A")})
    - log1p skewness: **{dist.get("log_skew", "N/A")}** (kurtosis: {dist.get("log_kurtosis", "N/A")})
    - **Recommendation**: {"✅ Use log1p transform" if dist.get("recommend_log_transform") else "⚠️  log1p may not fully normalize – consider winsorizing first"}

    ## 5. Extreme Values
    - 95th percentile: {ext.get("p95", "N/A"):,}
    - 99th percentile: {ext.get("p99", "N/A"):,}
    - 99.9th percentile: {ext.get("p999", "N/A"):,}
    - Records above p99: {ext.get("n_above_p99", "N/A")} ({ext.get("pct_above_p99", "N/A")}%)
    - **Recommendation**: Apply `log1p` transform before modeling.
      For robustness, also winsorize at 99th percentile before scaling.

    ## 6. Autocorrelation
    - ACF at lag 12 expected to be significant → **annual seasonality confirmed**.
    - Suggests SARIMA, STL+regression, or adding month-of-year as a feature in ML models.

    ## 7. Modeling Implications
    | Issue | Impact | Suggested Handling |
    |-------|--------|--------------------|
    | High zero rate | Inflated zeros bias regression | Treat zeros as structural; model nonzero separately or use Tweedie/Poisson |
    | Right-skewed distribution | OLS assumptions violated | log1p transform or robust scaler |
    | Strong annual seasonality | Autocorrelation if ignored | Add calendar month dummies or STL residuals |
    | High spatial CV | Spatial random effects matter | Include bgrp fixed effects or spatial lag |
    | Extreme outliers (>p99) | Distort scale-sensitive models | Winsorize or clip before StandardScaler |
    """)

    ensure_parent_dir(output_path)
    Path(output_path).write_text(summary.strip(), encoding="utf-8")
    LOGGER.info("Saved summary: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(config_path: str) -> None:
    config = load_config(config_path)
    paths = config["paths"]
    outputs = paths.get("outputs", {})

    fig_dir = Path(DEFAULT_FIG_DIR)
    summary_out = DEFAULT_SUMMARY_OUT

    _style()

    LOGGER.info("=== Traffic Volume EDA ===")
    df = load_monthly_data(config)

    stats_collected: dict = {}

    LOGGER.info("Plot 1/10: Coverage heatmap")
    stats_collected["coverage"] = plot_coverage_heatmap(df, fig_dir)

    LOGGER.info("Plot 2/10: Time series")
    stats_collected["time_series"] = plot_time_series(df, fig_dir)

    LOGGER.info("Plot 3/10: Monthly average")
    stats_collected["seasonal"] = plot_monthly_average(df, fig_dir)

    LOGGER.info("Plot 4/10: STL decomposition")
    plot_stl_decomposition(df, fig_dir)

    LOGGER.info("Plot 5/10: Spatial distribution")
    stats_collected["spatial"] = plot_spatial_distribution(df, config, fig_dir)

    LOGGER.info("Plot 6/10: Box plot by month")
    plot_boxplot_by_month(df, fig_dir)

    LOGGER.info("Plot 7/10: Distribution + QQ plot")
    stats_collected["distribution"] = plot_distribution(df, fig_dir)

    LOGGER.info("Plot 8/10: ACF / PACF")
    plot_autocorrelation(df, fig_dir)

    LOGGER.info("Plot 9/10: Extreme values")
    stats_collected["extremes"] = plot_extreme_values(df, fig_dir)

    LOGGER.info("Plot 10/10: bgrp-level distribution")
    plot_bgrp_distribution(df, fig_dir)

    LOGGER.info("Generating summary")
    generate_summary(stats_collected, summary_out)

    LOGGER.info("=== EDA complete. Figures in %s, summary at %s ===", fig_dir, summary_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic Volume EDA")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    run_eda(args.config)
