from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import ensure_parent_dir, load_config, setup_logging


DEFAULT_TRAFFIC_SUMMARY = "data/processed/traffic_monthly_summary.csv"
DEFAULT_TRAFFIC_NONZERO = "data/processed/traffic_monthly_nonzero.csv"
DEFAULT_MONTHLY_DYNAMIC = "data/processed/monthly_dynamic_with_activity.csv"
DEFAULT_MONTHLY_QUALITY = "data/processed/monthly_dynamic_quality.csv"

DEFAULT_FIG_SUMMARY = "figures/monthly_traffic_summary.png"
DEFAULT_FIG_NONZERO = "figures/monthly_nonzero_traffic_distribution.png"
DEFAULT_FIG_DYNAMIC = "figures/monthly_dynamic_feature_trends.png"
DEFAULT_FIG_QUALITY = "figures/monthly_dynamic_quality_check.png"


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


def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, **kwargs)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df = df.dropna(subset=["month"]).sort_values("month")
    return df


def _plot_traffic_summary(path: str, output: str) -> None:
    df = _read_csv(path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    series = [
        ("traffic_volume_sum", "Monthly total traffic volume", "#c44e52"),
        ("traffic_obs_count", "Monthly traffic observation count", "#4c72b0"),
        ("bgrp_nonzero_traffic_volume", "BGRPs with nonzero traffic volume", "#55a868"),
    ]
    for ax, (col, title, color) in zip(axes, series):
        ax.plot(df["month"], pd.to_numeric(df[col], errors="coerce"), color=color, linewidth=1.8)
        ax.set_title(title)
        ax.set_ylabel(col)

    axes[-1].set_xlabel("Month")
    fig.suptitle("Monthly Traffic Summary Across NYC", fontsize=14, y=0.98)
    fig.autofmt_xdate()
    fig.tight_layout()

    ensure_parent_dir(output)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_nonzero_distribution(path: str, output: str) -> None:
    df = _read_csv(path)
    df["year"] = df["month"].dt.year

    yearly = (
        df.groupby("year", as_index=False)
        .agg(
            median_traffic=("traffic_volume_sum", "median"),
            p90_traffic=("traffic_volume_sum", lambda s: s.quantile(0.9)),
            active_bgrp_months=("bgrp_id", "size"),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].bar(yearly["year"], yearly["active_bgrp_months"], color="#8172b3", width=0.8)
    axes[0].set_title("Active BGRP-month observations with nonzero traffic")
    axes[0].set_ylabel("Count")

    axes[1].plot(yearly["year"], yearly["median_traffic"], color="#dd8452", linewidth=2, label="Median")
    axes[1].plot(yearly["year"], yearly["p90_traffic"], color="#c44e52", linewidth=2, label="90th percentile")
    axes[1].set_title("Distribution of nonzero traffic volume by year")
    axes[1].set_ylabel("Traffic volume")
    axes[1].set_xlabel("Year")
    axes[1].legend(frameon=False)

    fig.suptitle("Distribution of Nonzero Traffic Observations", fontsize=14, y=0.98)
    fig.tight_layout()

    ensure_parent_dir(output)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamic_trends(path: str, output: str) -> None:
    df = _read_csv(path, usecols=[
        "month",
        "traffic_volume_sum",
        "traffic_obs_count",
        "traffic_hist_daily_sum",
        "event_new_count",
        "event_active_count",
    ])
    monthly = df.groupby("month", as_index=False).sum(numeric_only=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    traffic_cols = [
        ("traffic_volume_sum", "Traffic volume sum", "#4c72b0"),
        ("traffic_obs_count", "Traffic observation count", "#55a868"),
        ("traffic_hist_daily_sum", "Historical traffic daily sum", "#c44e52"),
    ]
    for col, label, color in traffic_cols:
        axes[0].plot(monthly["month"], monthly[col], label=label, color=color, linewidth=1.8)
    axes[0].set_title("Traffic-related monthly dynamic features")
    axes[0].set_ylabel("Value")
    axes[0].legend(frameon=False, ncol=3)

    event_cols = [
        ("event_new_count", "New activity events", "#8172b3"),
        ("event_active_count", "Active activity events", "#937860"),
    ]
    for col, label, color in event_cols:
        axes[1].plot(monthly["month"], monthly[col], label=label, color=color, linewidth=1.8)
    axes[1].set_title("Activity-related monthly dynamic features")
    axes[1].set_ylabel("Value")
    axes[1].set_xlabel("Month")
    axes[1].legend(frameon=False)

    fig.suptitle("Temporal Trends in Monthly Dynamic Features", fontsize=14, y=0.98)
    fig.autofmt_xdate()
    fig.tight_layout()

    ensure_parent_dir(output)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_quality(path: str, output: str) -> None:
    df = pd.read_csv(path)
    df = df.sort_values("missing_rate", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes[0].barh(df["column"], df["missing_rate"], color="#c44e52")
    axes[0].set_title("Missing rate by feature")
    axes[0].set_xlabel("Missing rate")
    axes[0].set_xlim(0, 1.05)

    axes[1].barh(df["column"], df["zero_rate"], color="#4c72b0")
    axes[1].set_title("Zero rate by feature")
    axes[1].set_xlabel("Zero rate")
    axes[1].set_xlim(0, 1.05)

    fig.suptitle("Monthly Dynamic Data Quality Check", fontsize=14, y=0.98)
    fig.tight_layout()

    ensure_parent_dir(output)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def build_visuals(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    outputs = paths.get("outputs", {})

    traffic_summary = outputs.get("traffic_monthly_summary_csv", DEFAULT_TRAFFIC_SUMMARY)
    traffic_nonzero = outputs.get("traffic_monthly_nonzero_csv", DEFAULT_TRAFFIC_NONZERO)
    monthly_dynamic = outputs.get("monthly_dynamic_csv", DEFAULT_MONTHLY_DYNAMIC)
    monthly_quality = outputs.get("monthly_dynamic_quality_csv", DEFAULT_MONTHLY_QUALITY)

    _style()
    _plot_traffic_summary(traffic_summary, DEFAULT_FIG_SUMMARY)
    _plot_nonzero_distribution(traffic_nonzero, DEFAULT_FIG_NONZERO)
    _plot_dynamic_trends(monthly_dynamic, DEFAULT_FIG_DYNAMIC)
    _plot_quality(monthly_quality, DEFAULT_FIG_QUALITY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot monthly dynamic data visuals")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    build_visuals(args.config)
