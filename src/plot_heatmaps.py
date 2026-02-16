from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from .common import ensure_parent_dir, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def _plot_metric_quantiles_with_fallback(
    gdf: gpd.GeoDataFrame,
    metric: str,
    legend_title: str,
    title: str,
    output: str,
) -> None:
    ensure_parent_dir(output)

    metric_values = pd.to_numeric(gdf[metric], errors="coerce")
    vmax = metric_values.quantile(0.99)
    if pd.isna(vmax) or vmax <= 0:
        vmax = metric_values.max()
    if pd.isna(vmax) or vmax <= 0:
        vmax = 1.0

    clipped_col = f"{metric}_clipped"
    plot_gdf = gdf.copy()
    plot_gdf[clipped_col] = metric_values.clip(lower=0, upper=vmax)

    fig, ax = plt.subplots(figsize=(11, 11))
    used_quantiles = False

    try:
        try:
            import mapclassify  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Quantile heatmap plotting requires 'mapclassify'. Install it via requirements.txt."
            ) from exc

        plot_gdf.plot(
            column=clipped_col,
            cmap="YlOrRd",
            scheme="quantiles",
            k=11,
            linewidth=0,
            edgecolor="none",
            legend=True,
            legend_kwds={"title": legend_title},
            ax=ax,
            missing_kwds={"color": "lightgrey"},
        )
        used_quantiles = True
    except Exception as exc:  # fallback requested
        LOGGER.warning(
            "Quantile plotting failed for '%s' (%s). Falling back to continuous scale [0, %.4f].",
            metric,
            exc,
            vmax,
        )
        plot_gdf.plot(
            column=clipped_col,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            linewidth=0,
            edgecolor="none",
            legend=True,
            legend_kwds={"title": legend_title},
            ax=ax,
            missing_kwds={"color": "lightgrey"},
        )

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    LOGGER.info(
        "Saved heatmap: %s (metric=%s, vmax=%.4f, mode=%s)",
        output,
        metric,
        vmax,
        "quantiles" if used_quantiles else "continuous_fallback",
    )


def plot_heatmaps(config_path: str) -> None:
    cfg = load_config(config_path)["paths"]
    features_path = cfg["outputs"]["features_geojson"]
    gdf = gpd.read_file(features_path)

    density_output = str(Path("figures") / "heatmap_building_density_quantiles_red.png")
    heterogeneity_output = str(Path("figures") / "heatmap_height_heterogeneity_quantiles_red.png")

    _plot_metric_quantiles_with_fallback(
        gdf=gdf,
        metric="building_density",
        legend_title="Building Density (quantiles)",
        title="NYC Block Building Density (Red = High, Q11)",
        output=density_output,
    )
    _plot_metric_quantiles_with_fallback(
        gdf=gdf,
        metric="height_std",
        legend_title="Height Heterogeneity (Std, quantiles)",
        title="NYC Block Height Heterogeneity (Red = High, Q11)",
        output=heterogeneity_output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-quality red quantile heatmaps")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    plot_heatmaps(args.config)
