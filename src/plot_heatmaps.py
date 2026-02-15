from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import matplotlib.pyplot as plt

from .common import ensure_parent_dir, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def _plot_metric(gdf: gpd.GeoDataFrame, metric: str, title: str, output: str) -> None:
    ensure_parent_dir(output)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(
        column=metric,
        cmap="viridis",
        linewidth=0.05,
        edgecolor="white",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "Missing"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved heatmap: %s", output)


def plot_heatmaps(config_path: str) -> None:
    cfg = load_config(config_path)["paths"]
    features_path = cfg["outputs"]["features_geojson"]
    gdf = gpd.read_file(features_path)

    _plot_metric(gdf, "building_density", "NYC Block Building Density", cfg["outputs"]["fig_density"])
    _plot_metric(gdf, "height_std", "NYC Block Height Heterogeneity (Std)", cfg["outputs"]["fig_height_std"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create heatmaps from block-level features")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    plot_heatmaps(args.config)
