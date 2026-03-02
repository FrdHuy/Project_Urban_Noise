from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import pandas as pd

from .common import TARGET_EPSG, ensure_parent_dir, find_column, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def build_blocks(config_path: str) -> None:
    config = load_config(config_path)
    paths = config["paths"]

    gdb_path = paths["mappluto_gdb"]
    layer = paths["mappluto_layer"]
    output = paths["outputs"]["blocks_geojson"]

    LOGGER.info("Reading MapPLUTO layer '%s' from %s", layer, gdb_path)
    gdf = gpd.read_file(gdb_path, layer=layer)

    borough_col = find_column(gdf.columns, ["Borough", "boro", "boro_code", "BoroCode"])
    block_col = find_column(gdf.columns, ["Block", "block", "BlockNum"])

    if borough_col is None or block_col is None:
        raise ValueError("Could not detect Borough/Block columns in MapPLUTO data")

    gdf = gdf.rename(columns={borough_col: "Borough", block_col: "Block"})
    if gdf.crs is None:
        raise ValueError("MapPLUTO input has no CRS; expected EPSG:2263")
    if int(gdf.crs.to_epsg() or -1) != TARGET_EPSG:
        LOGGER.warning("MapPLUTO CRS is %s; reprojecting to EPSG:%s", gdf.crs, TARGET_EPSG)
        gdf = gdf.to_crs(epsg=TARGET_EPSG)

    gdf["Borough"] = gdf["Borough"].astype(str).str.strip()
    gdf["Block"] = pd.to_numeric(gdf["Block"], errors="coerce")
    gdf = gdf.dropna(subset=["Block", "geometry"])
    gdf["Block"] = gdf["Block"].astype(int).astype(str)

    LOGGER.info("Dissolving lots to block geometry")
    blocks = gdf[["Borough", "Block", "geometry"]].dissolve(by=["Borough", "Block"], as_index=False)
    blocks["block_id"] = blocks["Borough"].astype(str) + "_" + blocks["Block"].astype(str)
    blocks["block_area_ft2"] = blocks.geometry.area

    ensure_parent_dir(output)
    LOGGER.info("Writing %s blocks to %s", len(blocks), output)
    blocks.to_file(output, driver="GeoJSON")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dissolved block geometry from MapPLUTO")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    build_blocks(args.config)
