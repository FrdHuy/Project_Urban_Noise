from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .common import TARGET_EPSG, ensure_parent_dir, find_column, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def _normalize_block_id(df: pd.DataFrame, borough_col: str, block_col: str) -> pd.Series:
    borough = df[borough_col].astype(str).str.strip()
    block = pd.to_numeric(df[block_col], errors="coerce").fillna(-1).astype(int).astype(str)
    return borough + "_" + block


def _summarize_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            stats = df[col].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
            LOGGER.info("Summary stats for %s: %s", col, {k: round(v, 4) if pd.notna(v) else v for k, v in stats.items()})


def compute_block_features(config_path: str) -> None:
    cfg = load_config(config_path)["paths"]

    blocks_path = cfg["outputs"]["blocks_geojson"]
    features_geojson = cfg["outputs"]["features_geojson"]
    features_csv = cfg["outputs"]["features_csv"]
    mappluto_gdb = cfg["mappluto_gdb"]
    mappluto_layer = cfg["mappluto_layer"]

    LOGGER.info("Loading blocks from %s", blocks_path)
    blocks = gpd.read_file(blocks_path)
    if blocks.crs is None:
        raise ValueError("Blocks layer has no CRS; expected EPSG:2263")
    if int(blocks.crs.to_epsg() or -1) != TARGET_EPSG:
        LOGGER.warning("Blocks CRS is %s; reprojecting to EPSG:%s", blocks.crs, TARGET_EPSG)
        blocks = blocks.to_crs(epsg=TARGET_EPSG)

    LOGGER.info("Loaded %s blocks", len(blocks))

    wanted = ["Borough", "Block", "BldgArea", "ResArea", "NumFloors", "BuiltFAR", "BBL", "geometry"]
    LOGGER.info("Loading MapPLUTO parcels from %s (layer=%s)", mappluto_gdb, mappluto_layer)
    parcels = gpd.read_file(mappluto_gdb, layer=mappluto_layer)

    available = [c for c in wanted if c in parcels.columns]
    missing = sorted(set(wanted) - set(available))
    if missing:
        LOGGER.warning("MapPLUTO missing requested columns: %s", missing)

    borough_col = find_column(parcels.columns, ["Borough", "boro", "boro_code", "BoroCode"])
    block_col = find_column(parcels.columns, ["Block", "block", "BlockNum"])
    if borough_col is None or block_col is None:
        raise ValueError("MapPLUTO must include Borough and Block columns")

    keep_cols = sorted(set(available + [borough_col, block_col]))
    parcels = parcels[keep_cols].copy()
    parcels["block_id"] = _normalize_block_id(parcels, borough_col=borough_col, block_col=block_col)

    parcels["BldgArea"] = pd.to_numeric(parcels.get("BldgArea"), errors="coerce").fillna(0.0)
    parcels["ResArea"] = pd.to_numeric(parcels.get("ResArea"), errors="coerce").fillna(0.0)
    parcels["NumFloors"] = pd.to_numeric(parcels.get("NumFloors"), errors="coerce")
    parcels["BuiltFAR"] = pd.to_numeric(parcels.get("BuiltFAR"), errors="coerce")

    parcels["NumFloors_valid"] = parcels["NumFloors"].where(parcels["NumFloors"] > 0)

    block_agg = (
        parcels.groupby("block_id", as_index=False)
        .agg(
            bldgarea_sum=("BldgArea", "sum"),
            resarea_sum=("ResArea", "sum"),
            numfloors_mean=("NumFloors_valid", "mean"),
            numfloors_std=("NumFloors_valid", "std"),
            builtfar_mean=("BuiltFAR", "mean"),
            parcel_count=("block_id", "size"),
        )
    )

    block_agg["numfloors_mean"] = block_agg["numfloors_mean"].fillna(0.0)
    block_agg["numfloors_std"] = block_agg["numfloors_std"].fillna(0.0)

    out = blocks.merge(block_agg, on="block_id", how="left")

    fill_zero_cols = [
        "bldgarea_sum",
        "resarea_sum",
        "numfloors_mean",
        "numfloors_std",
        "builtfar_mean",
        "parcel_count",
    ]
    for col in fill_zero_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    floors_denom = out["numfloors_mean"].clip(lower=1.0)
    out["footprint_area_sum"] = out["bldgarea_sum"] / floors_denom
    out["gross_floor_area_sum"] = out["bldgarea_sum"]

    out["FAR"] = np.where(out["block_area_ft2"] > 0, out["gross_floor_area_sum"] / out["block_area_ft2"], 0.0)
    out["building_density"] = np.where(out["block_area_ft2"] > 0, out["footprint_area_sum"] / out["block_area_ft2"], 0.0)
    out["height_mean"] = out["numfloors_mean"] * 10.0
    out["height_std"] = out["numfloors_std"] * 10.0
    out["sky_openness_proxy"] = (1.0 - out["building_density"]).clip(lower=0.0, upper=1.0)

    joined_blocks = out["bldgarea_sum"].gt(0).sum()
    join_hit_rate = joined_blocks / len(out) if len(out) else 0.0
    LOGGER.info("Join hit rate (blocks with parcel data): %s/%s (%.2f%%)", joined_blocks, len(out), join_hit_rate * 100)

    _summarize_numeric(
        out,
        [
            "FAR",
            "building_density",
            "height_mean",
            "height_std",
            "sky_openness_proxy",
            "parcel_count",
        ],
    )

    ensure_parent_dir(features_geojson)
    ensure_parent_dir(features_csv)
    LOGGER.info("Writing block features to %s and %s", features_geojson, features_csv)
    out.to_file(features_geojson, driver="GeoJSON")
    out.drop(columns="geometry").to_csv(features_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute block features from blocks + MapPLUTO only")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    compute_block_features(args.config)
