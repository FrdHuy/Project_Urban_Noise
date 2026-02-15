from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .common import (
    TARGET_EPSG,
    ensure_parent_dir,
    find_column,
    load_config,
    normalize_key_columns,
    numeric_column,
    setup_logging,
)

LOGGER = logging.getLogger(__name__)


HEIGHT_CANDIDATES = ["height", "heightroof", "height_ft", "max_height", "mean_height", "z"]
FOOTPRINT_CANDIDATES = ["shape_area", "footprint_area", "area", "area_ft2", "geom_area"]
FLOOR_AREA_CANDIDATES = ["gross_floor_area", "gfa", "bldgarea", "floor_area", "resarea", "comarea"]


def _load_building_records(path: str) -> pd.DataFrame:
    LOGGER.info("Reading building data from %s", path)
    return pd.read_csv(path, low_memory=False)


def _attach_block_id(blocks: gpd.GeoDataFrame, df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_key_columns(df)

    borough_col = find_column(df.columns, ["Borough", "boro", "boro_code", "BoroCode"])
    block_col = find_column(df.columns, ["Block", "block", "BlockNum"])
    bbl_col = find_column(df.columns, ["BBL", "bbl"])

    merged = None
    if borough_col and block_col:
        LOGGER.info("Joining records by Borough + Block")
        keyed = df.rename(columns={borough_col: "Borough", block_col: "Block"}).copy()
        keyed["Borough"] = keyed["Borough"].astype(str).str.strip()
        keyed["Block"] = pd.to_numeric(keyed["Block"], errors="coerce").fillna(-1).astype(int).astype(str)
        keyed["block_id"] = keyed["Borough"] + "_" + keyed["Block"]
        merged = keyed
    elif bbl_col and "BBL" in blocks.columns:
        LOGGER.info("Joining records by BBL")
        keyed = df.rename(columns={bbl_col: "BBL"}).copy()
        keyed["BBL"] = keyed["BBL"].astype(str).str.replace(".0$", "", regex=True)
        block_keys = blocks[["block_id", "BBL"]].drop_duplicates()
        merged = keyed.merge(block_keys, on="BBL", how="left")
    else:
        LOGGER.warning("No usable key columns found for building records; attempting spatial fallback")
        merged = _spatial_fallback(blocks, df)

    if "block_id" not in merged.columns:
        merged["block_id"] = pd.NA
    return merged




def _spatial_fallback(blocks: gpd.GeoDataFrame, df: pd.DataFrame) -> pd.DataFrame:
    lon_col = find_column(df.columns, ["longitude", "lon", "x", "xcoord", "x_coord"])
    lat_col = find_column(df.columns, ["latitude", "lat", "y", "ycoord", "y_coord"])
    if lon_col is None or lat_col is None:
        LOGGER.warning("Spatial fallback unavailable (missing coordinate columns)")
        out = df.copy()
        out["block_id"] = pd.NA
        return out

    pts = df.copy()
    pts[lon_col] = pd.to_numeric(pts[lon_col], errors="coerce")
    pts[lat_col] = pd.to_numeric(pts[lat_col], errors="coerce")
    pts = pts.dropna(subset=[lon_col, lat_col])
    if pts.empty:
        LOGGER.warning("Spatial fallback found no valid coordinates")
        out = df.copy()
        out["block_id"] = pd.NA
        return out

    gpts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts[lon_col], pts[lat_col]), crs="EPSG:4326")
    gpts = gpts.to_crs(blocks.crs)
    joined = gpd.sjoin(gpts, blocks[["block_id", "geometry"]], how="left", predicate="within")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))

def _aggregate_block_metrics(blocks: gpd.GeoDataFrame, records: pd.DataFrame) -> pd.DataFrame:
    work = records.copy()
    work["height"] = numeric_column(work, HEIGHT_CANDIDATES, default=0.0)
    work["footprint_area"] = numeric_column(work, FOOTPRINT_CANDIDATES, default=0.0)
    work["floor_area"] = numeric_column(work, FLOOR_AREA_CANDIDATES, default=0.0)

    grouped = (
        work.dropna(subset=["block_id"])
        .groupby("block_id", as_index=False)
        .agg(
            footprint_area_sum=("footprint_area", "sum"),
            gross_floor_area_sum=("floor_area", "sum"),
            height_mean=("height", "mean"),
            height_std=("height", "std"),
            building_count=("height", "count"),
        )
    )
    grouped["height_std"] = grouped["height_std"].fillna(0.0)

    out = blocks.merge(grouped, on="block_id", how="left")
    for col in ["footprint_area_sum", "gross_floor_area_sum", "height_mean", "height_std", "building_count"]:
        out[col] = out[col].fillna(0.0)

    out["building_density"] = np.where(
        out["block_area_ft2"] > 0,
        out["footprint_area_sum"] / out["block_area_ft2"],
        0.0,
    )
    out["FAR"] = np.where(
        out["block_area_ft2"] > 0,
        out["gross_floor_area_sum"] / out["block_area_ft2"],
        0.0,
    )
    out["sky_openness_proxy"] = (1 - out["building_density"]).clip(lower=0.0)
    return out


def _compute_hw_ratio(blocks: gpd.GeoDataFrame, centerline_path: str) -> pd.DataFrame:
    if not Path(centerline_path).exists():
        LOGGER.warning("Centerline file not found at %s; setting H/W ratio to NaN", centerline_path)
        return pd.DataFrame({"block_id": blocks["block_id"], "street_hw_ratio": np.nan})

    cl = pd.read_csv(centerline_path, low_memory=False)
    borough_col = find_column(cl.columns, ["Borough", "boro", "boro_code"])
    block_col = find_column(cl.columns, ["Block", "block"])
    width_col = find_column(
        cl.columns,
        ["width", "street_width", "cartwaywidth", "roadbedwidth", "rw_width", "full_street_width"],
    )
    length_col = find_column(cl.columns, ["length", "shape_length", "segment_length", "len_ft"])

    if borough_col and block_col:
        cl = cl.rename(columns={borough_col: "Borough", block_col: "Block"})
        cl["Borough"] = cl["Borough"].astype(str).str.strip()
        cl["Block"] = pd.to_numeric(cl["Block"], errors="coerce").fillna(-1).astype(int).astype(str)
        cl["block_id"] = cl["Borough"] + "_" + cl["Block"]
    else:
        LOGGER.warning("Centerline lacks Borough/Block columns. Cannot compute per-block H/W reliably")
        return pd.DataFrame({"block_id": blocks["block_id"], "street_hw_ratio": np.nan})

    if width_col:
        cl["street_width"] = pd.to_numeric(cl[width_col], errors="coerce")
        method = f"centerline width column '{width_col}'"
    else:
        proxy_col = length_col
        if proxy_col is None:
            cl["street_width"] = np.nan
            method = "no width/length columns available"
        else:
            cl["street_width"] = pd.to_numeric(cl[proxy_col], errors="coerce") / 10.0
            method = f"proxy width from {proxy_col}/10"

    LOGGER.info("H/W method: %s", method)

    per_block_width = cl.groupby("block_id", as_index=False).agg(street_width_mean=("street_width", "mean"))
    hw = blocks[["block_id", "height_mean"]].merge(per_block_width, on="block_id", how="left")
    hw["street_hw_ratio"] = np.where(
        hw["street_width_mean"].fillna(0) > 0,
        hw["height_mean"] / hw["street_width_mean"],
        np.nan,
    )
    hw["street_hw_method"] = method
    return hw[["block_id", "street_hw_ratio", "street_hw_method"]]


def compute_block_features(config_path: str) -> None:
    cfg = load_config(config_path)["paths"]
    blocks_path = cfg["outputs"]["blocks_geojson"]
    features_geojson = cfg["outputs"]["features_geojson"]
    features_csv = cfg["outputs"]["features_csv"]

    blocks = gpd.read_file(blocks_path)
    if blocks.crs is None:
        raise ValueError("Blocks file has no CRS")
    if int(blocks.crs.to_epsg() or -1) != TARGET_EPSG:
        LOGGER.warning("Blocks CRS %s != EPSG:%s, reprojecting", blocks.crs, TARGET_EPSG)
        blocks = blocks.to_crs(epsg=TARGET_EPSG)

    records = []
    for path in [cfg["building_csv"], cfg["bldg_3d_metrics_csv"]]:
        if Path(path).exists():
            records.append(_attach_block_id(blocks, _load_building_records(path)))
        else:
            LOGGER.warning("Building source missing: %s", path)

    all_records = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["block_id"])
    result = _aggregate_block_metrics(blocks, all_records)

    hw = _compute_hw_ratio(result, cfg["centerline_csv"])
    result = result.merge(hw, on="block_id", how="left")

    ensure_parent_dir(features_geojson)
    ensure_parent_dir(features_csv)
    LOGGER.info("Writing feature outputs to %s and %s", features_geojson, features_csv)
    result.to_file(features_geojson, driver="GeoJSON")
    result.drop(columns="geometry").to_csv(features_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute block-level urban form features")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    compute_block_features(args.config)
