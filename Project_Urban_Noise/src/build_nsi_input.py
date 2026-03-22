from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .common import TARGET_EPSG, ensure_parent_dir, find_column, load_config, setup_logging

LOGGER = logging.getLogger(__name__)

DEFAULT_EXPOSURE = "data/processed/exposure_weekly.csv"
DEFAULT_OUTPUT = "data/processed/nsi_input_weekly.csv"
DEFAULT_MAPPLUTO_GDB = "data/raw/MapPLUTO25v4_unclipped.gdb"
DEFAULT_MAPPLUTO_LAYER = "MapPLUTO_25v4_unclipped"
DEFAULT_Y_SOURCE = "data/raw/noise_complaints.csv"


def _to_monday_week(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    monday = dt - pd.to_timedelta(dt.dt.weekday, unit="D")
    return monday.dt.strftime("%Y-%m-%d")


def _load_bgrp_map(blocks_path: str) -> pd.DataFrame:
    blocks = gpd.read_file(blocks_path)
    if blocks.crs is None:
        raise ValueError("blocks.geojson has no CRS")
    if int(blocks.crs.to_epsg() or -1) != TARGET_EPSG:
        blocks = blocks.to_crs(epsg=TARGET_EPSG)

    bgrp_col = find_column(blocks.columns, ["bgrp_id", "bgrp", "borocb2020", "geoid", "bg_id"])
    if bgrp_col is None:
        LOGGER.warning("No bgrp field detected in blocks; falling back to block_id as bgrp_id")
        blocks["bgrp_id"] = blocks["block_id"].astype(str)
    else:
        blocks["bgrp_id"] = blocks[bgrp_col].astype(str)

    return blocks[[c for c in ["block_id", "bgrp_id"] if c in blocks.columns]].drop_duplicates()


def _compute_households(mappluto_gdb: str, mappluto_layer: str, bgrp_map: pd.DataFrame) -> pd.DataFrame:
    if not Path(mappluto_gdb).exists():
        LOGGER.warning("MapPLUTO GDB missing for household estimation; defaulting to 0")
        return bgrp_map[["bgrp_id"]].drop_duplicates().assign(households=0.0)

    gdf = gpd.read_file(mappluto_gdb, layer=mappluto_layer)
    borough_col = find_column(gdf.columns, ["Borough", "boro", "boro_code", "BoroCode"])
    block_col = find_column(gdf.columns, ["Block", "block", "BlockNum"])
    if borough_col is None or block_col is None:
        LOGGER.warning("MapPLUTO Borough/Block missing; households fallback to 0")
        return bgrp_map[["bgrp_id"]].drop_duplicates().assign(households=0.0)

    gdf["block_id"] = gdf[borough_col].astype(str).str.strip() + "_" + pd.to_numeric(
        gdf[block_col], errors="coerce"
    ).fillna(-1).astype(int).astype(str)

    hh_col = find_column(gdf.columns, ["unitsres", "unitstotal", "resunits", "households"])
    if hh_col is None:
        LOGGER.warning("No household-like column in MapPLUTO; households fallback to 0")
        return bgrp_map[["bgrp_id"]].drop_duplicates().assign(households=0.0)

    gdf["households_raw"] = pd.to_numeric(gdf[hh_col], errors="coerce").fillna(0.0)
    block_hh = gdf.groupby("block_id", as_index=False).agg(households=("households_raw", "sum"))
    out = bgrp_map.merge(block_hh, on="block_id", how="left")
    out["households"] = out["households"].fillna(0.0)
    return out.groupby("bgrp_id", as_index=False).agg(households=("households", "sum"))


def _compute_y_total(y_source_path: str, bgrp_map: pd.DataFrame) -> pd.DataFrame:
    if not Path(y_source_path).exists():
        LOGGER.warning("y_total source missing (%s); using zeros", y_source_path)
        return pd.DataFrame(columns=["bgrp_id", "week_start", "y_total"])

    df = pd.read_csv(y_source_path, low_memory=False)
    date_col = find_column(df.columns, ["created_date", "date", "timestamp", "complaint_date"])
    if date_col is None:
        LOGGER.warning("y_total date column unavailable; cannot build weekly counts, using zeros")
        return pd.DataFrame(columns=["bgrp_id", "week_start", "y_total"])

    df["week_start"] = _to_monday_week(df[date_col])
    bgrp_col = find_column(df.columns, ["bgrp_id", "bgrp", "geoid", "borocb2020"])
    if bgrp_col:
        out = df.rename(columns={bgrp_col: "bgrp_id"}).groupby(["bgrp_id", "week_start"], as_index=False).size()
        return out.rename(columns={"size": "y_total"})

    borough_col = find_column(df.columns, ["borough", "boro"])
    block_col = find_column(df.columns, ["block", "blocknum"])
    if borough_col and block_col and "block_id" in bgrp_map.columns:
        LOGGER.warning("y_total lacks bgrp_id; using Borough+Block -> block_id -> bgrp fallback")
        df["block_id"] = df[borough_col].astype(str).str.strip() + "_" + pd.to_numeric(df[block_col], errors="coerce").fillna(-1).astype(int).astype(str)
        tmp = df.merge(bgrp_map, on="block_id", how="left")
        out = tmp.groupby(["bgrp_id", "week_start"], as_index=False).size()
        return out.rename(columns={"size": "y_total"})

    LOGGER.warning("No key available for y_total alignment; returning empty (zeros later)")
    return pd.DataFrame(columns=["bgrp_id", "week_start", "y_total"])


def build_nsi_input(config_path: str) -> None:
    cfg = load_config(config_path).get("paths", {})
    outputs = cfg.get("outputs", {})

    blocks_path = outputs.get("blocks_geojson", "data/processed/blocks.geojson")
    exposure_path = outputs.get("exposure_weekly_csv", DEFAULT_EXPOSURE)
    nsi_out = outputs.get("nsi_input_weekly_csv", DEFAULT_OUTPUT)
    mappluto_gdb = cfg.get("mappluto_gdb", DEFAULT_MAPPLUTO_GDB)
    mappluto_layer = cfg.get("mappluto_layer", DEFAULT_MAPPLUTO_LAYER)
    y_source_path = cfg.get("y_total_csv", DEFAULT_Y_SOURCE)

    if not Path(exposure_path).exists():
        raise FileNotFoundError(
            f"Exposure weekly file not found at {exposure_path}. Run python -m src.compute_dynamic_features --config config.yaml first."
        )

    exposure = pd.read_csv(exposure_path, low_memory=False)
    exposure["week_start"] = pd.to_datetime(exposure["week_start"], errors="coerce")
    exposure = exposure.dropna(subset=["bgrp_id", "week_start"])
    exposure["week_start"] = exposure["week_start"].dt.strftime("%Y-%m-%d")

    bgrp_map = _load_bgrp_map(blocks_path)
    households = _compute_households(mappluto_gdb, mappluto_layer, bgrp_map)
    y_total = _compute_y_total(y_source_path, bgrp_map)

    out = exposure.merge(households, on="bgrp_id", how="left")
    out = out.merge(y_total, on=["bgrp_id", "week_start"], how="left")

    for col in ["y_total", "households", "traffic", "street_activity", "dob_permits"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)

    out = out[["bgrp_id", "week_start", "y_total", "households", "traffic", "street_activity", "dob_permits"]]
    out = out.sort_values(["bgrp_id", "week_start"]).drop_duplicates(["bgrp_id", "week_start"], keep="first")

    dupes = out.duplicated(["bgrp_id", "week_start"]).sum()
    if dupes:
        raise ValueError(f"Duplicate keys found in NSI input: {dupes}")

    LOGGER.info(
        "NSI weekly summary: bgrps=%s weeks=%s y_total[min,max]=[%.3f,%.3f] households[min,max]=[%.3f,%.3f]",
        out["bgrp_id"].nunique(),
        out["week_start"].nunique(),
        out["y_total"].min(),
        out["y_total"].max(),
        out["households"].min(),
        out["households"].max(),
    )

    ensure_parent_dir(nsi_out)
    out.to_csv(nsi_out, index=False)
    LOGGER.info("Wrote %s rows to %s", len(out), nsi_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NSI weekly model input table")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    build_nsi_input(args.config)
