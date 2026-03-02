from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .common import TARGET_EPSG, ensure_parent_dir, find_column, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


DEFAULT_TRAFFIC_PATH = "data/raw/Centerline_20260209.csv"
DEFAULT_ACTIVITY_PATH = "data/raw/Active_Cabaret_and_Catering_Licenses.csv"
DEFAULT_PERMITS_PATH = "data/raw/BUILDING_20260209.csv"
DEFAULT_OUTPUT = "data/processed/exposure_weekly.csv"


def _to_monday_week(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    monday = dt - pd.to_timedelta(dt.dt.weekday, unit="D")
    return monday.dt.strftime("%Y-%m-%d")


def _load_blocks_with_bgrp(blocks_path: str) -> gpd.GeoDataFrame:
    blocks = gpd.read_file(blocks_path)
    if blocks.crs is None:
        raise ValueError("blocks.geojson has no CRS")
    if int(blocks.crs.to_epsg() or -1) != TARGET_EPSG:
        LOGGER.warning("Blocks CRS %s != EPSG:%s, reprojecting", blocks.crs, TARGET_EPSG)
        blocks = blocks.to_crs(epsg=TARGET_EPSG)

    bgrp_col = find_column(blocks.columns, ["bgrp_id", "bgrp", "borocb2020", "geoid", "bg_id"])
    if bgrp_col is None:
        LOGGER.warning("No block-group column detected in blocks; falling back to block_id as bgrp_id")
        blocks["bgrp_id"] = blocks["block_id"].astype(str)
    else:
        blocks["bgrp_id"] = blocks[bgrp_col].astype(str)
    return blocks


def _week_index(traffic: pd.DataFrame, activity: pd.DataFrame, permits: pd.DataFrame) -> list[str]:
    weeks = pd.Series(dtype="object")
    for df in [traffic, activity, permits]:
        if "week_start" in df.columns:
            weeks = pd.concat([weeks, df["week_start"].dropna().astype(str)], ignore_index=True)
    weeks = weeks.drop_duplicates().sort_values()
    if len(weeks) == 0:
        monday = pd.Timestamp.today().normalize() - pd.to_timedelta(pd.Timestamp.today().weekday(), unit="D")
        LOGGER.warning("No valid week information found; using current Monday only: %s", monday.date())
        return [monday.strftime("%Y-%m-%d")]
    return weeks.tolist()


def _load_traffic(path: str, bgrps: gpd.GeoDataFrame) -> pd.DataFrame:
    if not Path(path).exists():
        LOGGER.warning("Traffic source missing (%s); using zeros", path)
        return pd.DataFrame(columns=["bgrp_id", "week_start", "traffic"])
    df = pd.read_csv(path, low_memory=False)

    date_col = find_column(df.columns, ["date", "timestamp", "sample_date", "obs_date"])
    if date_col:
        df["week_start"] = _to_monday_week(df[date_col])
    else:
        LOGGER.warning("Traffic date column unavailable; assigning current Monday as fallback")
        monday = pd.Timestamp.today().normalize() - pd.to_timedelta(pd.Timestamp.today().weekday(), unit="D")
        df["week_start"] = monday.strftime("%Y-%m-%d")

    value_col = find_column(df.columns, ["traffic", "volume", "count", "aadt", "segment_count"])
    if value_col is None:
        LOGGER.warning("Traffic value column unavailable; using row count proxy")
        df["traffic_value"] = 1.0
    else:
        df["traffic_value"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    bgrp_col = find_column(df.columns, ["bgrp_id", "bgrp", "geoid", "borocb2020"])
    if bgrp_col:
        out = df.rename(columns={bgrp_col: "bgrp_id"})[["bgrp_id", "week_start", "traffic_value"]]
        out = out.groupby(["bgrp_id", "week_start"], as_index=False).agg(traffic=("traffic_value", "sum"))
        return out

    block_col = find_column(df.columns, ["block_id", "block"])
    if block_col and "block_id" in bgrps.columns:
        LOGGER.warning("Traffic lacks bgrp_id; using block_id->bgrp_id fallback")
        block_to_bgrp = bgrps[["block_id", "bgrp_id"]].drop_duplicates()
        tmp = df.rename(columns={block_col: "block_id"}).merge(block_to_bgrp, on="block_id", how="left")
        out = tmp.groupby(["bgrp_id", "week_start"], as_index=False).agg(traffic=("traffic_value", "sum"))
        return out

    LOGGER.warning("Traffic has no geometry-ready keys; using uniform fallback proxy by week")
    wk = df.groupby("week_start", as_index=False).agg(total=("traffic_value", "sum"))
    if len(bgrps) == 0:
        return pd.DataFrame(columns=["bgrp_id", "week_start", "traffic"])
    wk["traffic"] = wk["total"] / len(bgrps)
    expanded = bgrps[["bgrp_id"]].drop_duplicates().assign(key=1).merge(wk.assign(key=1), on="key", how="outer").drop(columns=["key", "total"])
    return expanded[["bgrp_id", "week_start", "traffic"]]


def _load_activity(path: str, bgrps: gpd.GeoDataFrame, weeks: list[str]) -> pd.DataFrame:
    if not Path(path).exists():
        LOGGER.warning("Street activity source missing (%s); using zeros", path)
        return pd.DataFrame(columns=["bgrp_id", "week_start", "street_activity"])
    df = pd.read_csv(path, low_memory=False)

    val_col = find_column(df.columns, ["count", "licenses", "active", "license_count"])
    if val_col is None:
        df["street_activity_value"] = 1.0
    else:
        df["street_activity_value"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)

    bgrp_col = find_column(df.columns, ["bgrp_id", "bgrp", "geoid", "borocb2020"])
    if bgrp_col:
        static = df.rename(columns={bgrp_col: "bgrp_id"}).groupby("bgrp_id", as_index=False).agg(street_activity=("street_activity_value", "sum"))
    else:
        LOGGER.warning("Street activity has no bgrp key; using uniform density fallback")
        avg = df["street_activity_value"].sum() / max(len(bgrps), 1)
        static = bgrps[["bgrp_id"]].drop_duplicates().copy()
        static["street_activity"] = avg

    # static to weekly replicate
    if not weeks:
        return pd.DataFrame(columns=["bgrp_id", "week_start", "street_activity"])
    out = static.assign(key=1).merge(pd.DataFrame({"week_start": weeks, "key": 1}), on="key", how="inner").drop(columns="key")
    return out


def _load_permits(path: str, bgrps: gpd.GeoDataFrame) -> pd.DataFrame:
    if not Path(path).exists():
        LOGGER.warning("DOB permits source missing (%s); using zeros", path)
        return pd.DataFrame(columns=["bgrp_id", "week_start", "dob_permits"])
    df = pd.read_csv(path, low_memory=False)

    date_col = find_column(df.columns, ["issuedate", "issue_date", "filing_date", "date", "jobstartdate"])
    if date_col:
        df["week_start"] = _to_monday_week(df[date_col])
    else:
        LOGGER.warning("DOB permits date column unavailable; using current Monday fallback")
        monday = pd.Timestamp.today().normalize() - pd.to_timedelta(pd.Timestamp.today().weekday(), unit="D")
        df["week_start"] = monday.strftime("%Y-%m-%d")

    bgrp_col = find_column(df.columns, ["bgrp_id", "bgrp", "geoid", "borocb2020"])
    if bgrp_col:
        tmp = df.rename(columns={bgrp_col: "bgrp_id"})
        return tmp.groupby(["bgrp_id", "week_start"], as_index=False).size().rename(columns={"size": "dob_permits"})

    borough_col = find_column(df.columns, ["borough", "boro"])
    block_col = find_column(df.columns, ["block", "blocknum"])
    if borough_col and block_col and "block_id" in bgrps.columns:
        LOGGER.warning("DOB permits lacks bgrp_id; using Borough+Block -> block_id -> bgrp_id fallback")
        tmp = df.copy()
        tmp["block_id"] = tmp[borough_col].astype(str).str.strip() + "_" + pd.to_numeric(tmp[block_col], errors="coerce").fillna(-1).astype(int).astype(str)
        map_df = bgrps[["block_id", "bgrp_id"]].drop_duplicates()
        tmp = tmp.merge(map_df, on="block_id", how="left")
        return tmp.groupby(["bgrp_id", "week_start"], as_index=False).size().rename(columns={"size": "dob_permits"})

    LOGGER.warning("DOB permits key alignment unavailable; distributing weekly totals uniformly")
    wk = df.groupby("week_start", as_index=False).size().rename(columns={"size": "total"})
    if len(bgrps) == 0:
        return pd.DataFrame(columns=["bgrp_id", "week_start", "dob_permits"])
    wk["dob_permits"] = wk["total"] / len(bgrps)
    expanded = bgrps[["bgrp_id"]].drop_duplicates().assign(key=1).merge(wk.assign(key=1), on="key", how="outer").drop(columns=["key", "total"])
    return expanded[["bgrp_id", "week_start", "dob_permits"]]


def compute_dynamic_features(config_path: str) -> None:
    cfg = load_config(config_path).get("paths", {})
    outputs = cfg.get("outputs", {})

    blocks_path = outputs.get("blocks_geojson", "data/processed/blocks.geojson")
    exposure_out = outputs.get("exposure_weekly_csv", DEFAULT_OUTPUT)

    traffic_path = cfg.get("centerline_csv", DEFAULT_TRAFFIC_PATH)
    activity_path = cfg.get("activity_csv", DEFAULT_ACTIVITY_PATH)
    permits_path = cfg.get("building_csv", DEFAULT_PERMITS_PATH)

    bgrps = _load_blocks_with_bgrp(blocks_path)
    bgrps = bgrps[[c for c in ["block_id", "bgrp_id", "geometry"] if c in bgrps.columns]]

    traffic = _load_traffic(traffic_path, bgrps)
    permits = _load_permits(permits_path, bgrps)
    activity = _load_activity(activity_path, bgrps, _week_index(traffic, pd.DataFrame(), permits))

    weeks = _week_index(traffic, activity, permits)
    grid = bgrps[["bgrp_id"]].drop_duplicates().assign(key=1).merge(
        pd.DataFrame({"week_start": weeks, "key": 1}), on="key", how="inner"
    ).drop(columns="key")

    out = grid.merge(traffic, on=["bgrp_id", "week_start"], how="left")
    out = out.merge(activity, on=["bgrp_id", "week_start"], how="left")
    out = out.merge(permits, on=["bgrp_id", "week_start"], how="left")

    for col in ["traffic", "street_activity", "dob_permits"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)

    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out = out.dropna(subset=["week_start", "bgrp_id"])
    out["week_start"] = out["week_start"].dt.strftime("%Y-%m-%d")

    out = out.sort_values(["bgrp_id", "week_start"]).drop_duplicates(["bgrp_id", "week_start"], keep="first")

    dupes = out.duplicated(["bgrp_id", "week_start"]).sum()
    if dupes > 0:
        raise ValueError(f"Duplicate keys found in exposure_weekly: {dupes}")

    LOGGER.info(
        "Exposure weekly summary: bgrps=%s weeks=%s traffic[min,max]=[%.3f,%.3f] street_activity[min,max]=[%.3f,%.3f] dob_permits[min,max]=[%.3f,%.3f]",
        out["bgrp_id"].nunique(),
        out["week_start"].nunique(),
        out["traffic"].min(),
        out["traffic"].max(),
        out["street_activity"].min(),
        out["street_activity"].max(),
        out["dob_permits"].min(),
        out["dob_permits"].max(),
    )

    ensure_parent_dir(exposure_out)
    out.to_csv(exposure_out, index=False)
    LOGGER.info("Wrote %s rows to %s", len(out), exposure_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dynamic exposure features at bgrp x week")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    compute_dynamic_features(args.config)
