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
        LOGGER.warning("No bgrp_id-like field in blocks; using block_id fallback")
        blocks["bgrp_id"] = blocks["block_id"].astype(str)
    else:
        blocks["bgrp_id"] = blocks[bgrp_col].astype(str)
    return blocks


def _collect_weeks(*frames: pd.DataFrame) -> list[str]:
    wk = pd.Series(dtype="object")
    for df in frames:
        if "week_start" in df.columns:
            wk = pd.concat([wk, df["week_start"].dropna().astype(str)], ignore_index=True)
    wk = wk.drop_duplicates().sort_values()
    if wk.empty:
        monday = pd.Timestamp.today().normalize() - pd.to_timedelta(pd.Timestamp.today().weekday(), unit="D")
        LOGGER.warning("No date fields detected in dynamic sources; using current Monday %s", monday.date())
        return [monday.strftime("%Y-%m-%d")]
    return wk.tolist()


def _parse_week(df: pd.DataFrame, candidates: list[str], source_name: str) -> pd.Series:
    date_col = find_column(df.columns, candidates)
    if date_col is None:
        LOGGER.warning("%s date field missing; fallback to current Monday", source_name)
        monday = pd.Timestamp.today().normalize() - pd.to_timedelta(pd.Timestamp.today().weekday(), unit="D")
        return pd.Series([monday.strftime("%Y-%m-%d")] * len(df), index=df.index)
    return _to_monday_week(df[date_col])


def _traffic_spatial_join(df: pd.DataFrame, bgrps: gpd.GeoDataFrame) -> pd.DataFrame | None:
    lon_col = find_column(df.columns, ["longitude", "lon", "x", "xcoord", "x_coord"])
    lat_col = find_column(df.columns, ["latitude", "lat", "y", "ycoord", "y_coord"])
    if lon_col is None or lat_col is None:
        return None

    work = df.copy()
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work = work.dropna(subset=[lon_col, lat_col])
    if work.empty:
        return None

    LOGGER.warning("Traffic bgrp key missing; applying spatial fallback from point coordinates")
    gpts = gpd.GeoDataFrame(work, geometry=gpd.points_from_xy(work[lon_col], work[lat_col]), crs="EPSG:4326").to_crs(
        bgrps.crs
    )
    joined = gpd.sjoin(gpts, bgrps[["bgrp_id", "geometry"]], how="left", predicate="within")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def _load_traffic(path: str, bgrps: gpd.GeoDataFrame) -> pd.DataFrame:
    if not Path(path).exists():
        LOGGER.warning("Traffic source missing (%s); using zeros", path)
        return pd.DataFrame(columns=["bgrp_id", "week_start", "traffic"])

    df = pd.read_csv(path, low_memory=False)
    df["week_start"] = _parse_week(df, ["date", "timestamp", "sample_date", "obs_date"], "Traffic")

    value_col = find_column(df.columns, ["traffic", "volume", "count", "aadt", "segment_count"])
    df["traffic_value"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0) if value_col else 1.0
    if value_col is None:
        LOGGER.warning("Traffic intensity column missing; using event-count proxy")

    bgrp_col = find_column(df.columns, ["bgrp_id", "bgrp", "geoid", "borocb2020"])
    if bgrp_col:
        out = df.rename(columns={bgrp_col: "bgrp_id"})
        return out.groupby(["bgrp_id", "week_start"], as_index=False).agg(traffic=("traffic_value", "sum"))

    block_col = find_column(df.columns, ["block_id", "block"])
    if block_col and "block_id" in bgrps.columns:
        LOGGER.warning("Traffic bgrp_id missing; using block_id -> bgrp_id fallback")
        map_df = bgrps[["block_id", "bgrp_id"]].drop_duplicates()
        tmp = df.rename(columns={block_col: "block_id"}).merge(map_df, on="block_id", how="left")
        return tmp.groupby(["bgrp_id", "week_start"], as_index=False).agg(traffic=("traffic_value", "sum"))

    spatial_tmp = _traffic_spatial_join(df, bgrps)
    if spatial_tmp is not None:
        return spatial_tmp.groupby(["bgrp_id", "week_start"], as_index=False).agg(traffic=("traffic_value", "sum"))

    LOGGER.warning("Traffic alignment failed; using uniform weekly proxy distribution")
    wk = df.groupby("week_start", as_index=False).agg(total=("traffic_value", "sum"))
    if bgrps.empty:
        return pd.DataFrame(columns=["bgrp_id", "week_start", "traffic"])
    wk["traffic"] = wk["total"] / len(bgrps)
    grid = bgrps[["bgrp_id"]].drop_duplicates().assign(key=1).merge(wk.assign(key=1), on="key", how="inner")
    return grid[["bgrp_id", "week_start", "traffic"]]


def _load_activity(path: str, bgrps: gpd.GeoDataFrame, weeks: list[str]) -> pd.DataFrame:
    if not Path(path).exists():
        LOGGER.warning("Street activity source missing (%s); using zeros", path)
        return pd.DataFrame(columns=["bgrp_id", "week_start", "street_activity"])

    df = pd.read_csv(path, low_memory=False)
    val_col = find_column(df.columns, ["count", "licenses", "active", "license_count"])
    df["street_activity_value"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0) if val_col else 1.0

    bgrp_col = find_column(df.columns, ["bgrp_id", "bgrp", "geoid", "borocb2020"])
    if bgrp_col:
        static = df.rename(columns={bgrp_col: "bgrp_id"}).groupby("bgrp_id", as_index=False).agg(
            street_activity=("street_activity_value", "sum")
        )
    else:
        LOGGER.warning("Street activity static source has no bgrp_id; using uniform density fallback")
        static = bgrps[["bgrp_id"]].drop_duplicates().copy()
        static["street_activity"] = df["street_activity_value"].sum() / max(len(static), 1)

    week_df = pd.DataFrame({"week_start": weeks})
    return static.assign(key=1).merge(week_df.assign(key=1), on="key", how="inner").drop(columns="key")


def _load_permits(path: str, bgrps: gpd.GeoDataFrame) -> pd.DataFrame:
    if not Path(path).exists():
        LOGGER.warning("DOB permits source missing (%s); using zeros", path)
        return pd.DataFrame(columns=["bgrp_id", "week_start", "dob_permits"])

    df = pd.read_csv(path, low_memory=False)
    df["week_start"] = _parse_week(df, ["issuedate", "issue_date", "filing_date", "date", "jobstartdate"], "DOB permits")

    bgrp_col = find_column(df.columns, ["bgrp_id", "bgrp", "geoid", "borocb2020"])
    if bgrp_col:
        tmp = df.rename(columns={bgrp_col: "bgrp_id"})
        return tmp.groupby(["bgrp_id", "week_start"], as_index=False).size().rename(columns={"size": "dob_permits"})

    borough_col = find_column(df.columns, ["borough", "boro"])
    block_col = find_column(df.columns, ["block", "blocknum"])
    if borough_col and block_col and "block_id" in bgrps.columns:
        LOGGER.warning("DOB permits missing bgrp_id; using Borough+Block -> block_id -> bgrp_id fallback")
        tmp = df.copy()
        tmp["block_id"] = tmp[borough_col].astype(str).str.strip() + "_" + pd.to_numeric(
            tmp[block_col], errors="coerce"
        ).fillna(-1).astype(int).astype(str)
        tmp = tmp.merge(bgrps[["block_id", "bgrp_id"]].drop_duplicates(), on="block_id", how="left")
        return tmp.groupby(["bgrp_id", "week_start"], as_index=False).size().rename(columns={"size": "dob_permits"})

    LOGGER.warning("DOB permits alignment failed; distributing weekly totals uniformly")
    wk = df.groupby("week_start", as_index=False).size().rename(columns={"size": "total"})
    if bgrps.empty:
        return pd.DataFrame(columns=["bgrp_id", "week_start", "dob_permits"])
    wk["dob_permits"] = wk["total"] / len(bgrps)
    grid = bgrps[["bgrp_id"]].drop_duplicates().assign(key=1).merge(wk.assign(key=1), on="key", how="inner")
    return grid[["bgrp_id", "week_start", "dob_permits"]]


def compute_dynamic_features(config_path: str) -> None:
    cfg = load_config(config_path).get("paths", {})
    outputs = cfg.get("outputs", {})

    blocks_path = outputs.get("blocks_geojson", "data/processed/blocks.geojson")
    output_path = outputs.get("exposure_weekly_csv", DEFAULT_OUTPUT)

    traffic_path = cfg.get("traffic_csv", cfg.get("centerline_csv", DEFAULT_TRAFFIC_PATH))
    activity_path = cfg.get("activity_csv", DEFAULT_ACTIVITY_PATH)
    permits_path = cfg.get("building_csv", DEFAULT_PERMITS_PATH)

    bgrps = _load_blocks_with_bgrp(blocks_path)
    bgrps = bgrps[[c for c in ["block_id", "bgrp_id", "geometry"] if c in bgrps.columns]]

    traffic = _load_traffic(traffic_path, bgrps)
    permits = _load_permits(permits_path, bgrps)
    weeks = _collect_weeks(traffic, permits)
    activity = _load_activity(activity_path, bgrps, weeks)
    weeks = _collect_weeks(traffic, activity, permits)

    grid = bgrps[["bgrp_id"]].drop_duplicates().assign(key=1).merge(
        pd.DataFrame({"week_start": weeks, "key": 1}), on="key", how="inner"
    ).drop(columns="key")

    out = grid.merge(traffic, on=["bgrp_id", "week_start"], how="left")
    out = out.merge(activity, on=["bgrp_id", "week_start"], how="left")
    out = out.merge(permits, on=["bgrp_id", "week_start"], how="left")

    for col in ["traffic", "street_activity", "dob_permits"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out = out.dropna(subset=["bgrp_id", "week_start"])
    out["week_start"] = out["week_start"].dt.strftime("%Y-%m-%d")

    out = out.sort_values(["bgrp_id", "week_start"]).drop_duplicates(["bgrp_id", "week_start"], keep="first")
    dupes = out.duplicated(["bgrp_id", "week_start"]).sum()
    if dupes > 0:
        raise ValueError(f"Duplicate keys found in exposure_weekly: {dupes}")

    LOGGER.info(
        "Exposure summary: bgrps=%s, weeks=%s, traffic[min,max]=[%.3f,%.3f], street_activity[min,max]=[%.3f,%.3f], dob_permits[min,max]=[%.3f,%.3f]",
        out["bgrp_id"].nunique(),
        out["week_start"].nunique(),
        out["traffic"].min(),
        out["traffic"].max(),
        out["street_activity"].min(),
        out["street_activity"].max(),
        out["dob_permits"].min(),
        out["dob_permits"].max(),
    )

    ensure_parent_dir(output_path)
    out.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(out), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dynamic features at bgrp_id x week_start")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    compute_dynamic_features(args.config)
