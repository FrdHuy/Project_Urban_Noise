from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely import wkt

from .common import TARGET_EPSG, ensure_parent_dir, find_column, load_config, setup_logging

LOGGER = logging.getLogger(__name__)

DEFAULT_MONTHLY_CSV = "data/processed/monthly_dynamic_with_activity.csv"
DEFAULT_MONTHLY_PARQUET = "data/processed/monthly_dynamic.parquet"
DEFAULT_QUALITY_CSV = "data/processed/monthly_dynamic_quality.csv"


def _month_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp(how="start")


def _normalize_bbl(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.replace(r"\D", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def _normalize_block_id(df: pd.DataFrame, borough_col: str, block_col: str) -> pd.Series:
    borough = df[borough_col].astype(str).str.strip()
    block = pd.to_numeric(df[block_col], errors="coerce").fillna(-1).astype(int).astype(str)
    return borough + "_" + block


def _load_spatial_base(paths: dict, outputs: dict) -> gpd.GeoDataFrame:
    bgrp_geo = paths.get("bgrp_geojson")
    blocks_geo = outputs.get("blocks_geojson", "data/processed/blocks.geojson")

    if bgrp_geo and Path(bgrp_geo).exists():
        gdf = gpd.read_file(bgrp_geo)
        bgrp_col = find_column(gdf.columns, ["bgrp_id", "geoid", "geoid20", "bg_id", "block_group"])
        if bgrp_col is None:
            raise ValueError(f"bgrp_geojson exists but no bgrp key found: {bgrp_geo}")
        gdf["bgrp_id"] = gdf[bgrp_col].astype(str)
        if gdf.crs is None:
            raise ValueError(f"bgrp_geojson has no CRS: {bgrp_geo}")
        if int(gdf.crs.to_epsg() or -1) != TARGET_EPSG:
            gdf = gdf.to_crs(epsg=TARGET_EPSG)
        out_cols = ["bgrp_id", "geometry"]
        for c in ["Borough", "Block", "block_id"]:
            if c in gdf.columns:
                out_cols.append(c)
        return gdf[out_cols].drop_duplicates()

    if not Path(blocks_geo).exists():
        raise FileNotFoundError("Missing both bgrp_geojson and blocks_geojson fallback")

    LOGGER.warning("No bgrp_geojson configured; using block_id as bgrp_id fallback")
    gdf = gpd.read_file(blocks_geo)
    if gdf.crs is None:
        raise ValueError("blocks.geojson has no CRS")
    if int(gdf.crs.to_epsg() or -1) != TARGET_EPSG:
        gdf = gdf.to_crs(epsg=TARGET_EPSG)
    gdf["bgrp_id"] = gdf["block_id"].astype(str)
    return gdf[["bgrp_id", "block_id", "Borough", "Block", "geometry"]].drop_duplicates()


def _points_to_bgrp(points: gpd.GeoDataFrame, bgrp: gpd.GeoDataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=["bgrp_id"])
    joined = gpd.sjoin(points, bgrp[["bgrp_id", "geometry"]], how="left", predicate="within")
    bgrp_col = "bgrp_id"
    if bgrp_col not in joined.columns:
        candidates = [c for c in joined.columns if str(c).lower().startswith("bgrp_id")]
        if candidates:
            bgrp_col = candidates[0]
        else:
            raise KeyError("Spatial join produced no bgrp_id-like column")
    if bgrp_col != "bgrp_id":
        joined["bgrp_id"] = joined[bgrp_col]

    miss = joined["bgrp_id"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(
            joined.loc[miss, ["geometry"]],
            bgrp[["bgrp_id", "geometry"]],
            how="left",
            max_distance=150.0,
            distance_col="_dist",
        )
        joined.loc[miss, "bgrp_id"] = near["bgrp_id"].values
    return pd.DataFrame(joined.drop(columns=["index_right"], errors="ignore"))


def _build_bbl_crosswalk(paths: dict, bgrp: gpd.GeoDataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled:
        LOGGER.info("BBL crosswalk disabled by config (use_bbl_crosswalk=false)")
        return pd.DataFrame(columns=["BBL_norm", "bgrp_id"])

    gdb_path = paths.get("mappluto_gdb")
    layer = paths.get("mappluto_layer")
    if not gdb_path or not layer or not Path(gdb_path).exists():
        LOGGER.warning("MapPLUTO unavailable; BBL->bgrp crosswalk disabled")
        return pd.DataFrame(columns=["BBL_norm", "bgrp_id"])

    cols = ["BBL", "Borough", "Block", "geometry"]
    parcels = gpd.read_file(gdb_path, layer=layer, columns=cols)
    bbl_col = find_column(parcels.columns, ["BBL", "MapBLLot"])
    if bbl_col is None:
        LOGGER.warning("MapPLUTO has no BBL-like column; BBL crosswalk disabled")
        return pd.DataFrame(columns=["BBL_norm", "bgrp_id"])
    parcels["BBL_norm"] = _normalize_bbl(parcels[bbl_col])
    parcels = parcels[parcels["BBL_norm"].notna()].copy()

    # Fast key-based route for block fallback base.
    if "block_id" in bgrp.columns:
        borough_col = find_column(parcels.columns, ["Borough", "boro", "BoroCode"])
        block_col = find_column(parcels.columns, ["Block", "block", "BlockNum"])
        if borough_col and block_col:
            parcels["block_id"] = _normalize_block_id(parcels, borough_col, block_col)
            key_map = bgrp[["block_id", "bgrp_id"]].drop_duplicates()
            out = parcels.merge(key_map, on="block_id", how="left")[["BBL_norm", "bgrp_id"]].dropna().drop_duplicates()
            LOGGER.info("Built BBL crosswalk via key join rows=%s", len(out))
            if not out.empty:
                return out

    if parcels.crs is None:
        raise ValueError("MapPLUTO parcels have no CRS")
    if int(parcels.crs.to_epsg() or -1) != TARGET_EPSG:
        parcels = parcels.to_crs(epsg=TARGET_EPSG)

    assign = gpd.sjoin(parcels[["BBL_norm", "geometry"]], bgrp[["bgrp_id", "geometry"]], how="left", predicate="within")
    if assign["bgrp_id"].isna().any():
        miss = assign["bgrp_id"].isna()
        near = gpd.sjoin_nearest(
            assign.loc[miss, ["BBL_norm", "geometry"]],
            bgrp[["bgrp_id", "geometry"]],
            how="left",
            max_distance=150.0,
            distance_col="_dist",
        )
        assign.loc[miss, "bgrp_id"] = near["bgrp_id"].values
    out = assign[["BBL_norm", "bgrp_id"]].dropna().drop_duplicates()
    LOGGER.info("Built BBL crosswalk via spatial fallback rows=%s", len(out))
    return out


def _aggregate_traffic_automated(path: str, segment_map: pd.DataFrame) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["bgrp_id", "month", "traffic_volume_sum", "traffic_obs_count"])
    usecols = ["Yr", "M", "D", "Vol", "SegmentID"]
    raw = pd.read_csv(path, usecols=[c for c in usecols if c], low_memory=False)
    dt = pd.to_datetime(raw[["Yr", "M", "D"]].rename(columns={"Yr": "year", "M": "month", "D": "day"}), errors="coerce")
    raw["month"] = _month_start(dt)
    raw["traffic_val"] = pd.to_numeric(raw.get("Vol"), errors="coerce").fillna(0.0)
    raw["SegmentID"] = pd.to_numeric(raw.get("SegmentID"), errors="coerce")
    raw = raw.dropna(subset=["month", "SegmentID"]).copy()
    agg_seg = (
        raw.groupby(["SegmentID", "month"], as_index=False)
        .agg(traffic_volume_sum=("traffic_val", "sum"), traffic_obs_count=("traffic_val", "size"))
    )
    mapped = agg_seg.merge(segment_map, on="SegmentID", how="left")
    out = (
        mapped.dropna(subset=["bgrp_id", "month"])
        .groupby(["bgrp_id", "month"], as_index=False)
        .agg(traffic_volume_sum=("traffic_volume_sum", "sum"), traffic_obs_count=("traffic_obs_count", "sum"))
    )
    return out


def _aggregate_traffic_historical(path: str, bgrp: gpd.GeoDataFrame, segment_map: pd.DataFrame) -> pd.DataFrame:
    if not path or not Path(path).exists() or segment_map.empty:
        return pd.DataFrame(columns=["bgrp_id", "month", "traffic_hist_daily_sum"])
    df = pd.read_csv(path, low_memory=False)
    date_col = find_column(df.columns, ["Date", "date"])
    seg_col = find_column(df.columns, ["SegmentID", "segmentid"])
    if date_col is None or seg_col is None:
        return pd.DataFrame(columns=["bgrp_id", "month", "traffic_hist_daily_sum"])

    hourly_cols: list[str] = []
    for col in df.columns:
        c = str(col).strip().lower()
        if "am" in c or "pm" in c:
            hourly_cols.append(col)
    if not hourly_cols:
        return pd.DataFrame(columns=["bgrp_id", "month", "traffic_hist_daily_sum"])

    hist = df[[date_col, seg_col] + hourly_cols].copy()
    hist["daily_total"] = hist[hourly_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    hist["month"] = _month_start(hist[date_col])
    hist["SegmentID"] = pd.to_numeric(hist[seg_col], errors="coerce")
    hist = hist.dropna(subset=["month", "SegmentID"])
    hist = hist.merge(segment_map, on="SegmentID", how="left")
    out = (
        hist.dropna(subset=["bgrp_id", "month"])
        .groupby(["bgrp_id", "month"], as_index=False)
        .agg(traffic_hist_daily_sum=("daily_total", "sum"))
    )
    return out


def _segment_to_bgrp_map(traffic_path: str, bgrp: gpd.GeoDataFrame) -> pd.DataFrame:
    if not traffic_path or not Path(traffic_path).exists():
        return pd.DataFrame(columns=["SegmentID", "bgrp_id"])
    usecols = ["SegmentID", "WktGeom"]
    df = pd.read_csv(traffic_path, usecols=usecols, low_memory=False)
    df["SegmentID"] = pd.to_numeric(df["SegmentID"], errors="coerce")
    df = df.dropna(subset=["SegmentID", "WktGeom"]).drop_duplicates("SegmentID")
    if df.empty:
        return pd.DataFrame(columns=["SegmentID", "bgrp_id"])
    df["geometry"] = df["WktGeom"].map(lambda v: wkt.loads(v) if pd.notna(v) else None)
    pts = gpd.GeoDataFrame(df.dropna(subset=["geometry"]), geometry="geometry", crs=f"EPSG:{TARGET_EPSG}")
    mapped = _points_to_bgrp(pts, bgrp)
    out = mapped[["SegmentID", "bgrp_id"]].dropna().drop_duplicates("SegmentID")
    return out


def _active_months_rows(df: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
    rows: list[tuple[str, pd.Timestamp]] = []
    min_month = pd.Timestamp("2000-01-01")
    max_month = pd.Timestamp.today().to_period("M").to_timestamp(how="start")
    for bgrp_id, start, end in df[["bgrp_id", start_col, end_col]].itertuples(index=False):
        if pd.isna(bgrp_id) or pd.isna(start):
            continue
        start_m = pd.Timestamp(start).to_period("M").to_timestamp(how="start")
        if pd.isna(end):
            end_m = pd.Timestamp.today().to_period("M").to_timestamp(how="start")
        else:
            end_m = pd.Timestamp(end).to_period("M").to_timestamp(how="start")
        start_m = max(start_m, min_month)
        end_m = min(end_m, max_month)
        if end_m < start_m:
            continue
        months = pd.period_range(start_m, end_m, freq="M").to_timestamp(how="start")
        rows.extend((str(bgrp_id), m) for m in months)
    return pd.DataFrame(rows, columns=["bgrp_id", "month"])


def _aggregate_activity(path: str, bgrp: gpd.GeoDataFrame, bbl_map: pd.DataFrame) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["bgrp_id", "month", "event_new_count", "event_active_count"])

    df = pd.read_csv(path, low_memory=False)
    df["bgrp_id"] = pd.NA

    bbl_col = find_column(df.columns, ["BBL", "bbl"])
    if bbl_col and not bbl_map.empty:
        df["BBL_norm"] = _normalize_bbl(df[bbl_col])
        df = df.merge(bbl_map, on="BBL_norm", how="left", suffixes=("", "_bbl"))
        df["bgrp_id"] = df["bgrp_id"].fillna(df.get("bgrp_id_bbl"))
        df = df.drop(columns=[c for c in ["bgrp_id_bbl"] if c in df.columns])

    lon_col = find_column(df.columns, ["Longitude", "longitude", "lon"])
    lat_col = find_column(df.columns, ["Latitude", "latitude", "lat"])
    if lon_col and lat_col:
        miss = df["bgrp_id"].isna()
        if miss.any():
            work = df.loc[miss].copy()
            work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
            work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
            work = work.dropna(subset=[lon_col, lat_col])
            if not work.empty:
                pts = gpd.GeoDataFrame(work, geometry=gpd.points_from_xy(work[lon_col], work[lat_col]), crs="EPSG:4326")
                pts = pts.to_crs(epsg=TARGET_EPSG)
                mapped = _points_to_bgrp(pts, bgrp)
                df.loc[mapped.index, "bgrp_id"] = mapped["bgrp_id"].values

    issue_col = find_column(df.columns, ["Initial Issuance Date", "issue_date", "start_date"])
    exp_col = find_column(df.columns, ["Expiration Date", "end_date", "expire_date"])
    if issue_col is None:
        return pd.DataFrame(columns=["bgrp_id", "month", "event_new_count", "event_active_count"])

    df["issue_dt"] = pd.to_datetime(df[issue_col], errors="coerce")
    df["exp_dt"] = pd.to_datetime(df[exp_col], errors="coerce") if exp_col else pd.NaT
    new_counts = (
        df.dropna(subset=["bgrp_id", "issue_dt"])
        .assign(month=lambda d: _month_start(d["issue_dt"]))
        .groupby(["bgrp_id", "month"], as_index=False)
        .size()
        .rename(columns={"size": "event_new_count"})
    )

    active_rows = _active_months_rows(df.dropna(subset=["bgrp_id", "issue_dt"]), "issue_dt", "exp_dt")
    active_counts = (
        active_rows.groupby(["bgrp_id", "month"], as_index=False)
        .size()
        .rename(columns={"size": "event_active_count"})
    )

    out = new_counts.merge(active_counts, on=["bgrp_id", "month"], how="outer")
    out["event_new_count"] = pd.to_numeric(out["event_new_count"], errors="coerce").fillna(0.0)
    out["event_active_count"] = pd.to_numeric(out["event_active_count"], errors="coerce").fillna(0.0)
    return out


def _aggregate_weather(path: str, bgrp: gpd.GeoDataFrame) -> pd.DataFrame:
    if not path or not Path(path).exists():
        LOGGER.warning("Weather file not found, skipping weather features")
        return pd.DataFrame(columns=["bgrp_id", "month"])
    df = pd.read_csv(path, low_memory=False)
    date_col = find_column(df.columns, ["date", "datetime", "timestamp", "month"])
    if date_col is None:
        LOGGER.warning("Weather file has no date-like column, skipping")
        return pd.DataFrame(columns=["bgrp_id", "month"])
    df["month"] = _month_start(df[date_col])

    out = None
    bgrp_col = find_column(df.columns, ["bgrp_id", "geoid", "bg_id"])
    if bgrp_col:
        out = df.rename(columns={bgrp_col: "bgrp_id"})

    if out is None:
        lon_col = find_column(df.columns, ["longitude", "lon"])
        lat_col = find_column(df.columns, ["latitude", "lat"])
        if lon_col and lat_col:
            work = df.copy()
            work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
            work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
            work = work.dropna(subset=[lon_col, lat_col])
            if not work.empty:
                pts = gpd.GeoDataFrame(work, geometry=gpd.points_from_xy(work[lon_col], work[lat_col]), crs="EPSG:4326")
                mapped = _points_to_bgrp(pts.to_crs(epsg=TARGET_EPSG), bgrp)
                out = pd.DataFrame(mapped)

    if out is None:
        LOGGER.warning("Weather file has no bgrp key or coordinates, using citywide monthly averages")
        out = df.assign(_dummy=1).merge(bgrp[["bgrp_id"]].assign(_dummy=1), on="_dummy", how="inner").drop(columns="_dummy")

    numeric_cols: list[str] = []
    for col in out.columns:
        if str(col) in {"bgrp_id", "month"}:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
            out[col] = series
    if not numeric_cols:
        return pd.DataFrame(columns=["bgrp_id", "month"])

    agg = out.groupby(["bgrp_id", "month"], as_index=False)[numeric_cols].mean()
    rename = {c: f"weather_{c.strip().lower().replace(' ', '_')}" for c in numeric_cols}
    return agg.rename(columns=rename)


def _outer_months(frames: Iterable[pd.DataFrame]) -> pd.DatetimeIndex:
    months = pd.Series(dtype="datetime64[ns]")
    for df in frames:
        if "month" in df.columns:
            months = pd.concat([months, pd.to_datetime(df["month"], errors="coerce")], ignore_index=True)
    months = months.dropna().drop_duplicates().sort_values()
    if months.empty:
        now = pd.Timestamp.today().to_period("M").to_timestamp(how="start")
        return pd.DatetimeIndex([now])
    return pd.DatetimeIndex(months)


def _clip_month_window(df: pd.DataFrame, start_month: pd.Timestamp, end_month: pd.Timestamp) -> pd.DataFrame:
    if "month" not in df.columns or df.empty:
        return df
    out = df.copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    out = out.dropna(subset=["month"])
    return out[(out["month"] >= start_month) & (out["month"] <= end_month)]


def _write_quality_report(df: pd.DataFrame, feature_cols: list[str], path: str) -> None:
    rows = []
    total = len(df)
    for c in feature_cols:
        miss = df[c].isna().sum()
        zeros = (pd.to_numeric(df[c], errors="coerce").fillna(0.0) == 0).sum()
        rows.append(
            {
                "column": c,
                "rows": total,
                "missing_count": int(miss),
                "missing_rate": float(miss / total if total else 0.0),
                "zero_count": int(zeros),
                "zero_rate": float(zeros / total if total else 0.0),
            }
        )
    q = pd.DataFrame(rows)
    ensure_parent_dir(path)
    q.to_csv(path, index=False)


def build_monthly_dynamic_features(config_path: str) -> None:
    cfg = load_config(config_path)["paths"]
    outputs = cfg.get("outputs", {})

    bgrp = _load_spatial_base(cfg, outputs)
    bbl_map = _build_bbl_crosswalk(cfg, bgrp, enabled=bool(cfg.get("use_bbl_crosswalk", False)))

    traffic_csv = cfg.get("traffic_csv")
    traffic_hist = cfg.get("traffic_historical_csv", "data/raw/Traffic_Volume_Counts_(Historical)_20260209.csv")
    activity_csv = cfg.get("activity_csv")
    weather_csv = cfg.get("weather_csv")

    start_month = pd.Timestamp(cfg.get("monthly_start_month", "2000-01-01")).to_period("M").to_timestamp(how="start")
    end_month = pd.Timestamp(cfg.get("monthly_end_month", pd.Timestamp.today().strftime("%Y-%m-01"))).to_period("M").to_timestamp(how="start")

    segment_map = _segment_to_bgrp_map(traffic_csv, bgrp)
    traffic_auto = _aggregate_traffic_automated(traffic_csv, segment_map)
    traffic_hist_df = _aggregate_traffic_historical(traffic_hist, bgrp, segment_map)
    traffic = traffic_auto.merge(traffic_hist_df, on=["bgrp_id", "month"], how="outer")

    activity = _aggregate_activity(activity_csv, bgrp, bbl_map)
    weather = _aggregate_weather(weather_csv, bgrp)

    traffic = _clip_month_window(traffic, start_month, end_month)
    activity = _clip_month_window(activity, start_month, end_month)
    weather = _clip_month_window(weather, start_month, end_month)

    months = _outer_months([traffic, activity, weather])
    grid = bgrp[["bgrp_id"]].drop_duplicates().assign(_k=1).merge(
        pd.DataFrame({"month": months, "_k": 1}), on="_k", how="inner"
    ).drop(columns="_k")

    out = grid.merge(traffic, on=["bgrp_id", "month"], how="left")
    out = out.merge(activity, on=["bgrp_id", "month"], how="left")
    out = out.merge(weather, on=["bgrp_id", "month"], how="left")

    feature_cols = [c for c in out.columns if c not in {"bgrp_id", "month"}]
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values(["bgrp_id", "month"]).drop_duplicates(["bgrp_id", "month"], keep="first")
    out["month"] = pd.to_datetime(out["month"], errors="coerce").dt.strftime("%Y-%m-%d")

    missing = out.isna().sum().sum()
    LOGGER.info("Monthly rows=%s bgrps=%s months=%s missing_cells=%s", len(out), out["bgrp_id"].nunique(), out["month"].nunique(), int(missing))

    csv_out = outputs.get("monthly_dynamic_csv", DEFAULT_MONTHLY_CSV)
    parquet_out = outputs.get("monthly_dynamic_parquet", DEFAULT_MONTHLY_PARQUET)
    quality_out = outputs.get("monthly_dynamic_quality_csv", DEFAULT_QUALITY_CSV)

    _write_quality_report(out, feature_cols, quality_out)
    LOGGER.info("Wrote quality report: %s", quality_out)

    out[feature_cols] = out[feature_cols].fillna(0.0)

    ensure_parent_dir(csv_out)
    out.to_csv(csv_out, index=False)
    LOGGER.info("Wrote monthly CSV: %s", csv_out)

    ensure_parent_dir(parquet_out)
    try:
        out.to_parquet(parquet_out, index=False)
        LOGGER.info("Wrote monthly Parquet: %s", parquet_out)
    except Exception as exc:
        LOGGER.warning("Failed to write parquet (%s). Install pyarrow/fastparquet.", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly dynamic features at bgrp_id x month")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    build_monthly_dynamic_features(args.config)
