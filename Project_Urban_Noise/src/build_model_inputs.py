from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ensure_parent_dir, load_config, setup_logging

LOGGER = logging.getLogger(__name__)

DEFAULT_MONTHLY_CSV = "data/processed/monthly_dynamic_with_activity.csv"
DEFAULT_SCALER_PKL = "data/processed/scaler.pkl"
DEFAULT_MODEL_INPUT_CSV = "data/processed/model_inputs_monthly.csv"
DEFAULT_MODEL_TENSOR_NPZ = "data/processed/model_inputs_monthly_tensors.npz"


def _load_monthly_table(outputs: dict) -> pd.DataFrame:
    parquet_path = outputs.get("monthly_dynamic_parquet", "data/processed/monthly_dynamic.parquet")
    csv_path = outputs.get("monthly_dynamic_csv", DEFAULT_MONTHLY_CSV)
    if Path(parquet_path).exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            LOGGER.warning("Failed reading parquet (%s). Falling back to CSV.", exc)
    if Path(csv_path).exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"Missing monthly dynamic table. Expected {parquet_path} or {csv_path}")


def _pick_feature_cols(df: pd.DataFrame) -> list[str]:
    prefixes = ("traffic_", "event_", "weather_")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    if not cols:
        raise ValueError("No dynamic feature columns found. Expected prefixes: traffic_, event_, weather_")
    return cols


def _split_by_month(df: pd.DataFrame, model_cfg: dict) -> pd.DataFrame:
    months = sorted(df["month"].dropna().unique().tolist())
    if not months:
        raise ValueError("No valid month values found in monthly table")

    train_end = model_cfg.get("train_end_month")
    val_end = model_cfg.get("val_end_month")
    if train_end and val_end:
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts = pd.Timestamp(val_end)
    else:
        n = len(months)
        train_end_ts = months[max(int(n * 0.7) - 1, 0)]
        val_end_ts = months[max(int(n * 0.85) - 1, 0)]

    out = df.copy()
    out["split"] = "test"
    out.loc[out["month"] <= train_end_ts, "split"] = "train"
    out.loc[(out["month"] > train_end_ts) & (out["month"] <= val_end_ts), "split"] = "val"
    return out


def _fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    means = train_df[feature_cols].mean(axis=0, skipna=True).fillna(0.0)
    stds = train_df[feature_cols].std(axis=0, ddof=0, skipna=True).replace(0, 1.0).fillna(1.0)
    return {"feature_cols": feature_cols, "mean": means.to_dict(), "std": stds.to_dict()}


def _apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = df.copy()
    for col in scaler["feature_cols"]:
        mean = float(scaler["mean"][col])
        std = float(scaler["std"][col]) if float(scaler["std"][col]) != 0 else 1.0
        out[col] = (pd.to_numeric(out[col], errors="coerce").fillna(mean) - mean) / std
    return out


def _build_tensor_npz(df: pd.DataFrame, feature_cols: list[str], path: str) -> None:
    work = df.copy()
    work["month"] = pd.to_datetime(work["month"], errors="coerce")
    work = work.dropna(subset=["month"])

    bgrp_ids = sorted(work["bgrp_id"].astype(str).unique().tolist())
    months = sorted(pd.Timestamp(m) for m in work["month"].dropna().unique().tolist())
    full = pd.MultiIndex.from_product([bgrp_ids, months], names=["bgrp_id", "month"]).to_frame(index=False)
    full = full.merge(work[["bgrp_id", "month"] + feature_cols], on=["bgrp_id", "month"], how="left")
    full[feature_cols] = full[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    n_b = len(bgrp_ids)
    n_m = len(months)
    n_f = len(feature_cols)
    X = np.zeros((n_b, n_m, n_f), dtype=np.float32)

    b_map = {k: i for i, k in enumerate(bgrp_ids)}
    m_map = {k: i for i, k in enumerate(months)}
    for row in full.itertuples(index=False):
        bi = b_map[row.bgrp_id]
        mi = m_map[row.month]
        X[bi, mi, :] = np.asarray([getattr(row, c) for c in feature_cols], dtype=np.float32)

    ensure_parent_dir(path)
    np.savez_compressed(
        path,
        X=X,
        bgrp_ids=np.asarray(bgrp_ids, dtype=object),
        months=np.asarray([m.strftime("%Y-%m-%d") for m in months], dtype=object),
        feature_cols=np.asarray(feature_cols, dtype=object),
    )
    LOGGER.info("Wrote tensor NPZ: %s shape=%s", path, X.shape)


def build_model_inputs(config_path: str) -> None:
    cfg = load_config(config_path)["paths"]
    outputs = cfg.get("outputs", {})
    model_cfg = cfg.get("model_input", {})

    df = _load_monthly_table(outputs)
    if "bgrp_id" not in df.columns or "month" not in df.columns:
        raise ValueError("Monthly table must include bgrp_id and month columns")

    df = df.copy()
    df["bgrp_id"] = df["bgrp_id"].astype(str)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["bgrp_id", "month"]).sort_values(["bgrp_id", "month"])

    feature_cols = _pick_feature_cols(df)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    split_df = _split_by_month(df, model_cfg)

    train_df = split_df[split_df["split"] == "train"]
    scaler = _fit_scaler(train_df, feature_cols)
    scaled = _apply_scaler(split_df, scaler)

    scaler_out = outputs.get("scaler_pkl", DEFAULT_SCALER_PKL)
    ensure_parent_dir(scaler_out)
    with open(scaler_out, "wb") as f:
        pickle.dump(scaler, f)
    LOGGER.info("Wrote scaler: %s", scaler_out)

    model_csv = outputs.get("model_inputs_monthly_csv", DEFAULT_MODEL_INPUT_CSV)
    ensure_parent_dir(model_csv)
    csv_out = scaled.copy()
    csv_out["month"] = csv_out["month"].dt.strftime("%Y-%m-%d")
    csv_out.to_csv(model_csv, index=False)
    LOGGER.info("Wrote model input CSV: %s rows=%s", model_csv, len(scaled))

    npz_out = outputs.get("model_tensors_npz", DEFAULT_MODEL_TENSOR_NPZ)
    _build_tensor_npz(scaled, feature_cols, npz_out)

    LOGGER.info(
        "Split summary train=%s val=%s test=%s",
        int((scaled["split"] == "train").sum()),
        int((scaled["split"] == "val").sum()),
        int((scaled["split"] == "test").sum()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-ready inputs from monthly dynamic table")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    build_model_inputs(args.config)
