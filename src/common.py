from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

TARGET_EPSG = 2263


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config or "paths" not in config:
        raise ValueError("Config must define a top-level 'paths' object")
    return config


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    index = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in index:
            return index[key]
    return None


def numeric_column(df: pd.DataFrame, candidates: Iterable[str], default: float = 0.0) -> pd.Series:
    col = find_column(df.columns, candidates)
    if col is None:
        LOGGER.warning("No candidate columns found from %s; using default=%s", list(candidates), default)
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Borough" in out:
        out["Borough"] = out["Borough"].astype(str).str.strip()
    if "Block" in out:
        out["Block"] = pd.to_numeric(out["Block"], errors="coerce").fillna(-1).astype(int).astype(str)
    if "BBL" in out:
        out["BBL"] = out["BBL"].astype(str).str.replace(".0$", "", regex=True).str.strip()
    return out
