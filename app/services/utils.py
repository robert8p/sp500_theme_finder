from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")



def save_frame(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401

        df.to_parquet(path.with_suffix('.parquet'), index=False)
        return path.with_suffix('.parquet')
    except Exception:
        csv_path = path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        return csv_path



def load_frame(path_without_suffix: Path) -> pd.DataFrame:
    parquet_path = path_without_suffix.with_suffix('.parquet')
    csv_path = path_without_suffix.with_suffix('.csv')
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No persisted frame found for {path_without_suffix}")
