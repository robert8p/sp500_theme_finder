from __future__ import annotations

import io
from typing import List

import pandas as pd
import requests

from ..config import settings

SP500_CACHE = settings.cache_dir / "sp500_constituents.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_symbol(symbol: str) -> str:
    return symbol.replace(".", "-").strip().upper()


def _download_sp500_table() -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(WIKI_URL, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    if not tables:
        raise RuntimeError("No tables found on the S&P 500 Wikipedia page.")

    table = tables[0].copy()
    table.columns = [str(c).strip().lower().replace(" ", "_") for c in table.columns]

    rename_map = {
        "symbol": "symbol",
        "security": "security",
        "gics_sector": "sector",
        "gics_sub_industry": "sub_industry",
        "date_added": "date_added",
        "cik": "cik",
        "founded": "founded",
    }

    out = table.rename(columns=rename_map)
    keep = [
        c
        for c in ["symbol", "security", "sector", "sub_industry", "date_added", "cik", "founded"]
        if c in out.columns
    ]
    out = out[keep].copy()
    out["symbol"] = out["symbol"].astype(str).map(normalize_symbol)
    out = out.sort_values("symbol").reset_index(drop=True)
    return out


def load_sp500_constituents() -> pd.DataFrame:
    if settings.use_cached_sp500 and SP500_CACHE.exists():
        df = pd.read_csv(SP500_CACHE)
        if not df.empty:
            df["symbol"] = df["symbol"].astype(str).map(normalize_symbol)
            return df

    out = _download_sp500_table()
    SP500_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SP500_CACHE, index=False)
    return out


def sp500_symbols() -> List[str]:
    df = load_sp500_constituents()
    symbols = df["symbol"].dropna().astype(str).tolist()
    if settings.max_symbols and settings.max_symbols > 0:
        symbols = symbols[: settings.max_symbols]
    return symbols
