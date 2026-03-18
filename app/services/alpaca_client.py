from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from ..config import settings

NY_TZ = "America/New_York"


class AlpacaClient:
    def __init__(self) -> None:
        self.base_url = settings.alpaca_base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": settings.alpaca_api_key,
                "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
                "Accept": "application/json",
            }
        )

    def _request(self, path: str, params: Dict[str, str]) -> Dict:
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                response = self.session.get(url, params=params, timeout=settings.alpaca_timeout_seconds)
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Alpaca request failed for {url}: {last_error}")

    def fetch_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")

        page_token = None
        rows: List[Dict] = []
        while True:
            params = {
                "symbols": symbol,
                "timeframe": settings.data_interval,
                "start": start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
                "end": end.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
                "adjustment": "raw",
                "feed": "sip",
                "sort": "asc",
                "limit": "10000",
            }
            if page_token:
                params["page_token"] = page_token
            payload = self._request("/v2/stocks/bars", params)
            bars = payload.get("bars", {}).get(symbol, [])
            rows.extend(bars)
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        if not rows:
            return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"])

        df = pd.DataFrame(rows)
        df = df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "n": "trade_count",
                "vw": "vwap",
            }
        )
        df["symbol"] = symbol
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY_TZ)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.set_index("timestamp")
        df = df.between_time("09:30", "16:00", inclusive="left").reset_index()
        df["session_date"] = df["timestamp"].dt.date.astype(str)
        return df


def default_date_range() -> Tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(settings.lookback_months * 31))
    return start, end
