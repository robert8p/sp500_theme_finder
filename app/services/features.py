from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..config import settings

MARKET_OPEN_MINUTES = 9 * 60 + 30
MARKET_CLOSE_MINUTES = 16 * 60


FEATURE_COLUMNS = [
    "sma_20",
    "sma_50",
    "ema_9",
    "ema_20",
    "ema_50",
    "dist_sma_20_pct",
    "dist_ema_20_pct",
    "ema9_gt_ema20",
    "ema20_gt_ema50",
    "ma_stack_bullish",
    "ema_9_slope_3",
    "ema_20_slope_5",
    "rsi_14",
    "stoch_k_14",
    "roc_12",
    "ret_1",
    "ret_3",
    "ret_6",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14_pct",
    "realized_vol_12",
    "rolling_std_12",
    "bar_range_pct",
    "true_range_pct",
    "compression_12",
    "range_expansion_flag",
    "relative_volume_20",
    "volume_zscore_20",
    "obv",
    "obv_slope_5",
    "vwap_session",
    "dist_vwap_pct",
    "close_gt_vwap",
    "vwap_slope_5",
    "dist_intraday_high_pct",
    "dist_intraday_low_pct",
    "breakout_20_flag",
    "breakout_50_flag",
    "opening_range_break_flag",
    "pullback_from_high_pct",
    "higher_high_higher_low_3",
    "minutes_since_open",
    "minutes_to_close",
    "return_since_open",
    "gap_from_prev_close_pct",
    "spy_return_since_open",
    "rel_strength_vs_spy",
    "sector_return_since_open",
    "rel_strength_vs_sector",
    "vol_regime_ratio",
    "log_dollar_volume",
    "avg_dollar_volume_20",
]


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()



def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))



def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    rolling_low = low.rolling(window, min_periods=window).min()
    rolling_high = high.rolling(window, min_periods=window).max()
    denom = (rolling_high - rolling_low).replace(0, np.nan)
    return 100 * (close - rolling_low) / denom



def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()



def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()



def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std()
    return (s - mean) / std.replace(0, np.nan)



def _session_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].fillna(0)

    df["bar_index"] = np.arange(len(df))
    df["session_open"] = df["open"].iloc[0]
    df["session_high_so_far"] = high.cummax()
    df["session_low_so_far"] = low.cummin()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    df["vwap_session"] = cum_pv / cum_vol
    df["return_since_open"] = close / df["session_open"] - 1
    df["dist_intraday_high_pct"] = (df["session_high_so_far"] - close) / close
    df["dist_intraday_low_pct"] = (close - df["session_low_so_far"]) / close
    df["pullback_from_high_pct"] = (df["session_high_so_far"] - close) / df["session_high_so_far"].replace(0, np.nan)

    timestamp_local = df["timestamp"].dt.tz_convert("America/New_York")
    minutes = timestamp_local.dt.hour * 60 + timestamp_local.dt.minute
    df["minutes_since_open"] = minutes - MARKET_OPEN_MINUTES
    df["minutes_to_close"] = MARKET_CLOSE_MINUTES - minutes

    opening_range_high = df.loc[df["bar_index"] < 6, "high"].max()
    df["opening_range_high"] = opening_range_high
    df["opening_range_break_flag"] = (close > opening_range_high).astype(int)

    return df



def _symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].fillna(0)

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
    df["roc_12"] = close.pct_change(12)

    df["sma_20"] = close.rolling(20, min_periods=20).mean()
    df["sma_50"] = close.rolling(50, min_periods=50).mean()
    df["ema_9"] = _ema(close, 9)
    df["ema_20"] = _ema(close, 20)
    df["ema_50"] = _ema(close, 50)

    df["dist_sma_20_pct"] = close / df["sma_20"] - 1
    df["dist_ema_20_pct"] = close / df["ema_20"] - 1
    df["ema9_gt_ema20"] = (df["ema_9"] > df["ema_20"]).astype(int)
    df["ema20_gt_ema50"] = (df["ema_20"] > df["ema_50"]).astype(int)
    df["ma_stack_bullish"] = ((df["ema_9"] > df["ema_20"]) & (df["ema_20"] > df["ema_50"])) .astype(int)
    df["ema_9_slope_3"] = df["ema_9"].pct_change(3)
    df["ema_20_slope_5"] = df["ema_20"].pct_change(5)

    df["rsi_14"] = _rsi(close, 14)
    df["stoch_k_14"] = _stochastic(high, low, close, 14)
    df["macd"] = _ema(close, 12) - _ema(close, 26)
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    atr = _atr(high, low, close, 14)
    df["atr_14_pct"] = atr / close
    df["bar_range_pct"] = (high - low) / close
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["true_range_pct"] = true_range / close
    df["rolling_std_12"] = df["ret_1"].rolling(12, min_periods=12).std()
    df["realized_vol_12"] = df["ret_1"].rolling(12, min_periods=12).std() * math.sqrt(12)
    rolling_range_mean = df["bar_range_pct"].rolling(12, min_periods=12).mean()
    df["compression_12"] = df["bar_range_pct"] / rolling_range_mean
    df["range_expansion_flag"] = (df["bar_range_pct"] > rolling_range_mean * 1.5).astype(int)

    vol_mean_20 = volume.rolling(20, min_periods=20).mean()
    df["relative_volume_20"] = volume / vol_mean_20
    df["volume_zscore_20"] = _rolling_zscore(volume, 20)
    df["obv"] = _obv(close, volume)
    df["obv_slope_5"] = df["obv"].pct_change(5)

    df["dist_vwap_pct"] = close / df["vwap_session"] - 1
    df["close_gt_vwap"] = (close > df["vwap_session"]).astype(int)
    df["vwap_slope_5"] = df["vwap_session"].pct_change(5)

    df["breakout_20_flag"] = (close >= high.rolling(20, min_periods=20).max().shift(1)).astype(int)
    df["breakout_50_flag"] = (close >= high.rolling(50, min_periods=50).max().shift(1)).astype(int)

    df["higher_high_higher_low_3"] = ((high > high.shift(1)) & (high.shift(1) > high.shift(2)) & (low > low.shift(1)) & (low.shift(1) > low.shift(2))).astype(int)

    df["dollar_volume"] = close * volume
    df["log_dollar_volume"] = np.log1p(df["dollar_volume"])
    df["avg_dollar_volume_20"] = df["dollar_volume"].rolling(20, min_periods=20).mean()

    vol_long = df["realized_vol_12"].rolling(48, min_periods=24).mean()
    df["vol_regime_ratio"] = df["realized_vol_12"] / vol_long

    return df



def attach_market_and_sector_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spy = out.loc[out["symbol"] == "SPY", ["timestamp", "return_since_open", "ret_3"]].rename(
        columns={"return_since_open": "spy_return_since_open", "ret_3": "spy_ret_3"}
    )
    out = out.merge(spy, on="timestamp", how="left")
    out["spy_return_since_open"] = out["spy_return_since_open"].ffill()
    out["spy_ret_3"] = out["spy_ret_3"].ffill()
    out["rel_strength_vs_spy"] = out["return_since_open"] - out["spy_return_since_open"]

    sector_mean = (
        out.groupby(["timestamp", "sector"], dropna=False)["return_since_open"]
        .mean()
        .rename("sector_return_since_open")
        .reset_index()
    )
    out = out.merge(sector_mean, on=["timestamp", "sector"], how="left")
    out["rel_strength_vs_sector"] = out["return_since_open"] - out["sector_return_since_open"]
    return out



def build_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["symbol", "session_date", "timestamp"])
    future_high = (
        out.groupby(["symbol", "session_date"])["high"]
        .transform(lambda s: s.iloc[::-1].cummax().iloc[::-1].shift(-1))
    )
    out["future_high_to_close"] = future_high
    out["entry_price"] = out["close"]
    out["target_hit"] = (out["future_high_to_close"] >= out["entry_price"] * (1 + settings.target_pct)).astype(int)
    return out



def build_feature_dataset(bars: pd.DataFrame, constituents: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = bars.copy()
    if df.empty:
        raise RuntimeError("No bars available to build features.")

    meta = constituents[[c for c in constituents.columns if c in {"symbol", "security", "sector", "sub_industry"}]].copy()
    df = df.merge(meta, on="symbol", how="left")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    prev_day_close = (
        df.groupby(["symbol", "session_date"])["close"].last().groupby(level=0).shift(1).rename("prev_day_close").reset_index()
    )
    df = df.merge(prev_day_close, on=["symbol", "session_date"], how="left")

    session_enriched = (
        df.groupby(["symbol", "session_date"], group_keys=False).apply(_session_features).reset_index(drop=True)
    )
    symbol_enriched = session_enriched.groupby("symbol", group_keys=False).apply(_symbol_features).reset_index(drop=True)
    symbol_enriched["gap_from_prev_close_pct"] = symbol_enriched["session_open"] / symbol_enriched["prev_day_close"] - 1

    out = attach_market_and_sector_context(symbol_enriched)
    out = build_target(out)

    warmup_bars = 50
    max_bar_index = out.groupby(["symbol", "session_date"])["bar_index"].transform("max")
    out["bars_to_close"] = max_bar_index - out["bar_index"]
    out["eligible_observation"] = (
        (out["bar_index"] >= warmup_bars)
        & (out["bars_to_close"] >= settings.min_bars_to_close)
        & out[FEATURE_COLUMNS].notna().all(axis=1)
        & out["target_hit"].notna()
    )

    eligible = out.loc[out["eligible_observation"]].copy()
    eligible.replace([np.inf, -np.inf], np.nan, inplace=True)
    eligible = eligible.dropna(subset=FEATURE_COLUMNS + ["target_hit"])

    stats = {
        "total_rows": int(len(out)),
        "eligible_rows": int(len(eligible)),
        "symbols": int(out["symbol"].nunique()),
        "sessions": int(out[["symbol", "session_date"]].drop_duplicates().shape[0]),
    }
    return eligible, stats
