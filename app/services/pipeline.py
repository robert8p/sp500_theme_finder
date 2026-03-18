from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..config import settings
from ..state import state_store
from .alpaca_client import AlpacaClient, default_date_range
from .analysis import run_full_analysis
from .features import build_feature_dataset
from .reports import write_report
from .sp500 import load_sp500_constituents
from .utils import write_json, save_frame, load_frame

LATEST_SUMMARY_PATH = settings.export_dir / "latest_summary.json"


class AnalysisPipeline:
    def __init__(self) -> None:
        self.client = AlpacaClient()

    def _symbol_path(self, symbol: str) -> Path:
        return settings.bars_dir / symbol

    def _download_symbol(self, symbol: str, start: datetime, end: datetime) -> Tuple[str, str, str]:
        path = self._symbol_path(symbol)
        if settings.use_cached_bars and settings.skip_download_if_exists:
            if path.with_suffix(".parquet").exists() or path.with_suffix(".csv").exists():
                existing = path.with_suffix(".parquet") if path.with_suffix(".parquet").exists() else path.with_suffix(".csv")
                return symbol, "cached", str(existing)
        df = self.client.fetch_bars(symbol, start, end)
        if df.empty:
            return symbol, "empty", ""
        saved = save_frame(df, path)
        return symbol, "downloaded", str(saved)

    def download_bars(self, symbols: List[str]) -> Dict[str, object]:
        start, end = default_date_range()
        results = {"downloaded": [], "cached": [], "empty": [], "failed": []}
        state_store.update(phase="download", message=f"Fetching {len(symbols)} symbols from Alpaca", progress=0.1)
        total = len(symbols)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self._download_symbol, symbol, start, end): symbol for symbol in symbols}
            for idx, future in enumerate(as_completed(futures), start=1):
                symbol = futures[future]
                try:
                    _, status, detail = future.result()
                    results[status].append({"symbol": symbol, "detail": detail})
                except Exception as exc:
                    results["failed"].append({"symbol": symbol, "detail": str(exc)})
                if idx % 10 == 0 or idx == total:
                    state_store.update(
                        phase="download",
                        message=f"Processed {idx}/{total} symbol downloads",
                        progress=0.1 + 0.45 * (idx / total),
                    )
        return results

    def load_bars(self, symbols: List[str]) -> pd.DataFrame:
        frames = []
        for symbol in symbols:
            path = self._symbol_path(symbol)
            if path.with_suffix(".parquet").exists() or path.with_suffix(".csv").exists():
                frames.append(load_frame(path))
        if not frames:
            raise RuntimeError("No cached bar files were available after the download phase.")
        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if str(df["timestamp"].dt.tz) == "UTC":
            df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return df

    def run(self) -> Dict[str, object]:
        constituents = load_sp500_constituents()
        symbols = constituents["symbol"].tolist()
        if settings.max_symbols and settings.max_symbols > 0:
            symbols = symbols[: settings.max_symbols]
            constituents = constituents.loc[constituents["symbol"].isin(symbols)].copy()
        if "SPY" not in symbols:
            symbols = symbols + ["SPY"]

        download_status = self.download_bars(symbols)
        state_store.update(phase="preprocess", message="Loading and engineering features", progress=0.6)
        bars = self.load_bars(symbols)
        eligible, build_stats = build_feature_dataset(bars, constituents)
        save_frame(eligible, settings.processed_dir / "eligible_dataset")

        state_store.update(phase="analysis", message="Running model training, validation, and theme discovery", progress=0.75)
        summary = run_full_analysis(eligible)
        summary["data_build_stats"] = build_stats
        summary["download_status"] = {k: len(v) for k, v in download_status.items()}
        summary["bias_warnings"] = [
            "Current S&P 500 membership is used by default, which introduces survivorship bias.",
            "This is a retrospective association engine, not proof of deployable real-time edge.",
            "Multiple-testing / feature-mining risk remains even with time-based validation.",
            "Regime shifts can break historically robust intraday themes.",
        ]
        report_path = write_report(summary)
        summary["report_path"] = str(report_path)
        write_json(LATEST_SUMMARY_PATH, summary)
        return summary



def run_pipeline_job() -> None:
    state_store.start("Starting full pipeline")
    try:
        pipeline = AnalysisPipeline()
        summary = pipeline.run()
        state_store.finish(
            {
                "themes_found": len(summary.get("themes", [])),
                "eligible_rows": summary.get("data_build_stats", {}).get("eligible_rows", 0),
                "report_path": summary.get("report_path"),
            }
        )
    except Exception as exc:
        state_store.fail(f"Pipeline failed: {exc}")
        traceback.print_exc()
