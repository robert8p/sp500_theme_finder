from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = _env("APP_NAME", "S&P 500 Same-Day +1% Indicator Theme Finder")
    environment: str = _env("ENVIRONMENT", "development")
    host: str = _env("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)
    log_level: str = _env("LOG_LEVEL", "INFO")

    alpaca_api_key: str = _env("ALPACA_API_KEY", "")
    alpaca_secret_key: str = _env("ALPACA_SECRET_KEY", "")
    alpaca_base_url: str = _env("ALPACA_BASE_URL", "https://data.alpaca.markets")
    alpaca_timeout_seconds: int = _env_int("ALPACA_TIMEOUT_SECONDS", 30)

    data_interval: str = _env("DATA_INTERVAL", "5Min")
    lookback_months: int = _env_int("LOOKBACK_MONTHS", 6)
    target_pct: float = _env_float("TARGET_PCT", 0.01)
    min_theme_samples: int = _env_int("MIN_THEME_SAMPLES", 50)
    max_symbols: int = _env_int("MAX_SYMBOLS", 0)
    max_rule_conditions: int = _env_int("MAX_RULE_CONDITIONS", 14)
    min_bars_to_close: int = _env_int("MIN_BARS_TO_CLOSE", 3)
    max_rule_size: int = _env_int("MAX_RULE_SIZE", 4)

    data_dir: Path = Path(_env("DATA_DIR", "./data")).resolve()
    model_dir: Path = Path(_env("MODEL_DIR", "./data/models")).resolve()
    export_dir: Path = Path(_env("EXPORT_DIR", "./exports")).resolve()
    cache_dir: Path = Path(_env("CACHE_DIR", "./data/cache")).resolve()
    status_file: Path = Path(_env("STATUS_FILE", "./data/status.json")).resolve()

    admin_password: str = _env("ADMIN_PASSWORD", "")
    require_admin_password: bool = _env_bool("REQUIRE_ADMIN_PASSWORD", False)

    train_fraction: float = _env_float("TRAIN_FRACTION", 0.6)
    validation_fraction: float = _env_float("VALIDATION_FRACTION", 0.2)
    random_seed: int = _env_int("RANDOM_SEED", 42)

    use_cached_sp500: bool = _env_bool("USE_CACHED_SP500", True)
    use_cached_bars: bool = _env_bool("USE_CACHED_BARS", True)
    skip_download_if_exists: bool = _env_bool("SKIP_DOWNLOAD_IF_EXISTS", True)

    @property
    def bars_dir(self) -> Path:
        return self.data_dir / "bars"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.export_dir / "reports"

    @property
    def artifacts_dir(self) -> Path:
        return self.export_dir / "artifacts"


settings = Settings()

for path in [
    settings.data_dir,
    settings.model_dir,
    settings.export_dir,
    settings.cache_dir,
    settings.bars_dir,
    settings.processed_dir,
    settings.reports_dir,
    settings.artifacts_dir,
]:
    path.mkdir(parents=True, exist_ok=True)
