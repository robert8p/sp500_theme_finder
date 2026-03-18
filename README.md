# S&P 500 Same-Day +1% Indicator Theme Finder

A FastAPI web application that:
- downloads 6 months of intraday bar data from Alpaca for the S&P 500,
- engineers a broad intraday technical-indicator feature set,
- constructs a same-day forward target with strict no-lookahead handling,
- trains multiple models using time-based splits only,
- mines and ranks human-readable indicator combinations (“themes”), and
- exports research artifacts for validation and review.

## Architecture summary

### Backend
- **FastAPI** serves the UI and JSON APIs.
- **Alpaca ingestion service** downloads and caches intraday bars to Parquet.
- **Feature pipeline** computes rolling technical indicators and session-aware context.
- **Analysis pipeline** performs time-split model training, validation, rule mining, and artifact export.
- **State store** tracks progress, logs, and latest-run metadata for the UI.

### Frontend
- Static HTML/CSS/JS dashboard served by FastAPI.
- Sections: Overview, Run Analysis, Theme Results, Indicator Explorer, Validation, Time of Day, False Positives, Downloads.

### Persistence
- Render persistent disk for cached bars and trained models.
- Exports written to `./exports`.

## Key methodology decisions

1. **No lookahead in features**  
   Every engineered feature is based on current and prior bars only.

2. **Strict same-day forward target**  
   For each eligible timestamp, the target is 1 when the future intraday high reaches `entry_price * (1 + TARGET_PCT)` before the session close.

3. **Time-based validation only**  
   The dataset is split by trading sessions into train / validation / test. There is no random shuffling.

4. **Theme discovery is explicit**  
   In addition to classifiers, the app mines readable multi-condition rules (for example: `price > VWAP + 9 EMA > 20 EMA + relative volume > 1.5`) and scores them out-of-sample.

5. **Robustness over vanity stats**  
   Theme ranking combines validation lift, test lift, stability, and support size.

## File tree

```text
sp500_theme_finder/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── state.py
│   └── services/
│       ├── __init__.py
│       ├── alpaca_client.py
│       ├── analysis.py
│       ├── features.py
│       ├── pipeline.py
│       ├── reports.py
│       ├── sp500.py
│       └── utils.py
├── static/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── data/
├── exports/
├── .env.example
├── README.md
├── render.yaml
└── requirements.txt
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Then open `http://localhost:8000`.

## Deployment on Render

1. Create a new web service from the repo.
2. Ensure Render uses `render.yaml`, or manually set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Attach a persistent disk.
4. Add environment variables:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
   - optionally `REQUIRE_ADMIN_PASSWORD=true` and `ADMIN_PASSWORD=...`
5. Deploy.

## API endpoints

- `GET /api/health`
- `GET /api/status`
- `GET /api/overview`
- `POST /api/run-analysis`
- `GET /api/themes`
- `GET /api/indicator-importance`
- `GET /api/validation`
- `GET /api/time-of-day`
- `GET /api/false-positives`
- `GET /api/bias-warnings`
- `GET /api/downloads`
- `GET /api/download/{artifact_name}`

## Leakage prevention

- Features are rolling or session-cumulative up to the observation bar.
- The target uses **future bars only** after the observation bar.
- Train/validation/test uses session chronology.
- Theme scoring is shown separately for validation and test.

## How themes are generated

1. Compute full feature set for each eligible stock-timestamp observation.
2. Train baseline and stronger models for predictive context.
3. Score raw feature importance.
4. Convert top candidate technical states into boolean conditions.
5. Enumerate multi-condition combinations.
6. Rank combinations by out-of-sample lift, support, and stability.

## Bias and limitations

- Current S&P 500 membership introduces survivorship bias.
- Alpaca intraday completeness and feed choice matter.
- Feature-mining risk remains even with time splits.
- 6 months is enough for discovery, not enough to prove durable alpha.
- Theme extraction is rule-based and pragmatic, not a formal causal inference engine.

## Sensible next improvements

- Add historical constituent reconstruction.
- Add sector ETF and macro-regime context.
- Add walk-forward retraining and threshold calibration.
- Add richer visual charts.
- Add asynchronous task queue for large runs.
- Add DB-backed job history and user authentication.
