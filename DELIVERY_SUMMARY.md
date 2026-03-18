# Delivery Summary

## 1. Architecture summary

- **FastAPI backend** serves the dashboard and all analysis/export APIs.
- **Static frontend** provides Overview, Run Analysis, Theme Results, Indicator Explorer, Validation, Time of Day, False Positive Analysis, and Downloads.
- **Data ingestion layer** pulls intraday bars from Alpaca, caches by symbol, and reuses persisted files to avoid repeated downloads.
- **Feature engineering layer** computes trend, momentum, volatility, volume, price-structure, session-context, and relative-strength features.
- **Target-construction layer** computes same-day forward +1% hit labels without feature leakage.
- **Analysis layer** performs time-based train/validation/test splits, trains interpretable and stronger models, mines indicator combinations, and ranks themes by robustness.
- **Artifact layer** exports themes, feature importance, interaction importance, validation metrics, predictions, time-of-day analysis, false-positive analysis, and a markdown report.

## 2. Key methodology decisions

- Only **regular US session hours** are used.
- Features are built from **current and past bars only**.
- The target checks whether future intraday highs reach **entry price × 1.01** before the close.
- Validation uses **chronological session splits only**.
- Theme ranking emphasizes **support, stability, and out-of-sample lift** over cosmetic backtest performance.
- The default universe uses **current S&P 500 membership**, with survivorship bias explicitly disclosed.

## 3. Full project file tree

See `README.md` for the full file tree.

## 4. Complete code for every file

Included in this repository.

## 5. Deployment instructions

See `README.md` and `render.yaml`.

## 6. How to interpret discovered themes

A discovered theme is a **human-readable multi-condition setup** such as trend alignment plus above-VWAP plus strong relative volume. Each theme is shown with:
- train/validation/test support,
- hit-rate / precision,
- lift versus baseline,
- recall,
- stability score,
- robustness notes.

Use themes with strong **validation/test lift**, **non-trivial support**, and **stable hit-rates** first. Treat themes with weak support or unstable out-of-sample behaviour as exploratory only.

## 7. Known limitations and next improvements

- Current-member S&P 500 mode introduces survivorship bias.
- Multiple-testing risk remains in any feature-mining workflow.
- Six months is enough for discovery, not proof of durable live edge.
- Historical-constituent reconstruction, walk-forward retraining, and richer visualization are sensible next steps.
