from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ..config import settings
from .features import FEATURE_COLUMNS
from .utils import write_json, save_frame


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def time_split(df: pd.DataFrame) -> SplitData:
    dates = sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    if len(dates) < 15:
        raise RuntimeError("Not enough trading sessions after feature construction for a robust time split.")
    train_end = max(1, int(len(dates) * settings.train_fraction))
    val_end = max(train_end + 1, int(len(dates) * (settings.train_fraction + settings.validation_fraction)))
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])
    date_series = pd.to_datetime(df["session_date"]).dt.date
    train = df.loc[date_series.isin(train_dates)].copy()
    validation = df.loc[date_series.isin(val_dates)].copy()
    test = df.loc[date_series.isin(test_dates)].copy()
    return SplitData(train=train, validation=validation, test=test)



def feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df["target_hit"].astype(int).copy()



def _metrics(y_true: pd.Series, pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (pred_proba >= threshold).astype(int)
    return {
        "prevalence": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, pred_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y_true, pred_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, pred_proba)),
        "threshold": threshold,
        "predicted_positive_rate": float(np.mean(pred)),
    }



def train_models(split: SplitData) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Dict[str, float]]], pd.DataFrame]:
    X_train, y_train = feature_matrix(split.train)
    X_val, y_val = feature_matrix(split.validation)
    X_test, y_test = feature_matrix(split.test)

    models: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=settings.random_seed)),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, class_weight="balanced", random_state=settings.random_seed)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=3,
                        n_estimators=150,
                        random_state=settings.random_seed,
                    ),
                ),
            ]
        ),
    }

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    prediction_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        train_proba = model.predict_proba(X_train)[:, 1]
        metrics[name] = {
            "train": _metrics(y_train, train_proba),
            "validation": _metrics(y_val, val_proba),
            "test": _metrics(y_test, test_proba),
        }
        if name == "gradient_boosting":
            temp = split.test[["symbol", "session_date", "timestamp", "target_hit"]].copy()
            temp["prediction"] = test_proba
            prediction_rows.append(temp)

    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return models, metrics, predictions



def feature_importance(models: Dict[str, Pipeline], split: SplitData) -> pd.DataFrame:
    X_val, y_val = feature_matrix(split.validation)
    rows = []

    log_model = models["logistic_regression"]
    coef = log_model.named_steps["model"].coef_[0]
    for feature, value in zip(FEATURE_COLUMNS, coef):
        rows.append({"feature": feature, "source": "logistic_regression_abs_coef", "importance": abs(float(value))})

    tree_model = models["decision_tree"].named_steps["model"]
    for feature, value in zip(FEATURE_COLUMNS, tree_model.feature_importances_):
        rows.append({"feature": feature, "source": "decision_tree_importance", "importance": float(value)})

    gbdt_pipe = models["gradient_boosting"]
    perm = permutation_importance(gbdt_pipe, X_val, y_val, n_repeats=2, random_state=settings.random_seed, n_jobs=1)
    for feature, value in zip(FEATURE_COLUMNS, perm.importances_mean):
        rows.append({"feature": feature, "source": "permutation_validation", "importance": float(value)})

    imp = pd.DataFrame(rows)
    agg = imp.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    return agg.reset_index(drop=True)



def _condition_defs() -> Dict[str, Tuple[str, callable]]:
    return {
        "trend_above_vwap": ("price > VWAP", lambda d: d["close_gt_vwap"] == 1),
        "short_ma_up": ("9 EMA > 20 EMA", lambda d: d["ema9_gt_ema20"] == 1),
        "medium_ma_up": ("20 EMA > 50 EMA", lambda d: d["ema20_gt_ema50"] == 1),
        "full_ma_stack": ("9 EMA > 20 EMA > 50 EMA", lambda d: d["ma_stack_bullish"] == 1),
        "macd_positive": ("MACD histogram > 0", lambda d: d["macd_hist"] > 0),
        "rsi_bullish": ("RSI 55-70", lambda d: (d["rsi_14"] >= 55) & (d["rsi_14"] <= 70)),
        "rsi_oversold": ("RSI < 35", lambda d: d["rsi_14"] < 35),
        "stoch_oversold": ("stochastic < 20", lambda d: d["stoch_k_14"] < 20),
        "rv_high": ("relative volume > 1.5", lambda d: d["relative_volume_20"] > 1.5),
        "near_session_high": ("within 0.5% of intraday high", lambda d: d["dist_intraday_high_pct"] < 0.005),
        "positive_open_drive": ("return since open > 0", lambda d: d["return_since_open"] > 0),
        "positive_short_momentum": ("last 3-bar return > 0", lambda d: d["ret_3"] > 0),
        "strong_vs_spy": ("return since open > SPY", lambda d: d["rel_strength_vs_spy"] > 0),
        "strong_vs_sector": ("return since open > sector", lambda d: d["rel_strength_vs_sector"] > 0),
        "breakout_20": ("20-bar breakout", lambda d: d["breakout_20_flag"] == 1),
        "breakout_50": ("50-bar breakout", lambda d: d["breakout_50_flag"] == 1),
        "opening_range_break": ("above opening range high", lambda d: d["opening_range_break_flag"] == 1),
        "compression": ("range compression < 0.8x rolling", lambda d: d["compression_12"] < 0.8),
        "expansion": ("range expansion active", lambda d: d["range_expansion_flag"] == 1),
        "bullish_structure": ("higher highs and higher lows", lambda d: d["higher_high_higher_low_3"] == 1),
        "gap_up": ("gap from prior close > 0", lambda d: d["gap_from_prev_close_pct"] > 0),
        "low_pullback": ("pullback from high < 0.3%", lambda d: d["pullback_from_high_pct"] < 0.003),
        "vol_regime_hot": ("volatility regime ratio > 1.1", lambda d: d["vol_regime_ratio"] > 1.1),
    }



def _score_condition(mask: pd.Series, y: pd.Series) -> Dict[str, float]:
    support = int(mask.sum())
    if support == 0:
        return {"support": 0, "hit_rate": np.nan, "lift": np.nan}
    hit_rate = float(y.loc[mask].mean())
    baseline = float(y.mean())
    lift = hit_rate / baseline if baseline else np.nan
    return {"support": support, "hit_rate": hit_rate, "lift": lift}



def select_conditions(train: pd.DataFrame) -> List[str]:
    defs = _condition_defs()
    scores = []
    for key, (_, func) in defs.items():
        mask = func(train).fillna(False)
        scored = _score_condition(mask, train["target_hit"])
        scores.append({"condition_key": key, **scored})
    scored_df = pd.DataFrame(scores)
    scored_df = scored_df.loc[scored_df["support"] >= settings.min_theme_samples].copy()
    scored_df = scored_df.sort_values(["lift", "support"], ascending=[False, False])
    return scored_df["condition_key"].head(settings.max_rule_conditions).tolist()



def _evaluate_rule(df: pd.DataFrame, keys: Sequence[str]) -> Dict[str, float]:
    defs = _condition_defs()
    mask = pd.Series(True, index=df.index)
    descriptions = []
    for key in keys:
        desc, func = defs[key]
        mask &= func(df).fillna(False)
        descriptions.append(desc)
    support = int(mask.sum())
    if support == 0:
        return {
            "support": 0,
            "hit_rate": np.nan,
            "lift": np.nan,
            "precision": np.nan,
            "recall": 0.0,
            "condition_text": " + ".join(descriptions),
        }
    positives = int(df["target_hit"].sum())
    tp = int(df.loc[mask, "target_hit"].sum())
    hit_rate = tp / support
    baseline = float(df["target_hit"].mean())
    lift = hit_rate / baseline if baseline else np.nan
    recall = tp / positives if positives else 0.0
    return {
        "support": support,
        "hit_rate": hit_rate,
        "lift": lift,
        "precision": hit_rate,
        "recall": recall,
        "condition_text": " + ".join(descriptions),
    }



def _theme_name(keys: Sequence[str]) -> str:
    key_set = set(keys)
    if {"full_ma_stack", "trend_above_vwap", "macd_positive", "rv_high"}.issubset(key_set):
        return "Trend continuation with participation"
    if {"compression", "expansion", "breakout_20"}.issubset(key_set):
        return "Compression breakout"
    if {"rsi_oversold", "stoch_oversold", "strong_vs_spy"}.issubset(key_set):
        return "Washout reversal"
    if {"near_session_high", "positive_open_drive", "strong_vs_spy"}.issubset(key_set):
        return "Early leader pressing highs"
    if "opening_range_break" in key_set:
        return "Opening range continuation"
    if "breakout_50" in key_set:
        return "Range escape with strength"
    return " + ".join(k.replace("_", " ") for k in keys).title()



def discover_themes(split: SplitData) -> pd.DataFrame:
    chosen = select_conditions(split.train)
    if not chosen:
        return pd.DataFrame(columns=["theme_name", "conditions", "description"])

    results = []
    for size in range(2, settings.max_rule_size + 1):
        for keys in itertools.combinations(chosen, size):
            train_eval = _evaluate_rule(split.train, keys)
            if train_eval["support"] < settings.min_theme_samples:
                continue
            val_eval = _evaluate_rule(split.validation, keys)
            test_eval = _evaluate_rule(split.test, keys)
            if val_eval["support"] < max(10, settings.min_theme_samples // 5):
                continue
            val_test_lifts = [x for x in [val_eval["lift"], test_eval["lift"]] if pd.notna(x)]
            if not val_test_lifts:
                continue
            stability = 1.0 - abs((val_eval["hit_rate"] or 0) - (test_eval["hit_rate"] or 0))
            robustness = float(np.nanmean(val_test_lifts)) * stability * np.log1p(train_eval["support"])
            results.append(
                {
                    "theme_name": _theme_name(keys),
                    "condition_keys": list(keys),
                    "conditions": train_eval["condition_text"],
                    "description": train_eval["condition_text"],
                    "train_support": train_eval["support"],
                    "train_hit_rate": train_eval["hit_rate"],
                    "train_lift": train_eval["lift"],
                    "validation_support": val_eval["support"],
                    "validation_hit_rate": val_eval["hit_rate"],
                    "validation_lift": val_eval["lift"],
                    "test_support": test_eval["support"],
                    "test_hit_rate": test_eval["hit_rate"],
                    "test_lift": test_eval["lift"],
                    "precision": test_eval["precision"],
                    "recall": test_eval["recall"],
                    "stability_score": max(0.0, stability),
                    "robustness_score": robustness,
                    "robustness_notes": "Higher ranks require non-trivial support, out-of-sample lift, and stable hit-rates across validation and test.",
                }
            )

    if not results:
        return pd.DataFrame(columns=["theme_name", "conditions", "description"])
    out = pd.DataFrame(results).sort_values("robustness_score", ascending=False).drop_duplicates(subset=["conditions"]).head(30)
    return out.reset_index(drop=True)



def interaction_importance(themes: pd.DataFrame) -> pd.DataFrame:
    if themes.empty:
        return pd.DataFrame(columns=["interaction", "robustness_score", "test_lift"])
    rows = []
    for _, row in themes.iterrows():
        rows.append(
            {
                "interaction": row["conditions"],
                "robustness_score": float(row["robustness_score"]),
                "test_lift": float(row["test_lift"]),
            }
        )
    return pd.DataFrame(rows)



def time_of_day_analysis(df: pd.DataFrame, themes: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bucket", "observations", "hit_rate"])
    tmp = df.copy()
    tmp["bucket"] = (tmp["minutes_since_open"] // 30).astype(int) * 30
    base = (
        tmp.groupby("bucket")["target_hit"].agg(["count", "mean"]).reset_index().rename(columns={"count": "observations", "mean": "hit_rate"})
    )
    base["theme_name"] = "Overall baseline"
    frames = [base]
    defs = _condition_defs()
    for _, row in themes.head(5).iterrows():
        mask = pd.Series(True, index=tmp.index)
        for key in row["condition_keys"]:
            _, func = defs[key]
            mask &= func(tmp).fillna(False)
        filtered = tmp.loc[mask]
        if filtered.empty:
            continue
        grouped = (
            filtered.groupby("bucket")["target_hit"].agg(["count", "mean"]).reset_index().rename(columns={"count": "observations", "mean": "hit_rate"})
        )
        grouped["theme_name"] = row["theme_name"]
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)



def false_positive_analysis(split: SplitData, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["feature", "true_positive_mean", "false_positive_mean", "difference"])
    merged = split.test.merge(predictions[["timestamp", "symbol", "prediction"]], on=["timestamp", "symbol"], how="left")
    tagged = merged.loc[merged["prediction"] >= 0.6].copy()
    if tagged.empty:
        return pd.DataFrame(columns=["feature", "true_positive_mean", "false_positive_mean", "difference"])
    tagged["bucket"] = np.where(tagged["target_hit"] == 1, "true_positive", "false_positive")
    rows = []
    for feature in FEATURE_COLUMNS:
        tp_mean = tagged.loc[tagged["bucket"] == "true_positive", feature].mean()
        fp_mean = tagged.loc[tagged["bucket"] == "false_positive", feature].mean()
        rows.append(
            {
                "feature": feature,
                "true_positive_mean": float(tp_mean) if pd.notna(tp_mean) else np.nan,
                "false_positive_mean": float(fp_mean) if pd.notna(fp_mean) else np.nan,
                "difference": float(tp_mean - fp_mean) if pd.notna(tp_mean) and pd.notna(fp_mean) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("difference", key=lambda s: s.abs(), ascending=False).head(20)
    return out.reset_index(drop=True)



def save_artifacts(
    split: SplitData,
    models: Dict[str, Pipeline],
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    themes: pd.DataFrame,
    importances: pd.DataFrame,
    interactions: pd.DataFrame,
    tod: pd.DataFrame,
    false_positives: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Dict[str, str]:
    artifacts = {}

    save_frame(split.train, settings.processed_dir / "train")
    save_frame(split.validation, settings.processed_dir / "validation")
    save_frame(split.test, settings.processed_dir / "test")

    themes.to_csv(settings.artifacts_dir / "themes.csv", index=False)
    importances.to_csv(settings.artifacts_dir / "feature_importance.csv", index=False)
    interactions.to_csv(settings.artifacts_dir / "interaction_importance.csv", index=False)
    tod.to_csv(settings.artifacts_dir / "time_of_day.csv", index=False)
    false_positives.to_csv(settings.artifacts_dir / "false_positives.csv", index=False)
    predictions.to_csv(settings.artifacts_dir / "predictions.csv", index=False)
    write_json(settings.artifacts_dir / "model_metrics.json", metrics)

    for name, model in models.items():
        joblib.dump(model, settings.model_dir / f"{name}.joblib")

    artifacts.update(
        {
            "themes": str(settings.artifacts_dir / "themes.csv"),
            "feature_importance": str(settings.artifacts_dir / "feature_importance.csv"),
            "interaction_importance": str(settings.artifacts_dir / "interaction_importance.csv"),
            "time_of_day": str(settings.artifacts_dir / "time_of_day.csv"),
            "false_positives": str(settings.artifacts_dir / "false_positives.csv"),
            "predictions": str(settings.artifacts_dir / "predictions.csv"),
            "model_metrics": str(settings.artifacts_dir / "model_metrics.json"),
        }
    )
    return artifacts



def run_full_analysis(df: pd.DataFrame) -> Dict[str, object]:
    split = time_split(df)
    models, metrics, predictions = train_models(split)
    importances = feature_importance(models, split)
    themes = discover_themes(split)
    interactions = interaction_importance(themes)
    tod = time_of_day_analysis(split.test, themes)
    false_positives = false_positive_analysis(split, predictions)
    artifacts = save_artifacts(split, models, metrics, themes, importances, interactions, tod, false_positives, predictions)

    return {
        "split_sizes": {"train": int(len(split.train)), "validation": int(len(split.validation)), "test": int(len(split.test))},
        "metrics": metrics,
        "themes": themes.to_dict(orient="records"),
        "feature_importance": importances.to_dict(orient="records"),
        "interaction_importance": interactions.to_dict(orient="records"),
        "time_of_day": tod.to_dict(orient="records"),
        "false_positives": false_positives.to_dict(orient="records"),
        "artifacts": artifacts,
    }
