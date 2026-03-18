from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from ..config import settings


def write_report(summary: Dict) -> Path:
    report_path = settings.reports_dir / "latest_report.md"
    generated_at = datetime.now(timezone.utc).isoformat()
    themes = summary.get("themes", [])[:10]
    metrics = summary.get("metrics", {})
    importance = summary.get("feature_importance", [])[:15]

    lines = [
        f"# {settings.app_name}",
        "",
        f"Generated at: {generated_at}",
        "",
        "## Scope and warnings",
        "- Current S&P 500 membership is used by default. This introduces survivorship bias unless historical constituent reconstruction is added.",
        "- This workflow is retrospective and designed for research, not live deployment.",
        "- All features are intended to use data available only at the observation timestamp, but external data quality still matters.",
        "- Non-stationarity and regime shifts can invalidate historically strong themes.",
        "",
        "## Top themes",
    ]

    if themes:
        for idx, theme in enumerate(themes, start=1):
            lines.extend(
                [
                    f"### {idx}. {theme.get('theme_name', 'Unnamed theme')}",
                    f"- Conditions: {theme.get('conditions', '')}",
                    f"- Validation lift: {theme.get('validation_lift', float('nan')):.3f}",
                    f"- Test lift: {theme.get('test_lift', float('nan')):.3f}",
                    f"- Test precision / hit-rate: {theme.get('precision', float('nan')):.3f}",
                    f"- Stability score: {theme.get('stability_score', float('nan')):.3f}",
                    f"- Train / validation / test support: {theme.get('train_support', 0)} / {theme.get('validation_support', 0)} / {theme.get('test_support', 0)}",
                    "",
                ]
            )
    else:
        lines.extend(["No robust themes passed the minimum support and out-of-sample filters.", ""])

    lines.extend(["## Model validation", ""])
    for model_name, splits in metrics.items():
        test = splits.get("test", {})
        lines.extend(
            [
                f"### {model_name}",
                f"- Test ROC AUC: {test.get('roc_auc', float('nan')):.3f}",
                f"- Test average precision: {test.get('average_precision', float('nan')):.3f}",
                f"- Test precision: {test.get('precision', float('nan')):.3f}",
                f"- Test recall: {test.get('recall', float('nan')):.3f}",
                f"- Test Brier score: {test.get('brier', float('nan')):.3f}",
                "",
            ]
        )

    lines.extend(["## Top indicator importance signals", ""])
    for row in importance:
        lines.append(f"- {row.get('feature')}: {row.get('importance', float('nan')):.6f}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
