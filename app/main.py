from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .state import state_store
from .services.pipeline import LATEST_SUMMARY_PATH, run_pipeline_job
from .services.utils import read_json

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")



def _check_admin(x_admin_password: str | None) -> None:
    if settings.require_admin_password:
        if not x_admin_password or x_admin_password != settings.admin_password:
            raise HTTPException(status_code=401, detail="Invalid admin password")



def _summary() -> Dict[str, Any]:
    return read_json(LATEST_SUMMARY_PATH, default={}) or {}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "app": settings.app_name}


@app.get("/api/status")
def status() -> Dict[str, Any]:
    return state_store.get()


@app.get("/api/overview")
def overview() -> Dict[str, Any]:
    summary = _summary()
    return {
        "app_name": settings.app_name,
        "data_interval": settings.data_interval,
        "lookback_months": settings.lookback_months,
        "target_pct": settings.target_pct,
        "min_theme_samples": settings.min_theme_samples,
        "summary": summary,
        "status": state_store.get(),
    }


@app.post("/api/run-analysis")
def run_analysis(x_admin_password: str | None = Header(default=None)) -> Dict[str, Any]:
    _check_admin(x_admin_password)
    state = state_store.get()
    if state["is_running"]:
        raise HTTPException(status_code=409, detail="Analysis is already running")
    thread = threading.Thread(target=run_pipeline_job, daemon=True)
    thread.start()
    return {"accepted": True, "message": "Analysis started"}


@app.get("/api/themes")
def themes() -> Dict[str, Any]:
    summary = _summary()
    return {"themes": summary.get("themes", [])}


@app.get("/api/indicator-importance")
def indicator_importance() -> Dict[str, Any]:
    summary = _summary()
    return {
        "feature_importance": summary.get("feature_importance", []),
        "interaction_importance": summary.get("interaction_importance", []),
    }


@app.get("/api/validation")
def validation() -> Dict[str, Any]:
    summary = _summary()
    return {"metrics": summary.get("metrics", {}), "split_sizes": summary.get("split_sizes", {})}


@app.get("/api/time-of-day")
def time_of_day() -> Dict[str, Any]:
    summary = _summary()
    return {"rows": summary.get("time_of_day", [])}


@app.get("/api/false-positives")
def false_positives() -> Dict[str, Any]:
    summary = _summary()
    return {"rows": summary.get("false_positives", [])}


@app.get("/api/bias-warnings")
def bias_warnings() -> Dict[str, Any]:
    summary = _summary()
    return {"warnings": summary.get("bias_warnings", [])}


@app.get("/api/downloads")
def downloads() -> Dict[str, Any]:
    summary = _summary()
    return {
        "artifacts": summary.get("artifacts", {}),
        "report_path": summary.get("report_path"),
    }


@app.get("/api/download/{artifact_name}")
def download_artifact(artifact_name: str) -> FileResponse:
    summary = _summary()
    mapping = summary.get("artifacts", {})
    if artifact_name == "report":
        path = summary.get("report_path")
    else:
        path = mapping.get(artifact_name)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=Path(path).name)
