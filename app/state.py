from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunState:
    is_running: bool = False
    phase: str = "idle"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: str = ""
    progress: float = 0.0
    last_error: Optional[str] = None
    log_lines: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "phase": self.phase,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
            "progress": self.progress,
            "last_error": self.last_error,
            "log_lines": self.log_lines[-250:],
            "summary": self.summary,
        }


class StateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = RunState()
        self._persist()

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def reset(self) -> None:
        with self._lock:
            self._state = RunState()
            self._persist()

    def start(self, message: str = "Starting analysis") -> None:
        with self._lock:
            self._state.is_running = True
            self._state.phase = "starting"
            self._state.started_at = utc_now_iso()
            self._state.finished_at = None
            self._state.message = message
            self._state.progress = 0.0
            self._state.last_error = None
            self._state.log_lines = [f"[{utc_now_iso()}] {message}"]
            self._state.summary = {}
            self._persist()

    def update(self, *, phase=None, message=None, progress=None, summary_patch=None) -> None:
        with self._lock:
            if phase is not None:
                self._state.phase = phase
            if message is not None:
                self._state.message = message
                self._state.log_lines.append(f"[{utc_now_iso()}] {message}")
            if progress is not None:
                self._state.progress = progress
            if summary_patch:
                self._state.summary.update(summary_patch)
            self._persist()

    def fail(self, error: str) -> None:
        with self._lock:
            self._state.is_running = False
            self._state.phase = "failed"
            self._state.finished_at = utc_now_iso()
            self._state.last_error = error
            self._state.message = error
            self._state.log_lines.append(f"[{utc_now_iso()}] ERROR: {error}")
            self._persist()

    def finish(self, summary=None) -> None:
        with self._lock:
            self._state.is_running = False
            self._state.phase = "completed"
            self._state.finished_at = utc_now_iso()
            self._state.message = "Analysis complete"
            self._state.progress = 1.0
            if summary:
                self._state.summary.update(summary)
            self._state.log_lines.append(f"[{utc_now_iso()}] Analysis complete")
            self._persist()

    def _persist(self) -> None:
        settings.status_file.parent.mkdir(parents=True, exist_ok=True)
        settings.status_file.write_text(json.dumps(self._state.to_dict(), indent=2), encoding="utf-8")


state_store = StateStore()
