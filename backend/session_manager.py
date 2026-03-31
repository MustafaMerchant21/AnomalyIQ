"""
session_manager.py — AnomalyIQ Session I/O Helpers

Sessions MUST live in OS temp dir, outside backend/ to prevent Uvicorn --reload
from watching and restarting when .pkl/.json files are written.

Storage: Path(tempfile.gettempdir()) / "anomalyiq_sessions"
Atomic writes: write to .tmp, then os.replace() to prevent race conditions.
"""

import os
import json
import tempfile
import shutil
import time
from pathlib import Path
from typing import Any, Optional


SESSIONS_ROOT = Path(tempfile.gettempdir()) / "anomalyiq_sessions"


def get_session_dir(session_id: str) -> Path:
    """Return the directory for a given session."""
    return SESSIONS_ROOT / session_id


def ensure_session_dir(session_id: str) -> Path:
    """Create session directory if it doesn't exist, return path."""
    session_dir = get_session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_json(session_id: str, filename: str, data: Any) -> None:
    """Atomically save data as JSON to prevent half-written file reads."""
    session_dir = ensure_session_dir(session_id)
    target_path = session_dir / filename
    tmp_path = session_dir / (filename + ".tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=_json_serializer)
        os.replace(tmp_path, target_path)
    except Exception:
        # Clean up tmp on failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_json(session_id: str, filename: str) -> Optional[Any]:
    """Load JSON file from session directory. Returns None if not found."""
    session_dir = get_session_dir(session_id)
    target_path = session_dir / filename

    if not target_path.exists():
        return None

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_training_status(
    session_id: str,
    step: int,
    message: str,
    done: bool = False,
    error: Optional[str] = None
) -> None:
    """Write training progress status atomically."""
    status = {
        "step": step,
        "message": message,
        "done": done,
        "error": error,
        "timestamp": time.time()
    }
    save_json(session_id, "training_status.json", status)


def load_training_status(session_id: str) -> Optional[dict]:
    """Load training status for polling."""
    return load_json(session_id, "training_status.json")


def session_exists(session_id: str) -> bool:
    """Check if a session directory exists."""
    return get_session_dir(session_id).exists()


def get_session_file_path(session_id: str, filename: str) -> Path:
    """Return the full path to a file in the session directory."""
    return get_session_dir(session_id) / filename


def cleanup_old_sessions(max_sessions: int = 20) -> None:
    """
    Remove oldest sessions when count exceeds max_sessions.
    Called on FastAPI startup.
    """
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        sessions = [
            d for d in SESSIONS_ROOT.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        if len(sessions) <= max_sessions:
            return

        # Sort by modification time (oldest first)
        sessions.sort(key=lambda d: d.stat().st_mtime)

        # Remove oldest sessions beyond the limit
        sessions_to_remove = sessions[:len(sessions) - max_sessions]
        for session_dir in sessions_to_remove:
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except Exception:
                pass  # Best-effort cleanup

    except Exception:
        pass  # Never fail on cleanup


def load_session_context(session_id: str) -> dict:
    """
    Load the dataset context (name + domain) saved at upload time.
    Returns safe defaults if not found, so all LLM callers work even on old sessions.
    """
    ctx = load_json(session_id, "session_context.json") or {}
    return {
        "dataset_name": ctx.get("dataset_name", "Unnamed Dataset"),
        "domain_description": ctx.get("domain_description", "Anomaly detection"),
    }


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy/pandas types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
