"""Locations of application data, bundled helpers and logs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import APP_NAME


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Folder containing AudioRefine.exe (frozen) or the repository root (development)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def bundle_dir() -> Path:
    """Folder with bundled data files (PyInstaller's _internal folder when frozen)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_dir()))
    return app_dir()


def data_dir() -> Path:
    """Per-user data (models, logs, settings, temporary work files).

    Portable mode: if a folder named ``data`` exists next to the executable, it is used.
    Otherwise %LOCALAPPDATA%\\AudioRefine on Windows (~/.local/share/AudioRefine elsewhere).
    ``AUDIOREFINE_DATA_DIR`` overrides both (used by tests).
    """
    override = os.environ.get("AUDIOREFINE_DATA_DIR")
    if override:
        p = Path(override)
    else:
        portable = app_dir() / "data"
        if portable.is_dir() and os.access(portable, os.W_OK):
            p = portable
        elif sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            p = Path(base) / APP_NAME
        else:
            base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
            p = Path(base) / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    override = os.environ.get("AUDIOREFINE_MODELS_DIR")
    p = Path(override) if override else data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def work_root() -> Path:
    p = data_dir() / "work"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gpu_runtime_dir() -> Path:
    p = data_dir() / "gpu-runtime"
    p.mkdir(parents=True, exist_ok=True)
    return p


def settings_file() -> Path:
    return data_dir() / "settings.json"


def arfilter_path() -> Path:
    """The bundled DSP helper (DeepFilterNet + filters)."""
    override = os.environ.get("AUDIOREFINE_ARFILTER")
    if override:
        return Path(override)
    exe = "arfilter.exe" if sys.platform == "win32" else "arfilter"
    candidates = [
        bundle_dir() / "bin" / exe,
        app_dir() / "bin" / exe,
        app_dir() / "native" / "arfilter" / "target" / "release" / exe,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]
