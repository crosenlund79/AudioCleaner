"""Diagnostic logging: one log file per engine run in the user's AudioRefine logs folder.

Logs contain technical information only (versions, hardware, file names, durations, decisions,
timings, warnings, errors). Transcribed text is never written to the log.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from .paths import logs_dir

KEEP = 40


def setup(command: str, verbose: bool = False) -> Path:
    d = logs_dir()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = d / f"AudioRefine_{stamp}_{command}.log"
    k = 2
    while path.exists():
        path = d / f"AudioRefine_{stamp}_{command}_{k}.log"
        k += 1
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)
    for noisy in ("faster_whisper", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _prune(d)
    return path


def _prune(d: Path) -> None:
    logs = sorted(d.glob("AudioRefine_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in logs[KEEP:]:
        try:
            p.unlink()
        except OSError:
            pass
