"""Progress reporting with weighted stages, elapsed time and a simple remaining-time estimate."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


class CancelledError(Exception):
    """Raised inside the pipeline when the user cancels."""


class CancelToken:
    def __init__(self):
        self._ev = threading.Event()

    def cancel(self):
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def check(self):
        if self._ev.is_set():
            raise CancelledError()


@dataclass
class Stage:
    key: str
    label: str
    weight: float


STAGE_LABELS = {
    "prepare": "Checking recordings",
    "model": "Loading speech recognition",
    "decode": "Decoding audio",
    "vad": "Detecting speech",
    "analyse": "Analysing recordings",
    "enhance": "Enhancing speech",
    "transcribe": "Transcribing speech",
    "write_audio": "Writing audio files",
    "write_text": "Writing transcripts",
    "validate": "Validating results",
    "archive": "Archiving original recordings",
}


def default_weights(gpu: bool) -> dict[str, float]:
    return {
        "prepare": 1, "model": 2, "decode": 3, "vad": 2, "analyse": 1,
        "enhance": 25 if gpu else 10,
        "transcribe": 40 if gpu else 75,
        "write_audio": 3, "write_text": 1, "validate": 2, "archive": 1,
    }


class Progress:
    def __init__(self, emit: Callable[[dict], None], weights: dict[str, float] | None = None):
        self.emit = emit
        self.weights = weights or default_weights(False)
        self.order = list(self.weights)
        self.stage = None
        self.stage_frac = 0.0
        self.t0 = time.monotonic()
        self._last = 0.0
        self._lock = threading.Lock()
        self.detail = ""

    def set_weights(self, weights: dict[str, float]):
        with self._lock:
            self.weights = weights
            self.order = list(weights)

    def overall(self) -> float:
        tot = sum(self.weights.values())
        done = 0.0
        for k in self.order:
            if k == self.stage:
                done += self.weights[k] * self.stage_frac
                break
            done += self.weights[k]
        return min(1.0, done / tot) if tot else 0.0

    def start(self, key: str, detail: str = ""):
        with self._lock:
            self.stage = key
            self.stage_frac = 0.0
            self.detail = detail
        self._send(force=True)

    def update(self, frac: float, detail: str | None = None):
        with self._lock:
            self.stage_frac = max(0.0, min(1.0, frac))
            if detail is not None:
                self.detail = detail
        self._send()

    def _send(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last < 0.25:
            return
        self._last = now
        elapsed = now - self.t0
        p = self.overall()
        eta = None
        if p > 0.04 and elapsed > 20:
            eta = elapsed * (1 - p) / p
        self.emit({
            "type": "progress", "stage": self.stage, "label": STAGE_LABELS.get(self.stage, self.stage or ""),
            "detail": self.detail, "stage_fraction": round(self.stage_frac, 4), "fraction": round(p, 4),
            "elapsed": round(elapsed, 1), "eta": round(eta, 0) if eta is not None else None,
        })
