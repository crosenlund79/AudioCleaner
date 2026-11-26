"""Raw float32 sample files used as temporary working storage, plus the 48 kHz -> 16 kHz decimator.

Working files are headerless little-endian float32 mono. Reading uses ``np.fromfile`` with
offsets so no file handle stays open (important on Windows, where open handles block deletion).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np

from .constants import DECIMATION

F32 = np.dtype("<f4")


class RawWriter:
    """Append-only float32 file writer that counts samples."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._f = open(self.path, "wb")
        self.count = 0

    def write(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self._f.write(np.ascontiguousarray(x, dtype=F32).tobytes())
        self.count += int(x.size)

    def close(self) -> None:
        if not self._f.closed:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def raw_length(path: Path) -> int:
    return os.path.getsize(path) // 4


def read_raw(path: Path, start: int = 0, count: int | None = None) -> np.ndarray:
    total = raw_length(path)
    start = max(0, min(start, total))
    if count is None or start + count > total:
        count = total - start
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.fromfile(path, dtype=F32, count=count, offset=start * 4).astype(np.float32, copy=False)


def iter_raw(path: Path, block: int, start: int = 0, end: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    total = raw_length(path)
    end = total if end is None else min(end, total)
    pos = start
    while pos < end:
        n = min(block, end - pos)
        yield pos, read_raw(path, pos, n)
        pos += n


def _kaiser_lowpass(numtaps: int, cutoff: float, fs: float, beta: float) -> np.ndarray:
    n = np.arange(numtaps) - (numtaps - 1) / 2
    h = np.sinc(2 * cutoff / fs * n) * (2 * cutoff / fs)
    h *= np.kaiser(numtaps, beta)
    return h / h.sum()


class Decimator3:
    """Linear-phase 48 kHz -> 16 kHz decimator with exact, zero-delay alignment.

    Output sample i corresponds to input sample 3*i (the filter delay is compensated), and a
    stream of N input samples produces exactly ceil(N/3) output samples.
    Pass band to ~7.2 kHz, > 80 dB rejection above 8 kHz.
    """

    TAPS = 241  # odd; delay (TAPS-1)/2 = 120 = 3*40 samples

    def __init__(self):
        self.h_rev = _kaiser_lowpass(self.TAPS, 7600.0, 48000.0, 8.0)[::-1].astype(np.float64)
        self.delay = (self.TAPS - 1) // 2
        self.hist = np.zeros(self.TAPS - 1, dtype=np.float64)
        self.pos = 0       # number of input samples consumed
        self.next_i = 0    # next output index
        self.n_in = 0

    def _emit(self, x: np.ndarray) -> np.ndarray:
        L = self.TAPS
        ext = np.concatenate([self.hist, x.astype(np.float64)])
        # convolution output index m (global) uses window ext[j:j+L] with j = m - pos
        last_m = self.pos + len(x) - 1
        out = np.zeros(0, dtype=np.float32)
        m0 = DECIMATION * self.next_i + self.delay
        if m0 <= last_m and len(x) > 0:
            count = (last_m - m0) // DECIMATION + 1
            j0 = m0 - self.pos
            win = np.lib.stride_tricks.sliding_window_view(ext, L)[j0 : j0 + DECIMATION * count : DECIMATION]
            out = (win @ self.h_rev).astype(np.float32)
            self.next_i += count
        self.hist = ext[-(L - 1):]
        self.pos += len(x)
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        self.n_in += len(x)
        return self._emit(x)

    def flush(self) -> np.ndarray:
        total_out = -(-self.n_in // DECIMATION)
        out = self._emit(np.zeros(self.delay + DECIMATION, dtype=np.float64))
        keep = max(0, total_out - (self.next_i - len(out)))
        return out[:keep]
