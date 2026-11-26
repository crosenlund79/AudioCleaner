"""Voice activity detection with Silero VAD v6 (MIT), streamed over arbitrarily long audio.

The ONNX model and the windowing scheme are the ones shipped with faster-whisper. The model
output is kept as a probability per 32 ms window for the whole recording; speech regions are
derived from it with hysteresis. Probabilities are also reused by the loudness leveller.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .constants import ASR_SR
from .rawaudio import iter_raw, raw_length, read_raw

WINDOW = 512                 # samples at 16 kHz (32 ms)
CONTEXT = 64
BATCH = 10000                # windows per ONNX call (same as faster-whisper)
WINDOW_S = WINDOW / ASR_SR


def _model_path() -> str:
    from faster_whisper.utils import get_assets_path

    return os.path.join(get_assets_path(), "silero_vad_v6.onnx")


class StreamingSileroVAD:
    def __init__(self):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(_model_path(), providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), dtype="float32")
        self.c = np.zeros((1, 1, 128), dtype="float32")
        self.ctx = np.zeros(CONTEXT, dtype="float32")

    def windows(self, audio: np.ndarray) -> np.ndarray:
        """Probabilities for consecutive 512-sample windows (len(audio) must be a multiple of 512)."""
        assert audio.ndim == 1 and audio.shape[0] % WINDOW == 0
        if audio.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        wins = audio.reshape(-1, WINDOW).astype(np.float32)
        ctx = np.empty((wins.shape[0], CONTEXT), dtype=np.float32)
        ctx[0] = self.ctx
        ctx[1:] = wins[:-1, -CONTEXT:]
        batched = np.concatenate([ctx, wins], axis=1)
        outs = []
        for i in range(0, batched.shape[0], BATCH):
            out, self.h, self.c = self.session.run(None, {"input": batched[i:i + BATCH], "h": self.h, "c": self.c})
            outs.append(out)
        self.ctx = wins[-1, -CONTEXT:].copy()
        return np.concatenate(outs, axis=0).reshape(-1).astype(np.float32)


# Level normalisation of the VAD input (detection only; audio outputs are not affected).
# Silero misses very quiet speech (measured: an utterance at -65 dBFS was only half detected),
# so the signal it sees is brought towards -20 dBFS with a slow gain of at most +40 dB -- but
# only where the signal stands clearly above the local noise floor. Boosting a stretch of plain
# room noise would make Silero call it speech (observed in CI: Whisper then hallucinated).
VAD_TARGET_DB = -20.0
VAD_MAX_GAIN_DB = 40.0
VAD_MIN_CONTRAST_DB = 12.0       # envelope must exceed the local noise floor by this much
_ENV_FRAME = 480                 # 30 ms
_ENV_SPAN = 33                   # ~1 s moving maximum of frame levels
_FLOOR_SPAN = 333                # ~10 s window for the noise floor (moving minimum)


def _moving(fn, x: np.ndarray, r: int) -> np.ndarray:
    pad = np.pad(x, (r, r), mode="edge")
    return fn(np.lib.stride_tricks.sliding_window_view(pad, 2 * r + 1), axis=1)


def _level_gain(x_ext: np.ndarray, ctx_before: int, n: int) -> np.ndarray:
    """Per-sample gain for x_ext[ctx_before:ctx_before+n]."""
    m = len(x_ext) // _ENV_FRAME
    if m == 0:
        return np.ones(n, dtype=np.float32)
    fr = x_ext[: m * _ENV_FRAME].reshape(m, _ENV_FRAME).astype(np.float64)
    lv = 10 * np.log10(np.mean(fr * fr, axis=1) + 1e-12)
    env = _moving(np.max, lv, _ENV_SPAN // 2)
    smooth = np.convolve(np.pad(lv, 5, mode="edge"), np.ones(11) / 11, mode="valid")
    floor = _moving(np.min, smooth, _FLOOR_SPAN // 2)
    g_db = np.clip(VAD_TARGET_DB - env, 0.0, VAD_MAX_GAIN_DB)
    contrast = env - floor
    g_db = g_db * np.clip((contrast - VAD_MIN_CONTRAST_DB) / 6.0, 0.0, 1.0)   # fade in between 12 and 18 dB
    g_db = np.convolve(np.pad(g_db, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    centers = (np.arange(m) + 0.5) * _ENV_FRAME
    idx = np.arange(ctx_before, ctx_before + n, dtype=np.float64)
    return (10 ** (np.interp(idx, centers, g_db) / 20)).astype(np.float32)


def speech_probabilities(asr_raw: Path, progress: Callable[[float], None] | None = None,
                         cancel: Callable[[], None] | None = None, level_normalise: bool = True) -> np.ndarray:
    """Silero probability for every 32 ms window of a 16 kHz raw file."""
    vad = StreamingSileroVAD()
    total = raw_length(asr_raw)
    block = WINDOW * 2000  # 64 s
    ctx = ASR_SR * 6
    probs = []
    for pos, x in iter_raw(asr_raw, block):
        if cancel:
            cancel()
        if level_normalise:
            a = max(0, pos - ctx)
            ext = read_raw(asr_raw, a, (pos + len(x) + ctx) - a)
            x = x * _level_gain(ext, pos - a, len(x))
        if len(x) % WINDOW:
            x = np.pad(x, (0, WINDOW - len(x) % WINDOW))
        probs.append(vad.windows(np.clip(x, -1.0, 1.0)))
        if progress:
            progress(min(1.0, (pos + len(x)) / max(total, 1)))
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)


@dataclass(frozen=True)
class VadParams:
    threshold: float = 0.35
    neg_threshold: float = 0.2
    min_speech_s: float = 0.09
    min_silence_s: float = 1.0
    pad_s: float = 0.4


def speech_regions(probs: np.ndarray, total_samples: int, p: VadParams = VadParams()) -> list[tuple[int, int]]:
    """Speech regions in 16 kHz samples, using hysteresis (adapted from Silero/faster-whisper)."""
    min_speech = p.min_speech_s * ASR_SR
    min_silence = p.min_silence_s * ASR_SR
    pad = int(p.pad_s * ASR_SR)
    regions: list[list[int]] = []
    triggered = False
    start = 0
    temp_end = 0
    for i, pr in enumerate(probs):
        pos = i * WINDOW
        if pr >= p.threshold:
            temp_end = 0
            if not triggered:
                triggered = True
                start = pos
            continue
        if triggered and pr < p.neg_threshold:
            if not temp_end:
                temp_end = pos
            if pos - temp_end >= min_silence:
                if temp_end - start >= min_speech:
                    regions.append([start, temp_end])
                triggered = False
                temp_end = 0
    if triggered:
        end = temp_end if temp_end else min(len(probs) * WINDOW, total_samples)
        if end - start >= min_speech:
            regions.append([start, end])
    # padding, splitting the gap between close regions
    for i, r in enumerate(regions):
        if i == 0:
            r[0] = max(0, r[0] - pad)
        if i < len(regions) - 1:
            gap = regions[i + 1][0] - r[1]
            if gap < 2 * pad:
                r[1] += gap // 2
                regions[i + 1][0] -= gap // 2
            else:
                r[1] += pad
                regions[i + 1][0] -= pad
        else:
            r[1] = min(total_samples, r[1] + pad)
    return [(int(a), int(min(b, total_samples))) for a, b in regions if b > a]


def probs_at_frames(probs: np.ndarray, n_frames: int, frame_s: float) -> np.ndarray:
    """Resample window probabilities onto another frame grid (nearest window)."""
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)
    if probs.size == 0:
        return np.zeros(n_frames, dtype=np.float32)
    centers = (np.arange(n_frames) + 0.5) * frame_s
    idx = np.minimum((centers / WINDOW_S).astype(np.int64), probs.size - 1)
    return probs[idx]
