"""The sound event classifier and its audio front-end.

Model: EfficientAT ``mn10_as`` (MobileNetV3, width 1.0, 4.9 M parameters) trained on AudioSet
(527 sound classes, single-model mAP 0.471), by Schmid, Koutini and Widmer (MIT licence),
exported once to ONNX (``tools/sound/export_onnx.py``) and run here with ONNX Runtime on the CPU.
The model ships inside the application, so nothing is downloaded and analysis works offline.

Front-end (identical to the model's training front-end, verified against the PyTorch code to
< 5e-5): 32 kHz mono, pre-emphasis 0.97, 1024-point FFT of a 25 ms Hann window every 10 ms,
power spectrum, 128 Kaldi-style mel bands (0–15 kHz), ``(log(mel + 1e-5) + 4.5) / 5``.

The recording is analysed at two time scales (see ACOUSTIC_ANALYSIS.md for the measurements):

* **context** – 10-second windows of natural audio every 5 s: the model's training condition and
  the most accurate way to tell *whether* a sound is present (and the only way to recognise
  television or video-game audio);
* **localisation** – 2-second windows every second, each centred in 10 s of silence (the model
  saturates on inputs shorter than about 4 s, so short windows are padded): *where* it is.

Both give one score per AudioSet class (sigmoid outputs, *not* calibrated probabilities).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..paths import bundle_dir

log = logging.getLogger("audiorefine.sound")

SR = 32000
N_FFT, HOP, WIN = 1024, 320, 800
FPS = SR // HOP                     # 100 mel frames per second
WIN_S = 2.0                          # analysis window (audio content)
HOP_S = 1.0                          # window step
PAD_FRAMES = 1000                    # each window is centred in 10 s of silence (model input length)
CTX_S = 10.0                         # context window (natural audio)
CTX_HOP_S = 5.0
CTX_FRAMES = int(round(CTX_S * FPS))
CTX_HOP_FRAMES = int(round(CTX_HOP_S * FPS))
WIN_FRAMES = int(round(WIN_S * FPS))
HOP_FRAMES = int(round(HOP_S * FPS))
BATCH = 8
SILENT_MEL = float((np.log(1e-5) + 4.5) / 5.0)   # log-mel value of digital silence

MODEL_ID = "efficientat-mn10_as-onnx"
MODEL_FILE = "efficientat_mn10_as.onnx"
MODEL_SHA256 = "7c02401e65a500748fb5172cd3fc3bf654a926fa78cf7a63a7315938731cb43a"
MEL_FILE = "mel_basis_128x513.npy"
LABELS_FILE = "audioset_labels.json"


def asset_dir() -> Path:
    override = os.environ.get("AUDIOREFINE_SOUND_ASSETS")
    if override:
        return Path(override)
    for c in (bundle_dir() / "assets" / "sound", Path(__file__).resolve().parents[3] / "assets" / "sound"):
        if (c / MODEL_FILE).is_file():
            return c
    return bundle_dir() / "assets" / "sound"


@lru_cache(maxsize=1)
def labels() -> list[str]:
    return [d["name"] for d in json.loads((asset_dir() / LABELS_FILE).read_text(encoding="utf-8"))]


def _window() -> np.ndarray:
    w = np.zeros(N_FFT, np.float32)
    a = (N_FFT - WIN) // 2
    w[a:a + WIN] = np.hanning(WIN).astype(np.float32)       # symmetric Hann, centred in the FFT frame
    return w


class Frontend:
    """Log-mel front-end. ``chunk`` computes the frames of a long signal piece by piece and gives
    exactly the frames the whole-signal computation would give."""

    def __init__(self):
        self.mel = np.load(asset_dir() / MEL_FILE).astype(np.float32)   # [128, 513]
        self.win = _window()

    def logmel(self, x: np.ndarray) -> np.ndarray:
        """Whole signal (reference implementation of the model's front-end)."""
        x = np.asarray(x, np.float32)
        if len(x) < 2:
            x = np.pad(x, (0, 2 - len(x)))
        y = x[1:] - 0.97 * x[:-1]
        if len(y) <= N_FFT // 2:
            y = np.pad(y, (0, N_FFT // 2 + 1 - len(y)))
        y = np.pad(y, (N_FFT // 2, N_FFT // 2), mode="reflect")
        return self._frames(y, 1 + (len(y) - N_FFT) // HOP)

    def _frames(self, ypad: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((self.mel.shape[0], 0), np.float32)
        ypad = np.ascontiguousarray(ypad, dtype=np.float32)
        fr = np.lib.stride_tricks.as_strided(ypad, (n, N_FFT), (ypad.strides[0] * HOP, ypad.strides[0]))
        spec = np.fft.rfft(fr * self.win, axis=1)
        p = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
        return ((np.log(self.mel @ p.T + 1e-5) + 4.5) / 5.0).astype(np.float32)


class StreamingMel:
    """Feed audio blocks, get mel frames; equal to ``Frontend.logmel`` of the concatenated signal.

    Frame k is centred on pre-emphasised sample k*HOP and uses samples [k*HOP-512, k*HOP+512),
    reflected at the start and (on the final call) at the end, exactly like the reference."""

    def __init__(self, fe: Frontend):
        self.fe = fe
        self.prev_x: float | None = None     # last raw sample (pre-emphasis continuity)
        self.y = np.zeros(0, np.float32)     # pre-emphasised samples kept
        self.y0 = 0                          # index of self.y[0] in the pre-emphasised signal
        self.next_frame = 0

    def push(self, x: np.ndarray, final: bool = False) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if len(x):
            xx = x if self.prev_x is None else np.concatenate([np.float32([self.prev_x]), x])
            self.y = np.concatenate([self.y, xx[1:] - 0.97 * xx[:-1]]).astype(np.float32)
            self.prev_x = float(x[-1])
        half = N_FFT // 2
        L = self.y0 + len(self.y)
        if final:
            if L <= half:                      # very short signal: whole-signal reference
                if self.y0 != 0:
                    return np.zeros((self.fe.mel.shape[0], 0), np.float32)
                ypad = np.pad(np.pad(self.y, (0, half + 1 - len(self.y))), (half, half), mode="reflect")
                n = 1 + (len(ypad) - N_FFT) // HOP
                fr = self.fe._frames(ypad, n)[:, self.next_frame:]
                self.next_frame = n
                return fr
            last = L // HOP
        else:
            last = (L - half) // HOP if L > half else -1
        if last < self.next_frame:
            return np.zeros((self.fe.mel.shape[0], 0), np.float32)
        idx = np.arange(self.next_frame * HOP - half, last * HOP + half)
        idx = np.where(idx < 0, -idx, idx)
        if final:
            idx = np.where(idx >= L, 2 * (L - 1) - idx, idx)
        seg = self.y[idx - self.y0]
        n = last - self.next_frame + 1
        frames = self.fe._frames(seg, n)
        self.next_frame = last + 1
        keep_from = max(0, self.next_frame * HOP - half - HOP - 4 - self.y0)
        if self.next_frame * HOP - half - HOP - 4 <= half + 2:
            keep_from = 0                      # keep the start for the left reflection
        self.y = self.y[keep_from:]
        self.y0 += keep_from
        return frames


@lru_cache(maxsize=1)
def model_file_ok() -> bool:
    p = asset_dir() / MODEL_FILE
    if not p.is_file():
        return False
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest() == MODEL_SHA256


class SoundModel:
    def __init__(self, threads: int | None = None):
        import onnxruntime as ort

        p = asset_dir() / MODEL_FILE
        if not model_file_ok():
            raise RuntimeError(f"The sound model file is missing or damaged: {p}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, min(4, (os.cpu_count() or 2)))
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        self.frontend = Frontend()

    def context_scores(self, mel: np.ndarray) -> tuple[np.ndarray, list[float]]:
        """Whole mel array [128, T] -> scores of 10 s windows every 5 s (last one aligned to the end;
        a recording shorter than 10 s is one window, padded with silence) and their start times."""
        T = mel.shape[1]
        starts = list(range(0, max(1, T - CTX_FRAMES + 1), CTX_HOP_FRAMES))
        if T > CTX_FRAMES and starts[-1] + CTX_FRAMES < T:
            starts.append(T - CTX_FRAMES)
        wins = []
        for s in starts:
            w = mel[:, s:s + CTX_FRAMES]
            if w.shape[1] < CTX_FRAMES:
                w = np.concatenate([w, np.full((mel.shape[0], CTX_FRAMES - w.shape[1]), SILENT_MEL, np.float32)], 1)
            wins.append(w)
        return self.raw(np.stack(wins)), [s / FPS for s in starts]

    def raw(self, mel_windows: np.ndarray) -> np.ndarray:
        """[N, 128, T] windows used as they are -> [N, 527]."""
        out = []
        for i in range(0, len(mel_windows), BATCH):
            b = mel_windows[i:i + BATCH][:, None].astype(np.float32)
            out.append(self.session.run(None, {"mel": b})[0].reshape(len(b), -1))
        return np.concatenate(out) if out else np.zeros((0, 527), np.float32)

    def scores(self, mel_windows: np.ndarray) -> np.ndarray:
        """[N, 128, T] log-mel windows -> [N, 527] sigmoid scores; each window is centred in
        PAD_FRAMES of silence."""
        out = []
        n, f, t = mel_windows.shape if len(mel_windows) else (0, 128, WIN_FRAMES)
        a = max(0, (PAD_FRAMES - t) // 2)
        for i in range(0, n, BATCH):
            w = mel_windows[i:i + BATCH]
            b = np.full((len(w), 1, f, max(PAD_FRAMES, t)), SILENT_MEL, np.float32)
            b[:, 0, :, a:a + t] = w
            out.append(self.session.run(None, {"mel": b})[0].reshape(len(w), -1))
        return np.concatenate(out) if out else np.zeros((0, 527), np.float32)
