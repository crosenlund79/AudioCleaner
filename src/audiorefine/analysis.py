"""Per-recording acoustic analysis and the resulting (bounded) enhancement plan.

Every recording is analysed separately because recordings may come from different devices
and situations. The plan only chooses among a few conservative settings; there is no open-ended
adaptation, so behaviour stays predictable (see AUDIO_PROCESSING.md, "Adaptive decisions").
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .constants import FRAME, MASTER_SR
from .rawaudio import iter_raw, raw_length
from .vad import probs_at_frames


@dataclass
class SourceAnalysis:
    duration: float = 0.0
    peak_dbfs: float = -120.0
    clipped_fraction: float = 0.0
    dc_offset: float = 0.0
    speech_fraction: float = 0.0
    speech_level_dbfs: float = -120.0   # mean power of speech frames (active speech level)
    noise_level_dbfs: float = -120.0    # 20th percentile of non-speech frames
    snr_db: float = 0.0
    low_freq_excess_db: float = 0.0     # rumble/wind: power < 80 Hz relative to 200-2000 Hz (non-speech)
    low_freq_level_dbfs: float = -120.0 # absolute level of the 20-80 Hz band in pauses
    hum_hz: float = 0.0
    hum_harmonics: int = 0
    hum_prominence_db: float = 0.0
    bandwidth_hz: float = 0.0


@dataclass
class EnhancementPlan:
    pre_gain_db: float = 0.0
    hpf_hz: float = 70.0
    hpf_order: int = 2
    notch_hz: float = 0.0
    notch_count: int = 0
    df_atten_db: float = 0.0
    peaks: list[tuple[float, float, float]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def arfilter_args(self) -> list[str]:
        a = ["--sr", str(MASTER_SR), "--hpf", f"{self.hpf_hz:g}", "--hpf-order", str(self.hpf_order)]
        if self.notch_hz > 0 and self.notch_count > 0:
            a += ["--notch", f"{self.notch_hz:g}", "--notch-count", str(self.notch_count), "--notch-q", "30"]
        if self.df_atten_db > 0:
            a += ["--df-atten", f"{self.df_atten_db:g}"]
        for f, g, q in self.peaks:
            a += ["--peak", f"{f:g}:{g:g}:{q:g}"]
        return a

    def to_dict(self):
        return asdict(self)


def frame_stats(raw: Path, start: int = 0, end: int | None = None):
    """10 ms frame mean-square and peak arrays for a raw 48 kHz file region."""
    ms_list, pk_list = [], []
    rest = np.zeros(0, dtype=np.float32)
    for _, x in iter_raw(raw, FRAME * 6000, start, end):
        if rest.size:
            x = np.concatenate([rest, x])
        n = len(x) // FRAME
        fr = x[: n * FRAME].reshape(n, FRAME)
        ms_list.append(np.mean(np.square(fr, dtype=np.float64), axis=1))
        pk_list.append(np.max(np.abs(fr), axis=1) if n else np.zeros(0))
        rest = x[n * FRAME:]
    if rest.size:
        ms_list.append(np.array([np.mean(np.square(rest, dtype=np.float64))]))
        pk_list.append(np.array([np.max(np.abs(rest))]))
    ms = np.concatenate(ms_list) if ms_list else np.zeros(0)
    pk = np.concatenate(pk_list) if pk_list else np.zeros(0)
    return ms, pk


def _psd(raw: Path, frames_mask: np.ndarray | None, nfft: int = 65536, max_blocks: int = 120) -> np.ndarray | None:
    """Welch power spectrum (Hann, 50 % overlap) over blocks that are mostly non-speech if a mask is given."""
    total = raw_length(raw)
    if total < nfft:
        return None
    hop = nfft // 2
    starts = np.arange(0, total - nfft + 1, hop)
    if frames_mask is not None and frames_mask.size:
        keep = []
        for s in starts:
            a, b = s // FRAME, min(frames_mask.size, (s + nfft) // FRAME)
            if b > a and frames_mask[a:b].mean() > 0.8:
                keep.append(s)
        if len(keep) >= 4:
            starts = np.array(keep)
    if len(starts) > max_blocks:
        starts = starts[np.linspace(0, len(starts) - 1, max_blocks).astype(int)]
    win = np.hanning(nfft)
    acc = np.zeros(nfft // 2 + 1)
    for s in starts:
        x = np.fromfile(raw, dtype="<f4", count=nfft, offset=int(s) * 4).astype(np.float64)
        acc += np.abs(np.fft.rfft((x - x.mean()) * win)) ** 2
    return acc / max(len(starts), 1)


def prominent_of(psd: np.ndarray, f0: float, nfft: int = 65536) -> list[float]:
    """Prominence (dB) of harmonics 1-10 of ``f0`` over the median of their 4-12 Hz neighbourhood."""
    df = MASTER_SR / nfft
    out = []
    for h in range(1, 11):
        k = int(round(f0 * h / df))
        pk = psd[max(0, k - 2): k + 3].max()
        lo = psd[max(0, k - int(12 / df)): max(0, k - int(4 / df))]
        hi = psd[k + int(4 / df): k + int(12 / df)]
        neigh = np.median(np.concatenate([lo, hi])) if lo.size + hi.size else 0
        out.append(10 * math.log10((pk + 1e-20) / (neigh + 1e-20)))
    return out


def detect_hum(psd: np.ndarray | None, nfft: int = 65536) -> tuple[float, int, float]:
    """Mains hum at 50 or 60 Hz: returns (f0, highest prominent harmonic, max prominence dB)."""
    if psd is None:
        return 0.0, 0, 0.0
    best = (0.0, 0, 0.0)
    for f0 in (50.0, 60.0):
        prom = prominent_of(psd, f0, nfft)
        strong = [i + 1 for i, p in enumerate(prom) if p >= 12.0]
        if len(strong) >= 2 or (strong and prom[0] >= 20.0):
            score = sum(p for p in prom if p >= 12.0)
            if score > best[2]:
                best = (f0, max(strong), score)
    if best[1]:
        return best[0], best[1], float(max(prominent_of(psd, best[0], nfft)))
    return 0.0, 0, 0.0


def _band_power(psd: np.ndarray, f1: float, f2: float, nfft: int = 65536) -> float:
    df = MASTER_SR / nfft
    return float(psd[int(f1 / df): max(int(f1 / df) + 1, int(f2 / df))].mean())


def analyse_source(raw: Path, vad_probs_16k: np.ndarray, vad_offset_windows: int) -> tuple[SourceAnalysis, np.ndarray]:
    """Analyse one decoded recording. ``vad_probs_16k`` is the global VAD array; the recording's
    first 32 ms window index within it is ``vad_offset_windows``. Returns (analysis, speech_prob_per_frame)."""
    a = SourceAnalysis()
    total = raw_length(raw)
    a.duration = total / MASTER_SR
    ms, pk = frame_stats(raw)
    n = ms.size
    if n == 0:
        return a, np.zeros(0)
    frame_s = FRAME / MASTER_SR
    n_win = int(math.ceil(a.duration / 0.032)) + 1
    probs = vad_probs_16k[vad_offset_windows: vad_offset_windows + n_win]
    # align: window index relative to this recording start (offset within first window ignored, <32 ms)
    sp = probs_at_frames(probs, n, frame_s)
    speech = sp >= 0.5
    lv = 10 * np.log10(ms + 1e-14)
    a.peak_dbfs = float(20 * math.log10(float(pk.max()) + 1e-12))
    a.clipped_fraction = float(np.mean(pk >= 0.999))
    a.speech_fraction = float(speech.mean())
    if speech.sum() >= 50:
        a.speech_level_dbfs = float(10 * math.log10(ms[speech].mean() + 1e-14))
    else:
        a.speech_level_dbfs = float(np.percentile(lv, 95))
    nonspeech = ~speech
    if nonspeech.sum() >= 50:
        a.noise_level_dbfs = float(np.percentile(lv[nonspeech], 20))
    else:
        a.noise_level_dbfs = float(np.percentile(lv, 5))
    a.noise_level_dbfs = max(a.noise_level_dbfs, -100.0)   # digital silence is not 'noise'
    a.snr_db = a.speech_level_dbfs - a.noise_level_dbfs
    psd_ns = _psd(raw, nonspeech.astype(float))
    if psd_ns is not None:
        lf = _band_power(psd_ns, 20, 80)
        mid = _band_power(psd_ns, 200, 2000)
        a.low_freq_excess_db = float(10 * math.log10((lf + 1e-20) / (mid + 1e-20)))
        # absolute band level: Hann-windowed periodogram power -> mean-square per Hz * bandwidth
        win_pow = float(np.sum(np.hanning(65536) ** 2))
        a.low_freq_level_dbfs = float(10 * math.log10(lf * 2 * (60 / (MASTER_SR / 65536)) / win_pow / 65536 + 1e-20))
        f0, top, prom = detect_hum(psd_ns)
        a.hum_hz, a.hum_harmonics, a.hum_prominence_db = f0, top, float(prom)
    psd_sp = _psd(raw, speech.astype(float), max_blocks=60)
    if psd_sp is not None:
        ref = _band_power(psd_sp, 300, 3000)
        df = MASTER_SR / 65536
        ks = np.where(psd_sp[int(300 / df):] > ref * 10 ** (-5.0))[0]
        a.bandwidth_hz = float((ks.max() + int(300 / df)) * df) if ks.size else 0.0
    return a, sp


# --------------------------------------------------------------------------- the plan
DF_TARGET_SPEECH_DBFS = -26.0


def plan_enhancement(a: SourceAnalysis) -> EnhancementPlan:
    p = EnhancementPlan()
    # Static level so that DeepFilterNet sees speech at a normal level (it zeroes frames below -70 dBFS).
    p.pre_gain_db = float(np.clip(DF_TARGET_SPEECH_DBFS - a.speech_level_dbfs, -20.0, 45.0))
    # Rumble / wind / handling noise: steeper and slightly higher high-pass.
    if a.low_freq_excess_db > 10.0 and a.low_freq_level_dbfs > a.speech_level_dbfs - 35.0:
        p.hpf_hz, p.hpf_order = 80.0, 4
        p.reasons.append(f"strong low-frequency noise ({a.low_freq_excess_db:.0f} dB above mid band): 80 Hz 24 dB/oct high-pass")
    else:
        p.reasons.append("70 Hz 12 dB/oct high-pass (removes rumble, keeps voice fundamentals)")
    if a.hum_harmonics:
        # Only the fundamental and 2nd harmonic are notched: notching higher harmonics inside the
        # speech band measurably damages speech (ESTOI 0.996 -> 0.908 on clean speech with 9
        # notches, TEST_REPORT.md). DeepFilterNet removes the remaining harmonics.
        p.notch_hz, p.notch_count = a.hum_hz, min(2, a.hum_harmonics)
        p.reasons.append(f"mains hum at {a.hum_hz:.0f} Hz: notches at {a.hum_hz:.0f} and {2 * a.hum_hz:.0f} Hz")
    # Noise reduction strength from the measured signal-to-noise ratio (bounded attenuation).
    snr = a.snr_db
    if snr >= 40:
        p.df_atten_db = 6.0
    elif snr >= 25:
        p.df_atten_db = 10.0
    elif snr >= 12:
        p.df_atten_db = 14.0
    else:
        p.df_atten_db = 18.0
    p.reasons.append(f"estimated SNR {snr:.0f} dB: noise reduction limited to {p.df_atten_db:.0f} dB")
    # Gentle presence lift: 1-4 kHz carries most consonant information.
    p.peaks = [(3000.0, 2.0, 0.9)]
    return p
