"""Listening-path speech enhancement.

Per recording:  pre-gain -> [arfilter: high-pass -> hum notches -> DeepFilterNet3 (bounded) -> presence EQ]
                -> speech-aware leveller (slow AGC) -> look-ahead peak limiter -> junction fades

All stages preserve the timeline sample-for-sample (verified by tests): no samples are removed,
inserted, time-stretched or gated to silence.
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis import EnhancementPlan, frame_stats
from .constants import FRAME, JUNCTION_FADE_S, MASTER_SR
from .paths import arfilter_path
from .rawaudio import RawWriter, iter_raw, raw_length, read_raw

log = logging.getLogger("audiorefine.enhance")

CHUNK_S = 300            # parallel processing unit
PREROLL_S = 4.0          # warm-up audio before each chunk (discarded) so filter/network state is settled
XFADE = 960              # 20 ms crossfade between chunks

# Leveller
TARGET_SPEECH_DBFS = -20.0
LEVELLER_SIGMA_S = 0.75
LEVELLER_WIDE_SIGMA_S = 4.0
MAX_BOOST_OVER_REF_DB = 18.0
MAX_CUT_UNDER_REF_DB = 10.0
ABS_MAX_GAIN_DB = 45.0
ABS_MIN_GAIN_DB = -25.0
CEILING_DBFS = -1.0


class EnhanceError(Exception):
    pass


def _no_window_flags() -> int:
    return 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW


def arfilter_version() -> str:
    try:
        p = subprocess.run([str(arfilter_path()), "--info"], capture_output=True, text=True, timeout=60,
                           creationflags=_no_window_flags())
        return p.stdout.strip() if p.returncode == 0 else f"unavailable ({p.stderr.strip()[:200]})"
    except Exception as e:
        return f"unavailable ({e})"


def _run_arfilter(args: list[str], src: Path, start: int, end: int, gain: float, out_path: Path,
                  cancel: Callable[[], None] | None) -> None:
    exe = arfilter_path()
    if not exe.is_file():
        raise EnhanceError(f"The audio enhancement component is missing ({exe.name}).")
    with open(out_path, "wb") as fout:
        proc = subprocess.Popen([str(exe), *args], stdin=subprocess.PIPE, stdout=fout, stderr=subprocess.PIPE,
                                creationflags=_no_window_flags())
        err_chunks: list[bytes] = []
        t = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()), daemon=True)
        t.start()
        try:
            for _, x in iter_raw(src, 1 << 20, start, end):
                if cancel:
                    cancel()
                proc.stdin.write(np.ascontiguousarray(x * gain, dtype="<f4").tobytes())
            proc.stdin.close()
            rc = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        t.join(timeout=5)
    if rc != 0:
        msg = b"".join(err_chunks).decode("utf-8", "replace").strip()
        raise EnhanceError(f"Audio enhancement failed (code {rc}): {msg[:300]}")
    got = os.path.getsize(out_path) // 4
    if got != end - start:
        raise EnhanceError(f"Audio enhancement returned {got} samples, expected {end - start}.")


def run_filter_chain(src: Path, dst: Path, plan: EnhancementPlan, workdir: Path, workers: int,
                     progress: Callable[[float], None] | None = None, cancel: Callable[[], None] | None = None,
                     start: int = 0, end: int | None = None) -> None:
    """Run arfilter over ``src[start:end]`` in parallel chunks; ``dst`` gets exactly end-start samples.

    Warm-up audio before ``start`` (up to PREROLL_S) is taken from the source itself, so a part of a
    recording that begins at an hour boundary is processed with settled filter/network state.
    """
    src_total = raw_length(src)
    end = src_total if end is None else min(end, src_total)
    total = max(0, end - start)
    gain = 10 ** (plan.pre_gain_db / 20)
    args = plan.arfilter_args()
    chunk = int(CHUNK_S * MASTER_SR)
    pre = int(PREROLL_S * MASTER_SR)
    bounds = [(s, min(end, s + chunk)) for s in range(start, end, chunk)] or [(start, start)]
    jobs = []
    for k, (s, e) in enumerate(bounds):
        in_s = max(0, s - pre)
        in_e = min(end, e + XFADE) if k < len(bounds) - 1 else e
        jobs.append((k, s, e, in_s, in_e, workdir / f"chunk_{k:05d}.f32"))
    done = [0]
    lock = threading.Lock()

    def work(job):
        k, s, e, in_s, in_e, path = job
        _run_arfilter(args, src, in_s, in_e, gain, path, cancel)
        with lock:
            done[0] += e - s
            if progress:
                progress(done[0] / max(total, 1))

    if total > 0:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(work, j) for j in jobs]
            for f in futures:
                f.result()
    # assemble with crossfades
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, XFADE, dtype=np.float64))
    with RawWriter(dst) as w:
        tail = None
        for k, s, e, in_s, in_e, path in jobs:
            if total == 0:
                break
            y = read_raw(path)[s - in_s:]
            if tail is not None:
                n = min(XFADE, len(y), len(tail))
                y = y.copy()
                y[:n] = (tail[:n] * (1 - ramp[:n]) + y[:n] * ramp[:n]).astype(np.float32)
            main = y[: e - s]
            w.write(main)
            tail = y[e - s:] if len(y) > e - s else None
            os.remove(path)
        written = w.count
    if written != total:
        raise EnhanceError(f"Enhanced audio length mismatch ({written} != {total}).")


# ------------------------------------------------------------------------------ leveller
def _gauss_smooth(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    if x.size == 0:
        return x
    r = int(3 * sigma_frames)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    k /= k.sum()
    if x.size < 2 * r + 1:
        pad = np.pad(x, (r, r), mode="edge")
        return np.convolve(pad, k, mode="valid")[: x.size]
    # FFT convolution for long arrays
    n = x.size + 2 * r
    nfft = 1 << (n + k.size - 1).bit_length()
    pad = np.pad(x, (r, r), mode="edge")
    y = np.fft.irfft(np.fft.rfft(pad, nfft) * np.fft.rfft(k, nfft), nfft)
    return y[2 * r: 2 * r + x.size]


def _moving_max(x: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or x.size == 0:
        return x
    pad = np.pad(x, (r, r), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(pad, 2 * r + 1).max(axis=1)


def _moving_min(x: np.ndarray, r: int) -> np.ndarray:
    return -_moving_max(-x, r)


def _clip_transients(ms: np.ndarray, speech: np.ndarray, block: int = 50, headroom_db: float = 12.0) -> np.ndarray:
    """Limit frame energies to 12 dB above the local median speech level for *level estimation*.

    A door slam or a dropped phone would otherwise dominate the power average and make the
    leveller turn the surrounding speech down. (The audio itself is not changed here; the peak
    limiter handles the transient.)
    """
    n = ms.size
    lv = 10 * np.log10(ms + 1e-14)
    nb = -(-n // block)
    med = np.full(nb, np.nan)
    for b in range(nb):
        sl = slice(b * block, min(n, (b + 1) * block))
        sel = lv[sl][speech[sl]]
        if sel.size >= 5:
            med[b] = np.median(sel)
    ok = ~np.isnan(med)
    if not ok.any():
        return ms
    idx = np.arange(nb)
    med = np.interp(idx, idx[ok], med[ok])
    med = np.maximum(med, np.convolve(np.pad(med, 1, mode="edge"), np.ones(3) / 3, mode="valid"))
    ceiling = np.repeat(med, block)[:n] + headroom_db
    return np.minimum(ms, 10 ** (ceiling / 10))


@dataclass
class LevellerReport:
    global_speech_dbfs: float
    gain_min_db: float
    gain_median_db: float
    gain_max_db: float
    limited_fraction: float


def leveller_gains(ms: np.ndarray, pk: np.ndarray, speech_prob: np.ndarray) -> tuple[np.ndarray, LevellerReport]:
    """Frame gains (dB) for the enhanced signal: slow speech-level AGC + peak limit."""
    n = ms.size
    if n == 0:
        return np.zeros(0), LevellerReport(-120, 0, 0, 0, 0)
    speech = speech_prob[:n] >= 0.5 if speech_prob.size >= n else np.pad(speech_prob, (0, n - speech_prob.size)) >= 0.5
    fps = MASTER_SR / FRAME
    ms = _clip_transients(ms, speech)
    if speech.sum() >= 30:
        global_db = 10 * math.log10(ms[speech].mean() + 1e-14)
    else:
        global_db = float(np.percentile(10 * np.log10(ms + 1e-14), 95)) - 6
    w = speech.astype(np.float64)
    num = _gauss_smooth(w * ms, LEVELLER_SIGMA_S * fps)
    den = _gauss_smooth(w, LEVELLER_SIGMA_S * fps)
    num2 = _gauss_smooth(w * ms, LEVELLER_WIDE_SIGMA_S * fps)
    den2 = _gauss_smooth(w, LEVELLER_WIDE_SIGMA_S * fps)
    local = np.full(n, global_db)
    ok2 = den2 > 0.02
    local[ok2] = 10 * np.log10(num2[ok2] / den2[ok2] + 1e-14)
    ok = den > 0.05
    local[ok] = 10 * np.log10(num[ok] / den[ok] + 1e-14)
    g_ref = TARGET_SPEECH_DBFS - global_db
    g = TARGET_SPEECH_DBFS - local
    g = np.clip(g, g_ref - MAX_CUT_UNDER_REF_DB, g_ref + MAX_BOOST_OVER_REF_DB)
    g = np.clip(g, ABS_MIN_GAIN_DB, ABS_MAX_GAIN_DB)
    g = _gauss_smooth(g, 0.15 * fps)
    # look-ahead peak limit (2 frames either side), fast attack / short release
    pk_n = _moving_max(pk, 2)
    lim = CEILING_DBFS - 20 * np.log10(pk_n + 1e-9)
    gl = np.minimum(g, lim)
    gl = _moving_min(gl, 3)
    gl = _gauss_smooth(gl, 1.5)
    gl = np.minimum(gl, lim)
    rep = LevellerReport(global_speech_dbfs=float(global_db), gain_min_db=float(gl.min()),
                         gain_median_db=float(np.median(gl)), gain_max_db=float(gl.max()),
                         limited_fraction=float(np.mean(gl < g - 0.5)))
    return gl, rep


def apply_gains(src: Path, gains_db: np.ndarray, out: RawWriter, fade_in: bool = True, fade_out: bool = True,
                cancel: Callable[[], None] | None = None) -> int:
    """Apply frame gains (linear interpolation between frame centres) and junction fades."""
    total = raw_length(src)
    if total == 0:
        return 0
    lin = 10 ** (gains_db / 20)
    centers = (np.arange(lin.size) + 0.5) * FRAME
    nf = int(JUNCTION_FADE_S * MASTER_SR)
    fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nf))
    clipped = 0
    for pos, x in iter_raw(src, MASTER_SR * 30):
        if cancel:
            cancel()
        idx = np.arange(pos, pos + len(x), dtype=np.float64)
        y = x.astype(np.float64) * np.interp(idx, centers, lin)
        if fade_in and pos < nf:
            k = min(nf - pos, len(y))
            y[:k] *= fade[pos:pos + k]
        if fade_out and pos + len(x) > total - nf:
            a = max(0, total - nf - pos)
            seg = y[a:]
            fpos = pos + a - (total - nf)
            seg *= fade[::-1][fpos: fpos + len(seg)]
        over = np.abs(y) > 1.0
        if over.any():
            clipped += int(over.sum())
            np.clip(y, -1.0, 1.0, out=y)
        out.write(y.astype(np.float32))
    return clipped


def enhance_range(decoded: Path, start: int, end: int, plan: EnhancementPlan, speech_prob_frames: np.ndarray,
                  workdir: Path, out: RawWriter, workers: int, progress: Callable[[float], None] | None = None,
                  cancel: Callable[[], None] | None = None) -> dict:
    """Enhance one part of a decoded recording (one hour's piece) and append it to ``out``.

    ``speech_prob_frames`` covers the whole recording (10 ms frames); the leveller sees the frames
    of this part. 5 ms fades are applied at both ends of the part (they are recording/hour edges).
    """
    filtered = workdir / f"{decoded.stem}_{start}_filtered.f32"
    run_filter_chain(decoded, filtered, plan, workdir, workers,
                     progress=(lambda f: progress(0.85 * f)) if progress else None, cancel=cancel,
                     start=start, end=end)
    ms, pk = frame_stats(filtered)
    f0 = start // FRAME
    sp = speech_prob_frames[f0:f0 + ms.size]
    if sp.size < ms.size:
        sp = np.pad(sp, (0, ms.size - sp.size))
    gains, rep = leveller_gains(ms, pk, sp)
    clipped = apply_gains(filtered, gains, out, fade_in=True, fade_out=True, cancel=cancel)
    os.remove(filtered)
    if progress:
        progress(1.0)
    return {"leveller": rep.__dict__, "safety_clipped_samples": clipped}


def enhance_source(decoded: Path, plan: EnhancementPlan, speech_prob_frames: np.ndarray, workdir: Path,
                   out: RawWriter, workers: int, is_first: bool, is_last: bool,
                   progress: Callable[[float], None] | None = None, cancel: Callable[[], None] | None = None) -> dict:
    filtered = workdir / (decoded.stem + "_filtered.f32")
    run_filter_chain(decoded, filtered, plan, workdir, workers,
                     progress=(lambda f: progress(0.85 * f)) if progress else None, cancel=cancel)
    ms, pk = frame_stats(filtered)
    gains, rep = leveller_gains(ms, pk, speech_prob_frames)
    clipped = apply_gains(filtered, gains, out, fade_in=True, fade_out=True, cancel=cancel)
    os.remove(filtered)
    if progress:
        progress(1.0)
    return {"leveller": rep.__dict__, "safety_clipped_samples": clipped}
