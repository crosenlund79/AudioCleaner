"""Analysis of one original recording: decode → log-mel → window scores → events.

* The **original** file is read (never the enhanced audio, which deliberately removes many of the
  sounds this analysis looks for). It is opened read-only and decoded with FFmpeg (PyAV) directly
  to 32 kHz, the model's rate, in blocks, so memory use does not grow with the recording length.
* Stereo is mixed to mono by averaging; when the two channels are phase-inverted (averaging would
  cancel them) the difference is used instead, decided per block.
* Positions are counted in decoded samples from the start of the file, the same timeline the
  transcription uses (there at 48 kHz), so an offset here is an offset in the original recording.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .events import build_events
from .model import (CTX_FRAMES, CTX_HOP_FRAMES, CTX_S, FPS, HOP_FRAMES, HOP_S, SILENT_MEL, SR, WIN_FRAMES, WIN_S,
                    SoundModel, StreamingMel)
from .vocab import Vocabulary

log = logging.getLogger("audiorefine.sound")

BLOCK_S = 20.0
MODEL_BATCH = 256


class AudioReadError(Exception):
    pass


def iter_audio(path: Path, cancel: Callable[[], None] | None = None, stats: dict | None = None,
               block_s: float = BLOCK_S) -> Iterator[np.ndarray]:
    """Yield float32 mono blocks at 32 kHz."""
    import av

    stats = stats if stats is not None else {}
    stats.setdefault("damaged_packets", 0)
    stats.setdefault("downmix", "mono")
    block = int(block_s * SR)
    buf: list[np.ndarray] = []
    nbuf = 0

    def mix(parts: list[np.ndarray]) -> np.ndarray:
        a = np.concatenate(parts, axis=1)
        if a.shape[0] == 1:
            return a[0]
        if a.shape[0] == 2:
            l, r = a[0].astype(np.float64), a[1].astype(np.float64)
            sll, srr, slr = float(l @ l), float(r @ r), float(l @ r)
            if sll > 0 and srr > 0:
                corr = slr / np.sqrt(sll * srr)
                if (sll + srr + 2 * slr) / 4 < 0.25 * (sll + srr) / 2 and corr < -0.3:
                    stats["downmix"] = "difference"
                    return (0.5 * (a[0] - a[1])).astype(np.float32)
            stats["downmix"] = "average" if stats["downmix"] != "difference" else "difference"
        return a.mean(axis=0).astype(np.float32)

    try:
        container = av.open(str(path))
    except Exception as e:
        raise AudioReadError(f"Cannot open the file: {e}") from e
    with container as c:
        if not c.streams.audio:
            raise AudioReadError("No audio stream found.")
        s = c.streams.audio[0]
        s.thread_type = "AUTO"
        layout = "mono" if (s.channels or 1) == 1 else "stereo"
        resampler = None
        fmt = None
        packets = c.demux(s)
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                break
            except Exception:  # container-level damage: keep what was read
                break
            if cancel:
                cancel()
            try:
                frames = packet.decode()
            except Exception:
                stats["damaged_packets"] += 1
                continue
            for fr in frames:
                f = (fr.format.name, fr.layout.name, fr.sample_rate)
                if resampler is None or f != fmt:
                    if resampler is not None:
                        for o in resampler.resample(None):
                            buf.append(o.to_ndarray()); nbuf += buf[-1].shape[1]
                    resampler = av.AudioResampler(format="fltp", layout=layout, rate=SR)
                    fmt = f
                fr.pts = None
                for o in resampler.resample(fr):
                    a = o.to_ndarray()
                    buf.append(a); nbuf += a.shape[1]
                if nbuf >= block:
                    x = mix(buf)
                    buf, nbuf = [], 0
                    yield np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if resampler is not None:
            for o in resampler.resample(None):
                buf.append(o.to_ndarray()); nbuf += buf[-1].shape[1]
    if nbuf:
        yield np.nan_to_num(mix(buf), nan=0.0, posinf=0.0, neginf=0.0)


def category_matrix(vocab: Vocabulary) -> list[np.ndarray]:
    return [np.array(c.index, dtype=np.int64) for c in vocab.categories]


def combine(S_short: np.ndarray, short_starts: np.ndarray, S_ctx: np.ndarray, ctx_starts: np.ndarray,
            vocab: Vocabulary) -> np.ndarray:
    """Score per 2 s window: geometric mean of the window's own score and the best 10 s context
    window around it; media categories (music, TV, games) use the context score alone."""
    if len(short_starts) == 0:
        return S_short
    centre = short_starts + WIN_S / 2
    C = np.zeros_like(S_short)
    for k, t in enumerate(centre):
        cover = (ctx_starts <= t) & (t < ctx_starts + CTX_S)
        if not cover.any():
            cover = np.zeros(len(ctx_starts), bool)
            cover[int(np.argmin(np.abs(ctx_starts + CTX_S / 2 - t)))] = True
        C[k] = S_ctx[cover].max(axis=0)
    media = np.array([c.kind == "media" for c in vocab.categories])
    out = np.sqrt(np.clip(S_short, 0, 1) * np.clip(C, 0, 1))
    out[:, media] = C[:, media]
    return out.astype(np.float32)


def analyze_file(path: Path, model: SoundModel, vocab: Vocabulary, cancel: Callable[[], None] | None = None,
                 progress: Callable[[float], None] | None = None, expected_duration: float = 0.0,
                 keep_scores: bool = False) -> dict:
    """Analyse one recording. Returns duration, events (seconds from the start of the file),
    media-playback spans and diagnostics."""
    stats: dict = {}
    sm = StreamingMel(model.frontend)
    cols = category_matrix(vocab)
    frames = np.zeros((128, 0), np.float32)
    f0 = 0                                   # global index of frames[:, 0]
    nxt = {"short": 0, "ctx": 0}             # next window start frame per scale
    pend = {"short": [], "ctx": []}
    starts = {"short": [], "ctx": []}
    out = {"short": [], "ctx": []}
    n_samples = 0

    def run(kind: str):
        if not pend[kind]:
            return
        W = np.stack(pend[kind])
        P = model.scores(W) if kind == "short" else model.raw(W)
        out[kind].append(np.stack([P[:, ix].max(axis=1) for ix in cols], axis=1).astype(np.float32))
        pend[kind].clear()

    def window(a_global: int, n: int) -> np.ndarray:
        a = a_global - f0
        w = frames[:, max(0, a):a + n]
        if w.shape[1] < n:
            w = np.concatenate([w, np.full((128, n - w.shape[1]), SILENT_MEL, np.float32)], axis=1)
        return w

    def take_windows(final: bool):
        nonlocal frames, f0
        total = f0 + frames.shape[1]
        for kind, n, hop in (("short", WIN_FRAMES, HOP_FRAMES), ("ctx", CTX_FRAMES, CTX_HOP_FRAMES)):
            while nxt[kind] + n <= total:
                pend[kind].append(window(nxt[kind], n))
                starts[kind].append(nxt[kind] / FPS)
                nxt[kind] += hop
                if len(pend[kind]) >= MODEL_BATCH:
                    run(kind)
            if final:
                if not starts[kind]:                       # recording shorter than one window
                    pend[kind].append(window(0, n))
                    starts[kind].append(0.0)
                elif starts[kind][-1] * FPS + n < total - (hop // 2 if kind == "short" else 0):
                    pend[kind].append(window(total - n, n))  # last window aligned to the end
                    starts[kind].append((total - n) / FPS)
                run(kind)
        if final:
            return
        keep = max(0, min(nxt["short"], nxt["ctx"], total - CTX_FRAMES) - f0)
        frames = frames[:, keep:]
        f0 += keep

    for x in iter_audio(path, cancel, stats):
        n_samples += len(x)
        frames = np.concatenate([frames, sm.push(x)], axis=1)
        take_windows(False)
        if progress and expected_duration > 0:
            progress(min(1.0, n_samples / SR / expected_duration))
    if n_samples == 0:
        raise AudioReadError("No audio could be decoded (the file may be damaged).")
    frames = np.concatenate([frames, sm.push(np.zeros(0, np.float32), final=True)], axis=1)
    take_windows(True)
    nc = len(vocab.categories)
    Ss = np.concatenate(out["short"]) if out["short"] else np.zeros((0, nc), np.float32)
    Sc = np.concatenate(out["ctx"]) if out["ctx"] else np.zeros((0, nc), np.float32)
    st = np.array(starts["short"], dtype=np.float64)
    S = combine(Ss, st, Sc, np.array(starts["ctx"], dtype=np.float64), vocab)
    duration = n_samples / SR
    events, media = build_events(S, st, vocab, WIN_S, HOP_S, duration)
    res = {
        "duration": round(duration, 3), "samples_32k": n_samples, "windows": int(len(st)),
        "context_windows": len(starts["ctx"]),
        "events": events, "media_spans": [[round(a, 2), round(b, 2)] for a, b in media],
        "downmix": stats.get("downmix", "mono"), "damaged_packets": stats.get("damaged_packets", 0),
        "category_peak": {c.id: round(float(S[:, j].max()), 4) for j, c in enumerate(vocab.categories) if len(S)},
    }
    if keep_scores:
        res["_scores"] = S
        res["_starts"] = st
        res["_short"] = Ss
        res["_ctx"] = Sc
        res["_ctx_starts"] = np.array(starts["ctx"])
    return res
