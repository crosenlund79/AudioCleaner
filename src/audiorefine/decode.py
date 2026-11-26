"""Decoding recordings to the 48 kHz mono working format.

* Any format FFmpeg (via PyAV) can decode is accepted; the original file is opened read-only.
* Resampling to 48 kHz uses FFmpeg's swresample; samples are appended back-to-back. No gap or
  silence is ever inserted, and container timestamp gaps are ignored, so the working timeline
  is exactly the sequence of decoded samples.
* Damaged packets are skipped (and reported) as long as the file yields audio at all.
* Stereo is down-mixed to mono by averaging, except when the channels are phase-inverted
  (averaging would cancel the speech) or one channel is dead (averaging would only add its
  noise). Files with more channels are averaged.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .constants import MASTER_SR
from .rawaudio import RawWriter

log = logging.getLogger("audiorefine.decode")


class DecodeError(Exception):
    pass


@dataclass
class DecodeResult:
    samples: int = 0
    native_rate: int = 0
    native_channels: int = 0
    downmix: str = "mono"
    damaged_packets: int = 0
    stopped_early: str = ""
    warnings: list[str] = field(default_factory=list)


def _iter_frames(path: Path, layout: str | None, result: DecodeResult, cancel: Callable[[], None] | None):
    """Yield (channels, samples) float32 arrays at 48 kHz."""
    import av

    resampler = None
    in_fmt = None
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        s.thread_type = "AUTO"
        result.native_rate = int(s.rate or 0)
        result.native_channels = int(s.channels or 0)
        target_layout = layout or ("mono" if s.channels == 1 else "stereo" if s.channels == 2 else None)
        packets = c.demux(s)
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                break
            except av.error.EOFError:
                break
            except Exception as e:  # container-level damage: keep what we have
                result.stopped_early = f"{type(e).__name__}: {e}"
                break
            if cancel:
                cancel()
            try:
                frames = packet.decode()
            except (av.error.InvalidDataError, av.error.ValueError, ValueError) as e:
                result.damaged_packets += 1
                continue
            except Exception as e:
                result.damaged_packets += 1
                log.debug("decode error: %s", e)
                continue
            for fr in frames:
                fmt = (fr.format.name, fr.layout.name, fr.sample_rate)
                if resampler is None or in_fmt != fmt:
                    lay = target_layout or fr.layout.name
                    if resampler is not None:
                        for o in resampler.resample(None):
                            yield o.to_ndarray()
                        result.warnings.append("Audio format changed inside the file; handled.")
                    resampler = av.AudioResampler(format="fltp", layout=lay, rate=MASTER_SR)
                    in_fmt = fmt
                fr.pts = None
                for o in resampler.resample(fr):
                    yield o.to_ndarray()
        if resampler is not None:
            for o in resampler.resample(None):
                yield o.to_ndarray()


def _stereo_stats(path: Path, cancel) -> tuple[float, float, float]:
    sll = srr = slr = 0.0
    dummy = DecodeResult()
    for a in _iter_frames(path, "stereo", dummy, cancel):
        if a.shape[0] < 2:
            continue
        l, r = a[0].astype(np.float64), a[1].astype(np.float64)
        sll += float(l @ l)
        srr += float(r @ r)
        slr += float(l @ r)
    return sll, srr, slr


def choose_downmix(sll: float, srr: float, slr: float) -> str:
    if sll <= 0 and srr <= 0:
        return "average"
    if min(sll, srr) <= max(sll, srr) * 1e-3:          # one channel > 30 dB weaker: dead / unconnected
        return "left" if sll >= srr else "right"
    avg_energy = (sll + srr + 2 * slr) / 4
    mean_energy = (sll + srr) / 2
    corr = slr / math.sqrt(sll * srr) if sll > 0 and srr > 0 else 0.0
    if avg_energy < 0.25 * mean_energy and corr < -0.3:  # phase inverted: averaging cancels speech
        return "difference"
    return "average"


def decode_to_raw(path: Path, out: RawWriter, on_samples: Callable[[np.ndarray], None] | None = None,
                  cancel: Callable[[], None] | None = None) -> DecodeResult:
    """Decode ``path`` to 48 kHz mono float32, appended to ``out``. ``on_samples`` sees every block."""
    result = DecodeResult()
    import av

    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise DecodeError("No audio stream found.")
        channels = int(c.streams.audio[0].channels or 1)
    mode = "mono"
    if channels == 2:
        sll, srr, slr = _stereo_stats(path, cancel)
        mode = choose_downmix(sll, srr, slr)
    elif channels > 2:
        mode = "average"
    result.downmix = mode
    for a in _iter_frames(path, None, result, cancel):
        if a.ndim == 1:
            x = a
        elif a.shape[0] == 1:
            x = a[0]
        elif mode == "left":
            x = a[0]
        elif mode == "right":
            x = a[1]
        elif mode == "difference":
            x = 0.5 * (a[0] - a[1])
        else:
            x = a.mean(axis=0)
        x = np.ascontiguousarray(x, dtype=np.float32)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        out.write(x)
        if on_samples:
            on_samples(x)
        result.samples += len(x)
    if result.samples == 0:
        raise DecodeError("No audio could be decoded (the file may be damaged).")
    if result.damaged_packets:
        result.warnings.append(f"{result.damaged_packets} damaged audio packet(s) were skipped.")
    if result.stopped_early:
        result.warnings.append("The file ends abruptly (it may be truncated); the readable part was used.")
    return result
