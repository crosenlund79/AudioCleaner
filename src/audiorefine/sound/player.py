"""Playback of an original recording (or an enhanced audio file) from any position.

The file is decoded with FFmpeg (PyAV) – the same decoder the analysis used – starting from the
requested position, and played through the default sound output with PortAudio
(``sounddevice``). The position shown is the position in the file, so "play from the event"
starts exactly where the analysis placed it. Nothing is written; the file is only read.
"""
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("audiorefine.player")

OUT_SR = 48000
CHANNELS = 2
BLOCK = 2048


class PlayerError(Exception):
    pass


def probe_duration(path: Path) -> float:
    import av

    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        if s.duration is not None and s.time_base is not None:
            return float(s.duration * s.time_base)
        if c.duration:
            return c.duration / 1_000_000
    return 0.0


def decode_from(path: Path, start: float, stop_event: threading.Event | None = None):
    """Yield float32 [n, 2] blocks at 48 kHz starting at ``start`` seconds into the file."""
    import av

    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        tb = s.time_base
        first_pts_t = None
        if start > 0 and tb is not None:
            try:
                c.seek(int(start / tb), stream=s, backward=True, any_frame=False)
                seeked = True
            except Exception:
                seeked = False
        else:
            seeked = False
        res = av.AudioResampler(format="flt", layout="stereo", rate=OUT_SR)
        pos = None            # time (s) of the next output sample
        for packet in c.demux(s):
            if stop_event is not None and stop_event.is_set():
                return
            try:
                frames = packet.decode()
            except Exception:
                continue
            for fr in frames:
                if pos is None:
                    if seeked and fr.pts is not None:
                        # positions count from the first decoded sample (e.g. after the MP3 encoder delay)
                        pos = float((fr.pts - (s.start_time or 0)) * tb)
                    else:
                        pos = 0.0
                    if first_pts_t is None:
                        first_pts_t = pos
                fr.pts = None
                for o in res.resample(fr):
                    a = o.to_ndarray().reshape(-1, CHANNELS)
                    n = len(a)
                    if pos + n / OUT_SR <= start:
                        pos += n / OUT_SR
                        continue
                    if pos < start:
                        cut = int(round((start - pos) * OUT_SR))
                        a = a[cut:]
                        pos = start
                    yield np.ascontiguousarray(a, dtype=np.float32)
                    pos += len(a) / OUT_SR
        for o in res.resample(None):
            yield np.ascontiguousarray(o.to_ndarray().reshape(-1, CHANNELS), dtype=np.float32)


class SoundDeviceOutput:
    """Default output: PortAudio via sounddevice."""

    def __init__(self, callback: Callable[[np.ndarray], None]):
        try:
            import sounddevice as sd
        except Exception as e:  # PortAudio missing
            raise PlayerError(f"Audio output is not available ({e}).") from e
        self._cb = callback
        try:
            self.stream = sd.OutputStream(samplerate=OUT_SR, channels=CHANNELS, dtype="float32", blocksize=BLOCK,
                                          callback=self._callback)
        except Exception as e:
            raise PlayerError(f"No audio output device could be opened ({e}).") from e

    def _callback(self, outdata, frames, time_info, status):
        self._cb(outdata)

    def start(self):
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop()
        except Exception:
            pass

    def close(self):
        try:
            self.stream.close()
        except Exception:
            pass


class Player:
    """play(path, start) / pause() / resume() / seek(t) / stop(); position() in seconds of the file."""

    def __init__(self, output_factory: Callable[[Callable[[np.ndarray], None]], object] | None = None):
        self.output_factory = output_factory or SoundDeviceOutput
        self.path: Path | None = None
        self.duration = 0.0
        self._out = None
        self._q: queue.Queue = queue.Queue(maxsize=48)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._buf = np.zeros((0, CHANNELS), np.float32)
        self._start = 0.0
        self._played = 0
        self._eof = False
        self.playing = False
        self.finished = False
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- control
    def open(self, path: Path) -> float:
        self.stop()
        self.path = Path(path)
        try:
            self.duration = probe_duration(self.path)
        except Exception as e:
            raise PlayerError(f"The recording cannot be opened ({e}).") from e
        return self.duration

    def play(self, start: float = 0.0) -> None:
        if self.path is None:
            raise PlayerError("No recording selected.")
        self._halt()
        start = max(0.0, float(start))
        if self.duration:
            start = min(start, max(0.0, self.duration - 0.05))
        if self._out is None:
            self._out = self.output_factory(self._fill)     # may raise PlayerError: nothing started yet
        self._start, self._played, self._eof, self.finished = start, 0, False, False
        self._buf = np.zeros((0, CHANNELS), np.float32)
        self._q = queue.Queue(maxsize=48)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._decode, args=(self.path, start, self._stop, self._q), daemon=True)
        self._thread.start()
        try:
            self._out.start()
        except Exception as e:
            self._halt()
            raise PlayerError(f"The sound output could not be started ({e}).") from e
        self.playing = True

    def pause(self) -> None:
        if self._out is not None and self.playing:
            self._out.stop()
        self.playing = False

    def resume(self) -> None:
        if self._out is not None and not self.playing and not self.finished:
            self._out.start()
            self.playing = True

    def seek(self, t: float) -> None:
        was = self.playing
        self.play(t)
        if not was:
            self.pause()

    def stop(self) -> None:
        self._halt()
        if self._out is not None:
            self._out.close()
            self._out = None

    def position(self) -> float:
        return self._start + self._played / OUT_SR

    # ---------------------------------------------------------------- internals
    def _halt(self):
        self._stop.set()
        if self._out is not None:
            self._out.stop()
        if self._thread is not None:
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
            self._thread.join(timeout=2)
            self._thread = None
        self.playing = False

    @staticmethod
    def _decode(path, start, stop, q):
        try:
            for a in decode_from(path, start, stop):
                while not stop.is_set():
                    try:
                        q.put(a, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except Exception as e:
            log.warning("Playback decoding stopped: %s", e)
        q.put(None)

    def _fill(self, out: np.ndarray) -> None:
        n = len(out)
        while len(self._buf) < n and not self._eof:
            try:
                a = self._q.get(timeout=0.05)
            except queue.Empty:
                break
            if a is None:
                self._eof = True
                break
            self._buf = np.concatenate([self._buf, a])
        k = min(n, len(self._buf))
        out[:k] = self._buf[:k]
        out[k:] = 0
        self._buf = self._buf[k:]
        self._played += k
        if self._eof and k < n and len(self._buf) == 0:
            self.finished = True
            self.playing = False
