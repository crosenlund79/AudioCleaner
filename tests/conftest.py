from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(__file__).parent / "data"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test")
    config.addinivalue_line("markers", "model: needs a Whisper model (AUDIOREFINE_TEST_MODEL)")


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path_factory, monkeypatch):
    d = tmp_path_factory.mktemp("ardata")
    monkeypatch.setenv("AUDIOREFINE_DATA_DIR", str(d))
    return d


def tiny_model_dir() -> Path | None:
    for c in (os.environ.get("AUDIOREFINE_TEST_MODEL"), str(Path.home() / "models" / "faster-whisper-tiny"),
              "/home/user/models/faster-whisper-tiny"):
        if c and (Path(c) / "model.bin").is_file():
            return Path(c)
    return None


@pytest.fixture
def model_dir():
    d = tiny_model_dir()
    if d is None:
        pytest.skip("no Whisper test model available (set AUDIOREFINE_TEST_MODEL)")
    return d


def tone(seconds: float, sr: int, freq: float = 440.0, channels: int = 1, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(round(seconds * sr))) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 1:
        return x
    return np.stack([x] * channels, axis=1)


def speech(lang: str = "da_dk") -> tuple[np.ndarray, int]:
    x, sr = sf.read(DATA / f"fleurs_{lang}.flac", dtype="float32")
    return x, sr


def write_audio(path: Path, x: np.ndarray, sr: int, codec: str | None = None) -> Path:
    """Write audio in the format implied by the extension (lossy formats via PyAV)."""
    ext = path.suffix.lower()
    if ext in (".wav", ".flac", ".ogg", ".aiff", ".aif"):
        sf.write(path, x, sr, format={".ogg": "OGG", ".aif": "AIFF"}.get(ext))
        return path
    import av

    fmt = {".m4a": ("ipod", "aac"), ".mp3": ("mp3", "libmp3lame"), ".opus": ("ogg", "libopus"),
           ".wma": ("asf", "wmav2"), ".aac": ("adts", "aac")}[ext]
    if x.ndim == 1:
        x = x[:, None]
    ch = x.shape[1]
    rate = sr
    if fmt[1] == "libopus" and sr not in (8000, 12000, 16000, 24000, 48000):
        rate = 48000
    with av.open(str(path), "w", format=fmt[0]) as c:
        try:
            s = c.add_stream(codec or fmt[1], rate=rate)
            if fmt[1] == "wmav2":
                s.bit_rate = 128000
        except Exception:
            s = c.add_stream("mp3" if fmt[1] == "libmp3lame" else fmt[1], rate=rate)
        s.layout = "mono" if ch == 1 else "stereo"
        frame_size = 1152 if "mp3" in fmt[1] else 1024
        data = np.ascontiguousarray(x.T.astype(np.float32))
        resampler = av.AudioResampler(format=s.codec_context.format.name if s.codec_context.format else "fltp",
                                      layout=s.layout.name, rate=rate)
        for i in range(0, data.shape[1], frame_size):
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(data[:, i:i + frame_size]), format="fltp",
                                            layout=s.layout.name)
            fr.sample_rate = sr
            for f2 in resampler.resample(fr):
                for p in s.encode(f2):
                    c.mux(p)
        for f2 in resampler.resample(None):
            for p in s.encode(f2):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)
    return path
