import os

import numpy as np
import pytest
import soundfile as sf

from audiorefine.decode import DecodeError, choose_downmix, decode_to_raw
from audiorefine.discovery import scan_folder
from audiorefine.probe import probe
from audiorefine.rawaudio import RawWriter, read_raw
from conftest import tone, write_audio


def decode(path, tmp_path):
    out = tmp_path / (path.name + ".f32")
    with RawWriter(out) as w:
        r = decode_to_raw(path, w)
    return r, read_raw(out)


@pytest.mark.parametrize("sr", [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000])
def test_resampling_to_48k_preserves_duration(tmp_path, sr):
    p = write_audio(tmp_path / f"t{sr}.wav", tone(3.0, sr, 300), sr)
    r, y = decode(p, tmp_path)
    assert abs(len(y) - 3 * 48000) <= 2
    assert r.samples == len(y)
    rms = np.sqrt(np.mean(y[4800:-4800] ** 2))
    assert abs(rms - 0.3 / np.sqrt(2)) < 0.01


@pytest.mark.parametrize("ext", [".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".aiff", ".wma", ".aac"])
def test_mixed_formats_decode(tmp_path, ext):
    try:
        p = write_audio(tmp_path / f"file{ext}", tone(2.0, 44100 if ext != ".opus" else 48000, 500, channels=2), 44100 if ext != ".opus" else 48000)
    except Exception as e:  # encoder not available in this build
        pytest.skip(f"cannot create {ext}: {e}")
    info = probe(p)
    assert info.ok, info.error
    r, y = decode(p, tmp_path)
    # lossy codecs add encoder delay/padding handled by the demuxer; allow 60 ms
    assert abs(len(y) - 2 * 48000) < 0.06 * 48000, (ext, len(y))
    assert np.sqrt(np.mean(y ** 2)) > 0.1


def test_mono_and_stereo_downmix(tmp_path):
    sr = 48000
    x = tone(1.0, sr, 440)
    write_audio(tmp_path / "mono.wav", x, sr)
    write_audio(tmp_path / "stereo.wav", np.stack([x, x], 1), sr)
    write_audio(tmp_path / "inverted.wav", np.stack([x, -x], 1), sr)
    write_audio(tmp_path / "dead_right.wav", np.stack([x, np.zeros_like(x)], 1), sr)
    for name, mode in [("mono.wav", "mono"), ("stereo.wav", "average"), ("inverted.wav", "difference"),
                       ("dead_right.wav", "left")]:
        r, y = decode(tmp_path / name, tmp_path)
        assert r.downmix == mode, name
        assert abs(np.sqrt(np.mean(y ** 2)) - 0.3 / np.sqrt(2)) < 0.01, name  # never cancelled or halved


def test_choose_downmix_rules():
    assert choose_downmix(1.0, 1.0, 1.0) == "average"
    assert choose_downmix(1.0, 1.0, -1.0) == "difference"
    assert choose_downmix(1.0, 1e-6, 0.0) == "left"
    assert choose_downmix(1e-6, 1.0, 0.0) == "right"
    assert choose_downmix(1.0, 1.0, 0.0) == "average"  # independent channels (two speakers) are kept


def test_damaged_and_empty_files_reported(tmp_path):
    (tmp_path / "empty.wav").write_bytes(b"")
    (tmp_path / "garbage.m4a").write_bytes(os.urandom(4000))
    write_audio(tmp_path / "good.wav", tone(1.0, 16000), 16000)
    assert not probe(tmp_path / "empty.wav").ok
    assert not probe(tmp_path / "garbage.m4a").ok
    with pytest.raises(Exception):
        decode(tmp_path / "garbage.m4a", tmp_path)


def test_truncated_file_uses_readable_part(tmp_path):
    p = write_audio(tmp_path / "long.mp3", tone(5.0, 44100, 300), 44100)
    data = p.read_bytes()
    t = tmp_path / "truncated.mp3"
    t.write_bytes(data[: len(data) // 2])
    info = probe(t)
    assert info.ok
    r, y = decode(t, tmp_path)
    assert 1.5 * 48000 < len(y) < 3.5 * 48000


def test_scan_top_level_only_and_filters(tmp_path):
    root = tmp_path / "Audio dumps"
    root.mkdir()
    write_audio(root / "Recording2.wav", tone(1.0, 16000), 16000)
    write_audio(root / "Recording10.wav", tone(2.0, 16000), 16000)
    write_audio(root / "Recording1.flac", tone(1.5, 22050), 22050)
    (root / "notes.txt").write_text("unrelated")
    (root / "photo.jpg").write_bytes(b"\xff\xd8\xff")
    (root / ".hidden.wav").write_bytes(b"x")
    (root / "~$lock.wav").write_bytes(b"x")
    (root / "broken.mp3").write_bytes(os.urandom(3000))
    for sub in ("processed", "refined", "other"):
        (root / sub).mkdir()
        write_audio(root / sub / "old.wav", tone(1.0, 16000), 16000)
    res = scan_folder(root)
    assert [s.name for s in res.sources] == ["Recording1.flac", "Recording2.wav", "Recording10.wav"]
    assert [u.name for u in res.unreadable] == ["broken.mp3"]
    assert res.other_files == 2
    assert abs(res.total_duration - 4.5) < 0.05
