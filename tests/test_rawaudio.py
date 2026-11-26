import numpy as np
import pytest

from audiorefine.rawaudio import Decimator3, RawWriter, iter_raw, raw_length, read_raw


def decimate_all(x, block=None):
    d = Decimator3()
    if block is None:
        return np.concatenate([d.process(x), d.flush()])
    parts = [d.process(x[i:i + block]) for i in range(0, len(x), block)]
    return np.concatenate(parts + [d.flush()])


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 239, 240, 241, 1000, 48000, 48001, 48002])
def test_decimator_exact_length(n):
    x = np.random.default_rng(n).standard_normal(n).astype(np.float32) * 0.1
    assert len(decimate_all(x)) == -(-n // 3)


def test_decimator_block_independent():
    x = np.random.default_rng(1).standard_normal(100_003).astype(np.float32) * 0.1
    a = decimate_all(x)
    b = decimate_all(x, block=777)
    c = decimate_all(x, block=1)
    assert np.allclose(a, b, atol=1e-6) and np.allclose(a, c, atol=1e-6)


def test_decimator_zero_delay_alignment():
    x = np.zeros(48000, np.float32)
    x[3 * 5000] = 1.0
    y = decimate_all(x)
    assert int(np.argmax(np.abs(y))) == 5000


def test_decimator_passband_and_stopband():
    t = np.arange(48000 * 2) / 48000
    y1 = decimate_all(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    y9 = decimate_all(np.sin(2 * np.pi * 9000 * t).astype(np.float32))
    assert abs(np.sqrt(np.mean(y1[2000:-2000] ** 2)) - np.sqrt(0.5)) < 0.01
    assert np.sqrt(np.mean(y9[2000:-2000] ** 2)) < 1e-3  # > 57 dB rejection of aliases


def test_raw_roundtrip(tmp_path):
    p = tmp_path / "a.f32"
    x = np.arange(10_000, dtype=np.float32)
    with RawWriter(p) as w:
        w.write(x[:3000])
        w.write(x[3000:])
    assert raw_length(p) == 10_000
    assert np.array_equal(read_raw(p, 2500, 100), x[2500:2600])
    got = np.concatenate([b for _, b in iter_raw(p, 999)])
    assert np.array_equal(got, x)
