"""Build the controlled speech test corpus used in TEST_REPORT.md.

Source speech: Google FLEURS dev split (CC BY 4.0), Danish (da_dk) and English (en_us).
Utterances with distinct sentences are concatenated with natural pauses into ~4 minute
"conversations"; each is then degraded in several reproducible ways. Every condition keeps
exactly the same timeline as the clean version, so intelligibility (ESTOI) can be measured
against the clean reference and WER against the FLEURS reference transcripts.

Usage: python make_corpus.py <fleurs_root> <out_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 48000  # corpus is written at 48 kHz like a phone recording (content is 16 kHz band-limited)
RNG = np.random.default_rng(20261126)


def load_fleurs(root: Path, lang: str):
    rows = list(csv.reader(open(root / f"fleurs_{lang}" / "dev.tsv", encoding="utf-8"), delimiter="\t",
                           quoting=csv.QUOTE_NONE))
    seen, out = set(), []
    for r in rows:
        sid, fname, raw, norm = r[0], r[1], r[2], r[3]
        if sid in seen:
            continue
        seen.add(sid)
        out.append((fname, raw, norm, r[6] if len(r) > 6 else ""))
    return out


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def db(x):
    return 20 * np.log10(max(x, 1e-12))


def up48(x16):
    return signal.resample_poly(x16, 3, 1).astype(np.float32)


def build_conversation(root, lang, utts, target_s=240.0):
    pieces, refs, spans = [], [], []
    t = 0
    pieces.append(np.zeros(int(0.8 * SR), np.float32))
    t += len(pieces[-1])
    for fname, raw, norm, gender in utts:
        x, sr = sf.read(root / f"fleurs_{lang}" / "dev" / fname, dtype="float32")
        x = up48(x)
        x = x / (rms(x) + 1e-9) * 10 ** (-23 / 20)  # speech normalised to -23 dBFS RMS
        spans.append((t, t + len(x), gender))
        pieces.append(x)
        refs.append({"raw": raw, "norm": norm, "start": t / SR, "end": (t + len(x)) / SR})
        t += len(x)
        gap = np.zeros(int(RNG.uniform(0.4, 1.6) * SR), np.float32)
        pieces.append(gap)
        t += len(gap)
        if t / SR > target_s:
            break
    return np.concatenate(pieces), refs, spans


def babble(root, lang_pool, n_samples, talkers=8):
    out = np.zeros(n_samples, np.float32)
    for k in range(talkers):
        lang, utts = lang_pool[k % len(lang_pool)]
        stream, pos = np.zeros(n_samples, np.float32), 0
        order = RNG.permutation(len(utts))
        i = 0
        while pos < n_samples:
            x, _ = sf.read(root / f"fleurs_{lang}" / "dev" / utts[order[i % len(order)]][0], dtype="float32")
            x = up48(x)
            x /= rms(x) + 1e-9
            n = min(len(x), n_samples - pos)
            stream[pos:pos + n] = x[:n]
            pos += n
            i += 1
        out += stream
    return out / (rms(out) + 1e-9)


def colored_noise(n, exponent):
    """1/f^exponent noise, unit RMS."""
    spec = np.fft.rfft(RNG.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    spec /= f ** (exponent / 2)
    x = np.fft.irfft(spec, n).astype(np.float32)
    return x / rms(x)


def traffic(n):
    x = colored_noise(n, 1.6)  # brown-ish rumble
    sos = signal.butter(2, 2500, "low", fs=SR, output="sos")
    x = signal.sosfilt(sos, x).astype(np.float32)
    t = np.arange(n) / SR
    env = np.ones(n)
    for _ in range(int(n / SR / 12)):  # passing vehicles
        c = RNG.uniform(0, n / SR)
        env += 2.5 * np.exp(-0.5 * ((t - c) / RNG.uniform(1.5, 3.0)) ** 2)
    x = (x * env).astype(np.float32)
    return x / rms(x)


def hum(n, f0=50.0):
    t = np.arange(n) / SR
    x = np.zeros(n)
    for h, a in [(1, 1.0), (2, 0.3), (3, 0.6), (4, 0.15), (5, 0.35), (7, 0.2), (9, 0.1)]:
        x += a * np.sin(2 * np.pi * f0 * h * t + RNG.uniform(0, 6.28))
    x = x.astype(np.float32)
    return x / rms(x)


def music(n):
    """Synthetic 'TV/music' bed: chords with rhythm and a melody, unit RMS."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    notes = [220, 247, 262, 294, 330, 349, 392, 440]
    beat = 0.5
    for i in range(int(n / SR / beat) + 1):
        s, e = int(i * beat * SR), min(n, int((i + 1) * beat * SR))
        f = notes[RNG.integers(len(notes))]
        tt = t[s:e] - t[s]
        env = np.exp(-tt * 3)
        for h in range(1, 6):
            x[s:e] += env * np.sin(2 * np.pi * f * h * tt) / h
            x[s:e] += 0.5 * np.sin(2 * np.pi * (f / 2) * h * tt) / h
    x = x.astype(np.float32)
    return x / rms(x)


def rir(rt60, n_s=1.2, drr_db=-6.0):
    n = int(n_s * SR)
    t = np.arange(n) / SR
    tail = RNG.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.003 * SR)] = 0
    tail /= np.sqrt(np.sum(tail ** 2))
    h = tail * 10 ** (-drr_db / 20)
    h[0] += 1.0
    return h.astype(np.float32)


def bangs(n, speech_rms):
    x = np.zeros(n, np.float32)
    for _ in range(int(n / SR / 20)):
        c = int(RNG.uniform(1, n / SR - 1) * SR)
        L = int(0.25 * SR)
        burst = RNG.standard_normal(L) * np.exp(-np.arange(L) / (0.04 * SR))
        burst = signal.sosfilt(signal.butter(2, [80, 4000], "band", fs=SR, output="sos"), burst)
        burst = burst / np.max(np.abs(burst)) * 0.95
        x[c:c + L] += burst[: n - c]
    return x


def mix_snr(speech, noise, snr_db):
    s = rms(speech[np.abs(speech) > 1e-6]) if np.any(np.abs(speech) > 1e-6) else rms(speech)
    return speech + noise * (s / (rms(noise) + 1e-12)) * 10 ** (-snr_db / 20)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    pools = {lang: load_fleurs(root, lang) for lang in ("da_dk", "en_us")}
    manifest = {}
    for lang in ("da_dk", "en_us"):
        utts = pools[lang]
        clean, refs, spans = build_conversation(root, lang, utts[:60])
        n = len(clean)
        babble_pool = [(lg, pools[lg][60:]) for lg in ("da_dk", "en_us")]
        conds = {}
        conds["clean"] = clean
        # Quiet recording: speech at -58 dBFS with microphone self-noise at -80 dBFS
        conds["quiet"] = clean * 10 ** (-35 / 20) + colored_noise(n, 0.3) * 10 ** (-80 / 20)
        conds["babble_5db"] = mix_snr(clean, babble(root, babble_pool, n), 5)
        conds["traffic_0db"] = mix_snr(clean, traffic(n), 0)
        conds["hum_rumble"] = mix_snr(mix_snr(clean, hum(n), 15), colored_noise(n, 2.0), 10)
        rev = signal.fftconvolve(clean, rir(0.8, drr_db=6.0))[:n].astype(np.float32)
        conds["reverb_far"] = mix_snr(rev, colored_noise(n, 1.0), 20)
        # Two speakers at different distances: every other utterance 20 dB quieter
        var = clean.copy()
        for i, (s, e, _) in enumerate(spans):
            if i % 2:
                var[s:e] *= 10 ** (-20 / 20)
        conds["distance_mix"] = var + colored_noise(n, 1.0) * 10 ** (-60 / 20)
        conds["bangs"] = clean + bangs(n, rms(clean))
        conds["music_10db"] = mix_snr(clean, music(n), 10)
        files = {}
        for name, x in conds.items():
            peak = float(np.max(np.abs(x)))
            if peak > 0.99:
                x = x / peak * 0.99  # keep inside full scale (as a real recorder's limiter would)
            p = out / f"{lang}_{name}.wav"
            sf.write(p, x.astype(np.float32), SR, subtype="FLOAT")
            files[name] = str(p)
        sf.write(out / f"{lang}_reference_clean.wav", clean, SR, subtype="FLOAT")
        manifest[lang] = {"files": files, "refs": refs, "duration": n / SR}
        print(lang, f"{n / SR:.1f}s", len(refs), "utterances")
    json.dump(manifest, open(out / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
