import numpy as np
import pytest

from audiorefine import analysis, enhance
from audiorefine.constants import FRAME, MASTER_SR
from audiorefine.hallucination import collapse_repetition, is_known_artefact, segment_is_uncertain
from audiorefine.language import LanguagePolicy
from audiorefine.paths import arfilter_path
from audiorefine.rawaudio import RawWriter, read_raw
from audiorefine.transcriber import CLIP_MAX_S, clip_gain, plan_blocks, plan_clips
from audiorefine.vad import VadParams, speech_regions

needs_arfilter = pytest.mark.skipif(not arfilter_path().is_file(), reason="arfilter helper not built")


def write(tmp_path, name, x):
    p = tmp_path / name
    with RawWriter(p) as w:
        w.write(x)
    return p


@needs_arfilter
@pytest.mark.parametrize("n", [0, 1, 479, 480, 481, 1440, 48000 * 3 + 17])
def test_arfilter_preserves_length(tmp_path, n):
    x = (np.random.default_rng(n).standard_normal(n) * 0.05).astype(np.float32)
    src = write(tmp_path, "s.f32", x)
    plan = analysis.EnhancementPlan(df_atten_db=12, peaks=[(3000, 2, 0.9)])
    dst = tmp_path / "d.f32"
    enhance.run_filter_chain(src, dst, plan, tmp_path, workers=2)
    assert len(read_raw(dst)) == n


@needs_arfilter
def test_arfilter_is_time_aligned(tmp_path):
    rng = np.random.default_rng(1)
    sp = np.zeros(MASTER_SR * 4, np.float32)
    t = np.arange(MASTER_SR) / MASTER_SR
    # broadband, speech-like bursts (a pure tone would only measure the filters' phase response)
    burst = (rng.standard_normal(MASTER_SR) * np.sin(2 * np.pi * 3 * t) ** 2 * 0.2).astype(np.float32)
    sp[MASTER_SR:2 * MASTER_SR] = burst
    x = sp + (rng.standard_normal(sp.size) * 0.003).astype(np.float32)
    src = write(tmp_path, "s.f32", x)
    dst = tmp_path / "d.f32"
    enhance.run_filter_chain(src, dst, analysis.EnhancementPlan(df_atten_db=20), tmp_path, workers=1)
    y = read_raw(dst)
    c = [float(np.dot(sp[2000:-2000], y[2000 + k:len(y) - 2000 + k])) for k in range(-200, 201)]
    assert abs(int(np.argmax(c)) - 200) <= 2  # no delay introduced


@needs_arfilter
def test_parallel_chunks_match_single_pass(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(MASTER_SR * 25) * 0.02).astype(np.float32)
    x[::4800] += 0.3
    src = write(tmp_path, "s.f32", x)
    plan = analysis.EnhancementPlan(df_atten_db=12)
    enhance.run_filter_chain(src, tmp_path / "one.f32", plan, tmp_path, workers=1)
    monkeypatch.setattr(enhance, "CHUNK_S", 7)
    enhance.run_filter_chain(src, tmp_path / "many.f32", plan, tmp_path, workers=3)
    a, b = read_raw(tmp_path / "one.f32"), read_raw(tmp_path / "many.f32")
    assert len(a) == len(b) == len(x)
    err = np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a ** 2))
    assert err < 0.05  # chunk boundaries are inaudible: < -26 dB difference overall


def test_leveller_limits_peaks_and_never_gates(tmp_path):
    sr = MASTER_SR
    rng = np.random.default_rng(5)
    quiet = (rng.standard_normal(sr * 6) * 0.003).astype(np.float32)        # quiet "speech"
    x = quiet.copy()
    x[sr * 3: sr * 3 + 2400] += 0.95 * np.sign(rng.standard_normal(2400)).astype(np.float32)  # door slam
    ms = np.array([np.mean(x[i:i + FRAME] ** 2) for i in range(0, len(x), FRAME)])
    pk = np.array([np.max(np.abs(x[i:i + FRAME])) for i in range(0, len(x), FRAME)])
    speech_prob = np.ones(ms.size)
    g, rep = enhance.leveller_gains(ms, pk, speech_prob)
    src = write(tmp_path, "s.f32", x)
    out = tmp_path / "o.f32"
    with RawWriter(out) as w:
        clipped = enhance.apply_gains(src, g, w)
    y = read_raw(out)
    assert len(y) == len(x)
    assert clipped == 0
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-3   # -1 dBFS ceiling
    # the quiet signal is raised substantially away from the slam...
    a = slice(sr, 2 * sr)
    assert np.sqrt(np.mean(y[a] ** 2)) > 5 * np.sqrt(np.mean(x[a] ** 2))
    # ...and nothing becomes silent
    frames = y[: len(y) // FRAME * FRAME].reshape(-1, FRAME)
    assert np.min(np.sqrt(np.mean(frames[1:-1] ** 2, axis=1))) > 1e-3


def test_leveller_equalises_distant_speaker():
    fps = MASTER_SR // FRAME
    ms = np.concatenate([np.full(fps * 4, 10 ** (-20 / 10)), np.full(fps * 4, 10 ** (-40 / 10))] * 2)
    pk = np.sqrt(ms) * 4
    g, _ = enhance.leveller_gains(ms, pk, np.ones(ms.size))
    loud = 10 * np.log10(ms[fps * 2]) + g[fps * 2]
    soft = 10 * np.log10(ms[fps * 6]) + g[fps * 6]
    assert abs(loud - soft) < 6  # a 20 dB difference is reduced to a few dB


def test_hum_detection():
    from audiorefine.analysis import detect_hum

    nfft = 65536
    t = np.arange(nfft * 8) / MASTER_SR
    rng = np.random.default_rng(0)
    for f0 in (50.0, 60.0):
        x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3)) * 0.05 + rng.standard_normal(t.size) * 0.01
        win = np.hanning(nfft)
        psd = np.mean([np.abs(np.fft.rfft(x[i:i + nfft] * win)) ** 2 for i in range(0, len(x) - nfft, nfft // 2)], axis=0)
        f, n, _ = detect_hum(psd)
        assert f == f0 and n >= 2
    psd = np.mean([np.abs(np.fft.rfft(rng.standard_normal(nfft) * np.hanning(nfft))) ** 2 for _ in range(8)], axis=0)
    assert detect_hum(psd)[1] == 0


def test_plan_is_bounded():
    a = analysis.SourceAnalysis(speech_level_dbfs=-60, noise_level_dbfs=-65, snr_db=5, low_freq_excess_db=20,
                                low_freq_level_dbfs=-50, hum_hz=50, hum_harmonics=9)
    p = analysis.plan_enhancement(a)
    assert p.df_atten_db <= 18           # never unlimited noise suppression
    assert p.notch_count <= 2
    assert p.pre_gain_db <= 45
    clean = analysis.plan_enhancement(analysis.SourceAnalysis(speech_level_dbfs=-20, noise_level_dbfs=-80, snr_db=60))
    assert clean.df_atten_db <= 6 and clean.hpf_hz == 70


# ------------------------------------------------------------------ VAD / clip planning
def test_speech_regions_and_padding():
    probs = np.zeros(1000, np.float32)
    probs[100:200] = 0.9
    probs[230:300] = 0.9      # 30 windows (~1 s) gap -> merged? min_silence 1 s
    probs[700:710] = 0.9
    regs = speech_regions(probs, 1000 * 512, VadParams())
    assert len(regs) >= 2
    assert regs[0][0] == 100 * 512 - int(0.4 * 16000)
    for a, b in regs:
        assert 0 <= a < b <= 1000 * 512


def test_clip_and_block_planning():
    sr = 16000
    regions = [(i * 5 * sr, i * 5 * sr + 3 * sr) for i in range(400)]   # 3 s speech every 5 s, ~33 min
    clips = plan_clips(regions)
    assert all(b - a <= CLIP_MAX_S * sr for a, b in clips)
    assert sum(1 for _ in clips) < len(regions)
    blocks = plan_blocks(clips)
    assert len(blocks) == 2
    covered = [c for blk in blocks for c in blk.clips]
    assert covered == clips
    long = plan_clips([(0, sr * 1500)], audio=np.random.default_rng(0).standard_normal(sr * 1500).astype(np.float32) * 0.01)
    assert len(long) >= 3 and long[0][0] == 0 and long[-1][1] == sr * 1500
    assert all(x[1] == y[0] for x, y in zip(long, long[1:]))


def test_clip_gain_lifts_quiet_speech():
    x = (np.random.default_rng(0).standard_normal(16000 * 5) * 10 ** (-60 / 20)).astype(np.float32)
    g = clip_gain(x)
    assert 20 * np.log10(g) > 30
    loud = (np.random.default_rng(0).standard_normal(16000) * 0.3).astype(np.float32)
    assert clip_gain(loud) <= 1.0


# ------------------------------------------------------------------ hallucination / language
def test_known_artefacts():
    assert is_known_artefact("Tak fordi du så med!")
    assert is_known_artefact("Undertekster af Nicolai Winther")
    assert is_known_artefact("Thanks for watching.")
    assert not is_known_artefact("Jeg tror ikke, at det er en god idé.")


def test_repetition_collapse():
    words = [" Det", " er", " fint."] + [" ja", " tak"] * 8
    a, b = collapse_repetition(words)
    assert b - a == 10 and words[:a] == [" Det", " er", " fint.", " ja", " tak", " ja", " tak", " ja", " tak"]
    assert collapse_repetition([" ja", " ja", " ja"]) is None  # normal speech


def test_uncertainty_flags():
    assert segment_is_uncertain(-1.2, [0.9], 0.0, 1.5)
    assert segment_is_uncertain(-0.2, [0.2, 0.3, 0.9], 0.0, 1.5)
    assert not segment_is_uncertain(-0.2, [0.9, 0.95, 0.8], 0.0, 1.5)


def test_language_policy():
    p = LanguagePolicy()
    assert p.choose({"no": 0.6, "da": 0.3, "en": 0.05}) == "da"     # Norwegian confusion -> Danish
    assert p.choose({"en": 0.55, "da": 0.4}) == "da"                 # weak evidence: keep Danish
    assert p.choose({"en": 0.9, "da": 0.05}) == "en"                 # clear switch
    assert p.choose({"da": 0.5, "en": 0.45}) == "en"                 # hysteresis keeps English
    assert p.choose({"de": 0.97, "da": 0.01, "en": 0.01}) == "de"    # confident other language preserved
    assert p.choose({"de": 0.7, "da": 0.1, "en": 0.15}) == "en"                 # not confident enough: supported language
    q = LanguagePolicy()
    assert q.choose({"sv": 0.8, "da": 0.15}) == "da"
