"""Event-free background material at 32 kHz for false-positive rates (speech, meeting speech, quiet room, speech+music)."""
import glob, io, numpy as np, soundfile as sf, pyarrow.parquet as pq
from scipy.signal import resample_poly, lfilter
rng = np.random.default_rng(5)
SR = 32000
def rs(x, sr):
    g = np.gcd(SR, sr); return resample_poly(x, SR // g, sr // g).astype(np.float32)
def pink(n, level_db):
    w = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]; a = [1, -2.494956002, 2.017265875, -0.522189400]
    p = lfilter(b, a, w); p /= np.sqrt(np.mean(p ** 2)) + 1e-12
    return (p * 10 ** (level_db / 20)).astype(np.float32)
def norm(x, db):
    r = np.sqrt(np.mean(x ** 2)) + 1e-12; return x * (10 ** (db / 20) / r)
# 1. read speech (FLEURS dev, Danish + English), pauses 0.4-2.5 s, speech at -26 dBFS, room noise -62 dBFS
parts = []
files = sorted(glob.glob("/home/user/testdata/fleurs_da_dk/dev/*.wav"))[:220] + sorted(glob.glob("/home/user/testdata/fleurs_en_us/dev/*.wav"))[:220]
rng.shuffle(files)
for f in files:
    x, sr = sf.read(f, dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
    parts += [norm(rs(x, sr), -26), np.zeros(int(rng.uniform(0.4, 2.5) * SR), np.float32)]
y = np.concatenate(parts); y += pink(len(y), -62)
sf.write("/home/user/sed/bg/speech_fleurs.flac", y, SR, subtype="PCM_16"); print("speech", len(y) / SR / 60, "min")
# 2. meeting speech (AMI single distant microphone), utterances concatenated
pf = pq.ParquetFile("/home/user/testdata/ami/sdm_test0.parquet")
parts = []
for g in range(pf.metadata.num_row_groups):
    for r in pf.read_row_group(g, columns=["audio"]).to_pylist():
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
        parts += [rs(x, sr), np.zeros(int(rng.uniform(0.2, 1.0) * SR), np.float32)]
y = np.concatenate(parts)
sf.write("/home/user/sed/bg/meeting_ami.flac", np.clip(y, -1, 1), SR, subtype="PCM_16"); print("ami", len(y) / SR / 60, "min")
# 3. quiet room: pink noise -58 dBFS + faint 50 Hz hum, 30 min
n = 30 * 60 * SR; t = np.arange(n) / SR
y = pink(n, -58) + (10 ** (-60 / 20) * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
sf.write("/home/user/sed/bg/quiet_room.flac", y, SR, subtype="PCM_16"); print("quiet 30 min")
# 4. speech over music / babble / traffic etc. from the earlier speech corpus (degradation set)
parts = []
for f in sorted(glob.glob("/home/user/corpus/*_music_10db.wav")) + sorted(glob.glob("/home/user/corpus/*_hum_rumble.wav")) + sorted(glob.glob("/home/user/corpus/*_distance_mix.wav")):
    x, sr = sf.read(f, dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
    parts.append(rs(x, sr))
y = np.concatenate(parts)
sf.write("/home/user/sed/bg/speech_degraded.flac", np.clip(y, -1, 1), SR, subtype="PCM_16"); print("degraded", len(y) / SR / 60, "min")
