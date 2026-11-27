"""Create the sound-analysis end-to-end folder (real CC0/public-domain sounds from tests/data/sounds,
mixed into room noise and Danish speech at known clock times). Writes <folder>.expect.json.

  20251127_074000.flac   07:40–07:46  speech; knock 07:41:40; alarm clock 07:43:00; typing 07:44:00
  20251127_085945.m4a    08:59:45–09:01:45  laughter 08:59:52 (hour 08); knocking 09:00:20 (hour 09)
  20251127_171000.mp3    17:10–17:14  frying 17:10:30; microwave 17:12:00 (-> possible cooking)
  20251128_031500.wav    03:15–03:18 (next day)  snoring 03:15:30; cat 03:16:30
  processed/2025-11-26/20-00/20251126_201500.ogg  (already archived)  doorbell 20:15:10; baby crying 20:15:40
  Optagelse uden tid.flac  no recording time; file date 2025-11-28 06:30:40 -> estimated (≈); sneeze
  broken.wma, notes.txt  (damaged / unrelated, must stay untouched)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from conftest import write_audio  # noqa: E402

SR = 32000
SND = ROOT / "tests" / "data" / "sounds"
rng = np.random.default_rng(21)


def load(name):
    x, sr = sf.read(SND / f"{name}.ogg", dtype="float32")
    return x if sr == SR else upsample(x, sr)


def upsample(x, sr):
    n = int(round(len(x) * SR / sr))
    return np.interp(np.arange(n) * sr / SR, np.arange(len(x)), x).astype(np.float32)


def active_rms(x):
    f = x[: len(x) // 320 * 320].reshape(-1, 320)
    e = np.sqrt((f ** 2).mean(1))
    top = np.sort(e)[-max(1, len(e) // 4):]
    return float(np.sqrt(np.mean(top ** 2)) + 1e-12)


def room(seconds):
    return (rng.standard_normal(int(seconds * SR)) * 10 ** (-56 / 20)).astype(np.float32)


def put(base, sig, at, level_db=-24.0):
    sig = sig * (10 ** (level_db / 20) / active_rms(sig))
    a = int(at * SR)
    base[a:a + len(sig)] += sig[: max(0, len(base) - a)]
    return base


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    da, sr = sf.read(ROOT / "tests" / "data" / "fleurs_da_dk.flac", dtype="float32")
    da = upsample(da, sr)
    expect = []

    def ev(cat, clock, src, date):
        expect.append({"category": cat, "date": date, "clock": clock, "source": src})

    a = room(360)
    put(a, da[: 60 * SR], 20, -26)
    put(a, load("knock"), 100)
    put(a, load("alarm"), 180)
    put(a, load("typing"), 240)
    write_audio(out / "20251127_074000.flac", a, SR)
    ev("knock", "07:41:40", "20251127_074000.flac", "2025-11-27")
    ev("alarm", "07:43:00", "20251127_074000.flac", "2025-11-27")
    ev("typing", "07:44:00", "20251127_074000.flac", "2025-11-27")

    b = room(120)
    put(b, load("laughter"), 7)
    put(b, load("knock2"), 35)
    write_audio(out / "20251127_085945.m4a", b, SR)
    ev("laughter", "08:59:52", "20251127_085945.m4a", "2025-11-27")
    ev("knock", "09:00:20", "20251127_085945.m4a", "2025-11-27")

    c = room(240)
    put(c, load("frying"), 30)
    put(c, load("microwave"), 120)
    write_audio(out / "20251127_171000.mp3", c, SR)
    ev("frying", "17:10:30", "20251127_171000.mp3", "2025-11-27")
    ev("microwave", "17:12:00", "20251127_171000.mp3", "2025-11-27")

    d = room(180)
    put(d, load("snoring"), 30)
    put(d, load("cat"), 90)
    write_audio(out / "20251128_031500.wav", d, SR)
    ev("snoring", "03:15:30", "20251128_031500.wav", "2025-11-28")
    ev("cat", "03:16:30", "20251128_031500.wav", "2025-11-28")

    arch = out / "processed" / "2025-11-26" / "20-00"
    arch.mkdir(parents=True)
    e = room(60)
    put(e, load("doorbell"), 10)
    put(e, load("baby_cry"), 40)
    write_audio(arch / "20251126_201500.ogg", e, SR)
    ev("doorbell", "20:15:10", "20251126_201500.ogg", "2025-11-26")
    ev("crying", "20:15:40", "20251126_201500.ogg", "2025-11-26")

    f = room(20)
    put(f, load("sneeze"), 5)
    write_audio(out / "Optagelse uden tid.flac", f, SR)
    t = time.mktime((2025, 11, 28, 6, 30, 40, 0, 0, -1))
    os.utime(out / "Optagelse uden tid.flac", (t, t))
    ev("sneeze", "06:30:25", "Optagelse uden tid.flac", "2025-11-28")    # 20 s file ending 06:30:40

    (out / "broken.wma").write_bytes(rng.bytes(5000))
    (out / "notes.txt").write_text("unrelated", encoding="utf-8")
    hashes = {str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in out.rglob("*") if p.is_file()}
    exp = {"hashes": hashes, "events": expect,
           "hours": ["2025-11-26/20-00", "2025-11-27/07-00", "2025-11-27/08-00", "2025-11-27/09-00",
                     "2025-11-27/17-00", "2025-11-28/03-00", "2025-11-28/06-00"]}
    Path(str(out) + ".expect.json").write_text(json.dumps(exp, ensure_ascii=False, indent=1), encoding="utf-8")
    print("created", out)


if __name__ == "__main__":
    main()
