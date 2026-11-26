"""Create the end-to-end test folder used by the Windows build (real Danish/English speech).

Layout (stitched timeline):
  Recording 1.m4a   Danish FLEURS excerpt, AAC mono 48 kHz            0:00 - 1:08
  Recording 2.mp3   English FLEURS excerpt, MP3 stereo 44.1 kHz       1:08 - 1:49
  Recording 10.flac 10.5 min of quiet room noise with the Danish excerpt placed so that it
                    crosses the 10:00 segment boundary of the stitched recording
  broken.wma        a damaged file (must be reported and left untouched)
  notes.txt         an unrelated file (must be left untouched)
Writes <folder>.expect.json next to the folder with hashes and speech positions.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from conftest import write_audio  # noqa: E402


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    da, sr = sf.read(ROOT / "tests" / "data" / "fleurs_da_dk.flac", dtype="float32")
    en, _ = sf.read(ROOT / "tests" / "data" / "fleurs_en_us.flac", dtype="float32")
    refs = json.loads((ROOT / "tests" / "data" / "fleurs_refs.json").read_text(encoding="utf-8"))
    write_audio(out / "Recording 1.m4a", da, sr)
    write_audio(out / "Recording 2.mp3", np.stack([en, en * 0.9], 1), sr)
    t_a, t_b = len(da) / sr, len(en) / sr
    long_len = int(10.5 * 60 * sr)
    rng = np.random.default_rng(7)
    noise = (rng.standard_normal(long_len) * 0.002).astype(np.float32)
    start_in_c = 600.0 - (t_a + t_b) - t_a / 2           # centre of the clip at 10:00 stitched
    s = int(start_in_c * sr)
    c = noise.copy()
    c[s:s + len(da)] += da * 0.5
    write_audio(out / "Recording 10.flac", c, sr)
    (out / "broken.wma").write_bytes(rng.bytes(6000))
    (out / "notes.txt").write_text("unrelated file", encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    expect = {
        "hashes": hashes,
        "spans": [
            {"lang": "da", "start": 0.0, "end": t_a, "ref": refs["da_dk"]["text"]},
            {"lang": "en", "start": t_a, "end": t_a + t_b, "ref": refs["en_us"]["text"]},
            {"lang": "da", "start": t_a + t_b + start_in_c, "end": t_a + t_b + start_in_c + t_a, "ref": refs["da_dk"]["text"],
             "crosses": 600.0},
        ],
    }
    Path(str(out) + ".expect.json").write_text(json.dumps(expect, ensure_ascii=False, indent=1), encoding="utf-8")
    print("created", out, {k: v[:12] for k, v in hashes.items()})


if __name__ == "__main__":
    main()
