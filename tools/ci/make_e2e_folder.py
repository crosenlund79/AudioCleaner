"""Create the end-to-end test folder used by the Windows build (real Danish/English speech, hourly layout).

Recordings (start time from the file name unless noted):
  20251125_080400.m4a   Danish, AAC                                  -> 2025-11-25 08-00
  20251125_083000.mp3   English, stereo MP3 (incl. a very quiet utterance) -> 08-00
  20251125_085826.flac  Danish speech from 08:59:26 to 09:00:34      -> split into 08-00 and 09-00
  20251125_100000.flac  12 minutes; Danish speech across 10:10:00    -> 10-00: Audio_001 (10:00) + Audio_002
  20251126_071200.wav   English                                      -> 2025-11-26 07-00
  Optagelse uden tid.m4a  no time in name/metadata; file date 2025-11-26 14:32:40 -> ≈ 14-00
  broken.wma            damaged (reported, untouched)
  notes.txt             unrelated (untouched)
Writes <folder>.expect.json next to the folder.
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


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    da, sr = sf.read(ROOT / "tests" / "data" / "fleurs_da_dk.flac", dtype="float32")
    en, _ = sf.read(ROOT / "tests" / "data" / "fleurs_en_us.flac", dtype="float32")
    refs = json.loads((ROOT / "tests" / "data" / "fleurs_refs.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(7)
    t_da, t_en = len(da) / sr, len(en) / sr

    def noise(seconds):
        return (rng.standard_normal(int(seconds * sr)) * 0.002).astype(np.float32)

    write_audio(out / "20251125_080400.m4a", da, sr)
    write_audio(out / "20251125_083000.mp3", np.stack([en, en * 0.9], 1), sr)
    x = noise(150)                                # file starts 08:58:26, speech from 08:59:26 to ~09:00:34
    x[int(60 * sr):int(60 * sr) + len(da)] += da * 0.6
    write_audio(out / "20251125_085826.flac", x, sr)
    y = noise(12 * 60)
    s = int((600 - t_da / 2) * sr)
    y[s:s + len(da)] += da * 0.5
    write_audio(out / "20251125_100000.flac", y, sr)
    write_audio(out / "20251126_071200.wav", en, sr)
    write_audio(out / "Optagelse uden tid.m4a", np.concatenate([noise(2), da[: int(20 * sr)], noise(2)]), sr)
    t = time.mktime((2025, 11, 26, 14, 32, 40, 0, 0, -1))
    os.utime(out / "Optagelse uden tid.m4a", (t, t))
    (out / "broken.wma").write_bytes(rng.bytes(6000))
    (out / "notes.txt").write_text("unrelated file", encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    expect = {
        "hashes": hashes,
        "hours": ["2025-11-25/08-00", "2025-11-25/09-00", "2025-11-25/10-00", "2025-11-26/07-00", "2025-11-26/14-00"],
        "processed": {"20251125_080400.m4a": "2025-11-25/08-00", "20251125_083000.mp3": "2025-11-25/08-00",
                      "20251125_085826.flac": "2025-11-25/08-00", "20251125_100000.flac": "2025-11-25/10-00",
                      "20251126_071200.wav": "2025-11-26/07-00", "Optagelse uden tid.m4a": "2025-11-26/14-00"},
        "spans": [
            {"label": "Danish 08:04", "start": "2025-11-25 08:04:00", "seconds": t_da, "ref": refs["da_dk"]["text"]},
            {"label": "English 08:30 (incl. quiet utterance)", "start": "2025-11-25 08:30:00", "seconds": t_en, "ref": refs["en_us"]["text"]},
            {"label": "Danish across 09:00", "start": "2025-11-25 08:59:26", "seconds": t_da, "ref": refs["da_dk"]["text"]},
            {"label": "Danish across the 10-minute file boundary", "start": f"2025-11-25 10:{int((600 - t_da / 2) // 60):02d}:{int((600 - t_da / 2) % 60):02d}",
             "seconds": t_da, "ref": refs["da_dk"]["text"]},
            {"label": "English next day 07:12", "start": "2025-11-26 07:12:00", "seconds": t_en, "ref": refs["en_us"]["text"]},
        ],
    }
    Path(str(out) + ".expect.json").write_text(json.dumps(expect, ensure_ascii=False, indent=1), encoding="utf-8")
    print("created", out, {k: v[:12] for k, v in hashes.items()})


if __name__ == "__main__":
    main()
