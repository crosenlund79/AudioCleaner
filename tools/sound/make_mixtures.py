"""Difficult-condition test material from ESC-50 (Freesound recordings, independent of AudioSet):
each event clip is mixed into a 10 s background (conversation; TV-like music+speech through a small
loudspeaker; quiet room) at several levels relative to the background, optionally made 'distant'
(reverberation + 10 dB lower). Ground truth: the event's category, placed at 2.5–7.5 s.

usage: make_mixtures.py ESC50_DIR SPEECH.flac TVBED.flac OUT_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt

SR = 32000
ESC_TO_CAT = {
    "crying_baby": "crying", "sneezing": "sneeze", "coughing": "cough", "laughing": "laughter",
    "snoring": "snoring", "breathing": "breathing", "footsteps": "footsteps", "door_wood_knock": "knock",
    "mouse_click": "clicking", "keyboard_typing": "typing", "door_wood_creaks": "squeak",
    "vacuum_cleaner": "vacuum", "clock_alarm": "alarm", "clock_tick": "clock", "glass_breaking": "glass_break",
    "toilet_flush": "toilet", "pouring_water": "pour", "water_drops": "drip", "brushing_teeth": "toothbrush",
    "dog": "dog", "cat": "cat", "rooster": "rooster", "hen": "rooster", "crow": "bird", "chirping_birds": "bird",
    "rain": "rain", "wind": "wind", "thunderstorm": "thunder", "siren": "siren", "car_horn": "horn",
    "engine": "engine", "train": "train", "helicopter": "helicopter", "airplane": "aircraft", "chainsaw": "chainsaw",
    "church_bells": "church_bell", "fireworks": "fireworks", "hand_saw": "sawing", "insects": "insect",
    "crickets": "insect", "frog": "frog", "clapping": "clapping", "washing_machine": None, "can_opening": None,
    "drinking_sipping": None, "sea_waves": None, "crackling_fire": None, "pig": None, "cow": "cattle", "sheep": None,
}
rng = np.random.default_rng(11)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def active_rms(x):
    f = x[: len(x) // 320 * 320].reshape(-1, 320)
    e = np.sqrt((f ** 2).mean(1))
    top = np.sort(e)[-max(1, len(e) // 4):]
    return float(np.sqrt(np.mean(top ** 2)) + 1e-12)


def rir(rt60=0.7, n=int(0.8 * SR)):
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    h[0] = 0.0
    h /= np.sqrt(np.sum(h ** 2))
    d = np.zeros(n)
    d[0] = 0.35                       # weak direct path (distant source)
    return d + h


def main():
    esc, speech_f, tv_f, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(esc / "meta.json"))
    speech, _ = sf.read(speech_f, dtype="float32")
    tv, _ = sf.read(tv_f, dtype="float32")
    sos = butter(4, [180, 7000], btype="band", fs=SR, output="sos")
    tv = sosfilt(sos, tv).astype(np.float32)          # small loudspeaker
    items = []
    test = [m for m in meta if m["fold"] in (4, 5) and ESC_TO_CAT.get(m["category"])]
    for m in test:
        x, _ = sf.read(esc / f"{m['name']}.flac", dtype="float32")
        x = x[: 5 * SR]
        if rms(x) < 1e-4:
            continue
        for bgname in ("speech", "tv", "quiet"):
            n = 10 * SR
            if bgname == "speech":
                a = rng.integers(0, len(speech) - n); bg = speech[a:a + n].copy()
            elif bgname == "tv":
                a = rng.integers(0, len(tv) - n); bg = tv[a:a + n].copy() * 0.6
            else:
                bg = (rng.standard_normal(n) * 10 ** (-58 / 20)).astype(np.float32)
            ref = active_rms(bg) if bgname != "quiet" else 10 ** (-26 / 20)
            for cond, rel, far in (("0dB", 0, False), ("-10dB", -10, False), ("-20dB", -20, False), ("distant", -10, True)):
                ev = x.copy()
                if far:
                    ev = fftconvolve(ev, rir())[: len(x)].astype(np.float32)
                ev = ev * (ref * 10 ** (rel / 20) / active_rms(ev))
                y = bg.copy()
                y[int(2.5 * SR):int(2.5 * SR) + len(ev)] += ev
                y = np.clip(y, -1, 1)
                name = f"{m['name']}__{bgname}__{cond}"
                sf.write(out / f"{name}.flac", y, SR, subtype="PCM_16")
                items.append({"name": name, "esc": m["category"], "cat": ESC_TO_CAT[m["category"]], "bg": bgname,
                              "cond": cond})
    # backgrounds alone (false positives)
    for i in range(60):
        for bgname, src in (("speech", speech), ("tv", tv)):
            a = rng.integers(0, len(src) - 10 * SR)
            y = src[a:a + 10 * SR] * (0.6 if bgname == "tv" else 1.0)
            name = f"bgonly_{bgname}_{i:02d}"
            sf.write(out / f"{name}.flac", y, SR, subtype="PCM_16")
            items.append({"name": name, "esc": None, "cat": None, "bg": bgname, "cond": "none"})
    # clean events (reference)
    for m in test:
        name = f"{m['name']}__clean"
        x, _ = sf.read(esc / f"{m['name']}.flac", dtype="float32")
        sf.write(out / f"{name}.flac", x, SR, subtype="PCM_16")
        items.append({"name": name, "esc": m["category"], "cat": ESC_TO_CAT[m["category"]], "bg": "none", "cond": "clean"})
    json.dump(items, open(out / "items.json", "w"), indent=0)
    print(len(items), "mixtures")


if __name__ == "__main__":
    main()
