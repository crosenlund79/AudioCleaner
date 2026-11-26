"""Verify the result of a packaged end-to-end run (see make_e2e_folder.py). Exit 1 on failure."""
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import soundfile as sf

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def norm(t):
    t = unicodedata.normalize("NFKC", t).lower().replace("-", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def to_s(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))


def main():
    root = Path(sys.argv[1]).resolve()
    expect = json.loads(Path(str(root) + ".expect.json").read_text(encoding="utf-8"))
    batches = [p for p in (root / "refined").iterdir() if p.is_dir() and not p.name.startswith(".")]
    check(len(batches) == 1, f"one batch folder in refined ({[b.name for b in batches]})")
    b = batches[0]
    hidden = [p.name for p in (root / "refined").iterdir() if p.name.startswith(".incomplete")]
    check(not hidden, "no incomplete staging folder left")
    det = json.loads((b / "_AudioRefine_details.json").read_text(encoding="utf-8"))
    check(det["state"] == "complete", f"batch state complete ({det['state']})")
    names = [s["name"] for s in det["sources"]]
    check(names == ["Recording 1.m4a", "Recording 2.mp3", "Recording 10.flac"], f"natural order {names}")
    arch = root / "processed" / b.name
    for n in names:
        p = arch / n
        ok = p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == expect["hashes"][n]
        check(ok and not (root / n).exists(), f"{n} archived unchanged")
    for n in ("broken.wma", "notes.txt"):
        p = root / n
        check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == expect["hashes"][n], f"{n} left untouched")
    total = sum(s["decoded_samples"] for s in det["sources"])
    f1, f2 = sf.info(str(b / "Audio_001.flac")), sf.info(str(b / "Audio_002.flac"))
    check(f1.frames == 600 * 48000, f"Audio_001 exactly 10:00 ({f1.frames} samples)")
    check(f1.frames + f2.frames == total, f"no samples lost or duplicated ({f1.frames + f2.frames} == {total})")
    check(not (b / "Audio_003.flac").exists(), "exactly two segments")
    t2 = (b / "Audio_002.txt").read_text(encoding="utf-8")
    check(t2.startswith("Recording timeline: 00:10:00 – "), "Audio_002.txt uses the continuous timeline")
    srt2 = (b / "Audio_002.srt").read_text(encoding="utf-8")
    cues = re.findall(r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)", srt2)
    check(bool(cues) and to_s(cues[0][0]) < 60, f"Audio_002.srt is relative to its file (first cue {cues[:1]})")
    full = (b / "Transcript_Complete.txt").read_text(encoding="utf-8")
    segs = [(to_s(m.group(1)), to_s(m.group(2)), m.group(3)) for m in
            re.finditer(r"\[(\d\d:\d\d:\d\d\.\d\d\d) – (\d\d:\d\d:\d\d\.\d\d\d)\][^\n]*\n([^\n]*)", full)]
    import jiwer

    for sp in expect["spans"]:
        hyp = " ".join(t for s, e, t in segs if sp["start"] - 0.5 <= (s + e) / 2 < sp["end"] + 0.5)
        w = jiwer.wer(norm(sp["ref"]), norm(hyp)) if hyp else 1.0
        label = f"{sp['lang']} {sp['start']:.1f}-{sp['end']:.1f}s" + (" (crosses 10:00)" if sp.get("crosses") else "")
        print(f"   WER {label}: {w * 100:.1f}%")
        check(w < 0.25, f"word error rate {label} below 25% ({w * 100:.1f}%)")
    print("\n===== Transcript_Complete.txt =====\n" + full)
    print("===== Audio_002.txt =====\n" + t2[:3000])
    if failures:
        print(f"\n{len(failures)} check(s) failed")
        sys.exit(1)
    print("\nAll end-to-end checks passed")


if __name__ == "__main__":
    main()
