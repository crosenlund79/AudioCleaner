"""Verify a packaged end-to-end run in the hourly layout (see make_e2e_folder.py). Exit 1 on failure."""
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta
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


def main():
    root = Path(sys.argv[1]).resolve()
    exp = json.loads(Path(str(root) + ".expect.json").read_text(encoding="utf-8"))
    refined, processed = root / "refined", root / "processed"
    hours = sorted(f"{d.name}/{h.name}" for d in refined.iterdir() if d.is_dir() for h in d.iterdir() if h.is_dir())
    check(hours == exp["hours"], f"hour folders {hours}")
    check(not list(refined.rglob(".incomplete-*")) and not list(refined.rglob(".replaced-*")), "no leftover staging")
    for h in exp["hours"]:
        d = refined / h
        m = json.loads((d / "_AudioRefine_hour.json").read_text(encoding="utf-8"))
        check(m["state"] == "complete", f"{h}: manifest complete")
        for s in m["segments"]:
            info = sf.info(str(d / s["audio"]))
            check(info.frames == s["samples"] and info.samplerate == 48000, f"{h}/{s['audio']}: {info.frames} samples")
            check((d / s["txt"]).is_file() and (d / s["srt"]).is_file(), f"{h}/{s['audio']}: txt and srt present")
        check(sum(s["samples"] for s in m["segments"]) == m["total_samples"], f"{h}: no samples lost")
        check((d / "Transcript_Hour_Complete.txt").read_text(encoding="utf-8").startswith("AudioRefine – hour transcript"),
              f"{h}: hour transcript")
    h10 = json.loads((refined / "2025-11-25/10-00/_AudioRefine_hour.json").read_text(encoding="utf-8"))
    check([s["samples"] for s in h10["segments"]] == [600 * 48000, h10["total_samples"] - 600 * 48000],
          "10-00: Audio_001 exactly 10:00, Audio_002 the rest")
    for d in ("2025-11-25", "2025-11-26"):
        check((refined / d / "Transcript_Day_Complete.txt").is_file(), f"{d}: day transcript")
    for name, rel in exp["processed"].items():
        p = processed / rel / name
        ok = p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == exp["hashes"][name] and not (root / name).exists()
        check(ok, f"{name} archived unchanged in processed/{rel}")
    for n in ("broken.wma", "notes.txt"):
        p = root / n
        check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == exp["hashes"][n], f"{n} left untouched")
    master = (refined / "Transcript_Master_Complete.txt").read_text(encoding="utf-8")
    heads = re.findall(r"DATE: \S+|HOUR: \S+", master)
    check(heads == ["DATE: 2025-11-25", "HOUR: 08:00–09:00", "HOUR: 09:00–10:00", "HOUR: 10:00–11:00",
                    "DATE: 2025-11-26", "HOUR: 07:00–08:00", "HOUR: 14:00–15:00"], f"master order {heads}")
    check("Status: COMPLETE" in master and "END OF TRANSCRIPT" in master, "master complete")
    entries = re.findall(r"^\[(≈?)(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| (Audio_\d+) (\d\d:\d\d:\d\d)\][^\n]*\n([^\n]*)", master, re.M)
    lines = re.findall(r"^\[.*\].*$", master, re.M)
    check(len(lines) == len(set(lines)), "no duplicated entries in the master transcript")
    stamps = [e[1] for e in entries]
    check(stamps == sorted(stamps), "master entries in chronological order")
    check(any(e[0] == "≈" and e[1].startswith("2025-11-26 14:32") for e in entries), "estimated time marked with ≈")
    import jiwer

    for sp in exp["spans"]:
        a = datetime.strptime(sp["start"], "%Y-%m-%d %H:%M:%S")
        b = a + timedelta(seconds=sp["seconds"])
        hyp = " ".join(t for _, ts, _, _, t in entries
                       if a - timedelta(seconds=1) <= datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") < b + timedelta(seconds=1))
        hyp = hyp.replace("…", " ")
        w = jiwer.wer(norm(sp["ref"]), norm(hyp)) if hyp else 1.0
        print(f"   WER {sp['label']}: {w * 100:.1f}%")
        check(w < 0.25, f"word error rate {sp['label']} below 25% ({w * 100:.1f}%)")
    print("\n===== Transcript_Master_Complete.txt =====\n" + master)
    if failures:
        print(f"\n{len(failures)} check(s) failed")
        sys.exit(1)
    print("\nAll end-to-end checks passed")


if __name__ == "__main__":
    main()
