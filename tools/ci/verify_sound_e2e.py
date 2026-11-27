"""Verify a packaged 'analyze' run on the folder made by make_sound_folder.py, and (optionally) the
analysis of the transcription end-to-end folder (archived originals + annotated transcript).

usage: verify_sound_e2e.py SOUND_FOLDER [TRANSCRIBED_FOLDER]      exit 1 on failure"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def secs(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def main():
    root = Path(sys.argv[1]).resolve()
    exp = json.loads(Path(str(root) + ".expect.json").read_text(encoding="utf-8"))
    cats = json.loads((ROOT / "assets" / "sound" / "categories.json").read_text(encoding="utf-8"))
    enabled = {c["id"] for c in cats["categories"] if c["enabled"]}
    a = root / "analysis"
    hours = sorted(f"{d.name}/{h.name}" for d in a.iterdir() if d.is_dir() and d.name[:2] == "20" for h in d.iterdir()
                   if h.is_dir())
    check(hours == exp["hours"], f"hour folders {hours}")
    for h in hours:
        r = json.loads((a / h / "Sound_Events.json").read_text(encoding="utf-8"))
        check(r["state"] == "complete" and (a / h / "Sound_Events.txt").is_file(), f"{h}: hour record complete")
    for rel, hsh in exp["hashes"].items():
        p = root / rel
        check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == hsh, f"{rel} untouched in place")
    m = json.loads((a / "Sound_Events_Master.json").read_text(encoding="utf-8"))
    check(m["status"] == "COMPLETE", "master status COMPLETE")
    evs = m["events"]
    ids = [e["id"] for e in evs]
    check(len(ids) == len(set(ids)), f"no duplicate events ({len(ids)})")
    check([e["start"] for e in evs] == sorted(e["start"] for e in evs), "events in chronological order")
    found = 0
    wanted = [x for x in exp["events"] if x["category"] in enabled]
    for x in wanted:
        hit = [e for e in evs if e["category"] == x["category"] and e["source_name"] == x["source"]
               and e["start"][:10] == x["date"]
               and abs(secs(datetime.fromisoformat(e["start"]).strftime("%H:%M:%S")) - secs(x["clock"])) <= 5]
        found += bool(hit)
        print(f"   {'found' if hit else 'MISSED'} {x['category']} at {x['date']} {x['clock']} in {x['source']}"
              + (f" -> {hit[0]['start'][11:19]}–{hit[0]['end'][11:19]} {hit[0]['confidence']}" if hit else ""))
    check(found >= max(1, int(0.75 * len(wanted))), f"expected sounds found: {found} of {len(wanted)} (>= 75 %)")
    sneeze = [e for e in evs if e["source_name"] == "Optagelse uden tid.flac"]
    check(all(e["estimated"] for e in sneeze), "events of the recording without a time are marked estimated")
    txt = (a / "Sound_Events_Master.txt").read_text(encoding="utf-8")
    check("AUDIOREFINE — ACOUSTIC EVENT REPORT" in txt and "END OF REPORT" in txt, "master report written")
    check(txt.index("DATE: 2025-11-26") < txt.index("DATE: 2025-11-27") < txt.index("DATE: 2025-11-28"),
          "dates in order in the report")
    if sneeze:
        check("≈" in txt, "estimated times marked ≈ in the report")
    for d in ("2025-11-26", "2025-11-27", "2025-11-28"):
        check((a / d / "Sound_Events_Day.txt").is_file(), f"{d}: day report")
    print("\n===== Sound_Events_Master.txt =====\n" + txt[:6000])
    if len(sys.argv) > 2:
        t = Path(sys.argv[2]).resolve()
        ta = t / "analysis"
        tm = json.loads((ta / "Sound_Events_Master.json").read_text(encoding="utf-8"))
        check(tm["status"] == "COMPLETE", "transcribed folder: analysis complete")
        th = sorted(f"{d.name}/{h.name}" for d in ta.iterdir() if d.is_dir() and d.name[:2] == "20" for h in d.iterdir()
                    if h.is_dir())
        rh = sorted(f"{d.name}/{h.name}" for d in (t / "refined").iterdir() if d.is_dir() for h in d.iterdir() if h.is_dir())
        check(th == rh, f"transcribed folder: same hours as the transcription {th}")
        srcs = {s["path"] for s in json.loads((ta / "_AudioRefine_sources.json").read_text(encoding="utf-8"))["sources"].values()}
        check(srcs and all(s.startswith("processed/") for s in srcs), "transcribed folder: archived originals analysed")
        ann = (ta / "Transcript_Master_Annotated.txt").read_text(encoding="utf-8")
        clean = (t / "refined" / "Transcript_Master_Complete.txt").read_text(encoding="utf-8")
        blocks = [b for b in clean.split("\n\n") if b.startswith("[2025-") or b.startswith("[≈2025-")]
        check(blocks and all(b in ann for b in blocks), f"annotated transcript contains all {len(blocks)} transcript "
                                                         "entries unchanged")
        check("refined" not in str(ta / "Transcript_Master_Annotated.txt").split("analysis")[1],
              "annotated transcript stored under analysis")
        print("\n===== Transcript_Master_Annotated.txt (start) =====\n" + ann[:2500])
    if failures:
        print(f"\n{len(failures)} check(s) failed")
        sys.exit(1)
    print("\nAll sound analysis checks passed")


if __name__ == "__main__":
    main()
