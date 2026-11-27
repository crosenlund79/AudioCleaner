"""End-to-end tests of the chronological, hour-by-hour processing (real speech, small Whisper model)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiorefine import hourbatch, pipeline, segments as segmod
from audiorefine.constants import MASTER_SR
from audiorefine.hours import DAY_TRANSCRIPT, HOUR_MANIFEST, HOUR_TRANSCRIPT, MASTER_TRANSCRIPT
from audiorefine.paths import arfilter_path
from audiorefine.progress import CancelToken
from conftest import speech, tone, write_audio

pytestmark = [pytest.mark.model, pytest.mark.skipif(not arfilter_path().is_file(), reason="arfilter not built")]

RNG = np.random.default_rng(11)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rec(seconds: float, clip: np.ndarray | None = None, at: float = 2.0, sr: int = 16000) -> np.ndarray:
    x = (RNG.standard_normal(int(seconds * sr)) * 0.002).astype(np.float32)
    if clip is not None:
        a = int(at * sr)
        n = min(len(clip), len(x) - a)
        x[a:a + n] += clip[:n]
    return x


def run(root: Path, model_dir: Path, token: CancelToken | None = None, events=None):
    from audiorefine import log as logsetup

    ev = events if events is not None else []
    log_file = logsetup.setup("process")
    return pipeline.process_folder(root, ev.append, token or CancelToken(), pipeline.Options(model_dir=model_dir),
                                   log_file=log_file)


def master_text(root: Path) -> str:
    return (root / "refined" / MASTER_TRANSCRIPT).read_text(encoding="utf-8")


def entry_lines(txt: str) -> list[str]:
    return re.findall(r"^\[.*\]$|^\[.*\] \[uncertain\]$", txt, re.M)


@pytest.fixture
def multiday(tmp_path):
    da, sr = speech("da_dk")
    en, _ = speech("en_us")
    root = tmp_path / "Audio dumps"
    root.mkdir()
    write_audio(root / "20251125_080400.m4a", rec(40, da[: sr * 30]), sr)          # 08:04
    write_audio(root / "20251125_083000.mp3", rec(35, en[: sr * 25]), sr)          # 08:30 (same hour)
    write_audio(root / "20251125_085930.wav", rec(70, da[: sr * 40], at=10), sr)   # 08:59:30 -> crosses 09:00
    write_audio(root / "20251125_112000.flac", rec(30, en[: sr * 20]), sr)        # 11:20 (10:00 missing)
    write_audio(root / "20251125_235940.m4a", rec(50, en[: sr * 30], at=5), sr)   # crosses midnight
    write_audio(root / "20251126_071200.m4a", rec(30, da[sr * 40:]), sr)           # next day
    (root / "notes.txt").write_text("unrelated", encoding="utf-8")
    return root


def test_multi_day_hourly_structure_and_transcripts(multiday, model_dir):
    root = multiday
    originals = {p.name: sha(p) for p in root.iterdir() if p.suffix != ".txt"}
    events = []
    out = run(root, model_dir, events=events)
    assert out.status == "success", out.message
    refined = root / "refined"
    hours = sorted(str(p.relative_to(refined)).replace("\\", "/") for p in refined.glob("*/*") if p.is_dir())
    assert hours == ["2025-11-25/08-00", "2025-11-25/09-00", "2025-11-25/11-00", "2025-11-25/23-00",
                     "2025-11-26/00-00", "2025-11-26/07-00"]          # only hours with recordings
    # hours were finished one after the other, oldest first
    done = [(e["date"], e["hour"]) for e in events if e.get("type") == "hour_done"]
    assert done == sorted(done) and len(done) == 6
    for h in refined.glob("*/*"):
        if not h.is_dir():
            continue
        names = sorted(p.name for p in h.iterdir())
        assert names == ["Audio_001.flac", "Audio_001.srt", "Audio_001.txt", HOUR_TRANSCRIPT,
                         HOUR_MANIFEST, "_AudioRefine_words.json"]
    for d in ("2025-11-25", "2025-11-26"):
        assert (refined / d / DAY_TRANSCRIPT).is_file()
    # originals archived by the hour in which they started; unrelated file untouched
    proc = sorted(str(p.relative_to(root / "processed")).replace("\\", "/") for p in (root / "processed").rglob("*") if p.is_file())
    assert proc == ["2025-11-25/08-00/20251125_080400.m4a", "2025-11-25/08-00/20251125_083000.mp3",
                    "2025-11-25/08-00/20251125_085930.wav", "2025-11-25/11-00/20251125_112000.flac",
                    "2025-11-25/23-00/20251125_235940.m4a", "2025-11-26/07-00/20251126_071200.m4a"]
    for rel in proc:
        assert sha(root / "processed" / rel) == originals[Path(rel).name]
    assert (root / "notes.txt").read_text(encoding="utf-8") == "unrelated"
    # every sample of every recording is in exactly one hour
    pieces = {}
    for m in (json.loads(p.read_text(encoding="utf-8")) for p in refined.glob(f"*/*/{HOUR_MANIFEST}")):
        assert sum(p["out_end"] - p["out_start"] for p in m["pieces"]) == m["total_samples"]
        for p in m["pieces"]:
            pieces.setdefault(p["source"], []).append((p["src_start"], p["src_end"], p["source_samples"]))
    for name, ps in pieces.items():
        ps.sort()
        assert ps[0][0] == 0 and ps[-1][1] == ps[-1][2], name
        assert all(a[1] == b[0] for a, b in zip(ps, ps[1:])), name
    cross = sorted(pieces["20251125_085930.wav"])
    assert len(cross) == 2 and cross[0][1] == 30 * MASTER_SR        # split exactly at 09:00:00
    assert len(pieces["20251125_235940.m4a"]) == 2                   # split at midnight
    # master: chronological dates and hours, clock times, no duplicates
    txt = master_text(root)
    heads = re.findall(r"DATE: \S+|HOUR: \S+", txt)
    assert heads == ["DATE: 2025-11-25", "HOUR: 08:00–09:00", "HOUR: 09:00–10:00", "HOUR: 11:00–12:00",
                     "HOUR: 23:00–24:00", "DATE: 2025-11-26", "HOUR: 00:00–01:00", "HOUR: 07:00–08:00"]
    assert "Status: COMPLETE" in txt and "Hourly groups included: 6" in txt
    stamps = re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \|", txt, re.M)
    assert stamps == sorted(stamps) and len(stamps) >= 6
    assert len(entry_lines(txt)) == len(set(entry_lines(txt)))
    assert "2025-11-25 09:00:" in txt and "2025-11-26 00:00:" in txt
    # SRT relative to its own file; hour transcript references the audio file
    h9 = refined / "2025-11-25" / "09-00"
    srt = (h9 / "Audio_001.srt").read_text(encoding="utf-8")
    first = re.search(r"(\d\d):(\d\d):(\d\d),\d{3} -->", srt)
    assert first and int(first.group(1)) == 0 and int(first.group(2)) == 0
    assert "| Audio_001 00:00:" in (h9 / HOUR_TRANSCRIPT).read_text(encoding="utf-8")
    assert (h9 / HOUR_TRANSCRIPT).read_text(encoding="utf-8").startswith("AudioRefine – hour transcript\nRecording date: 2025-11-25\nRecording hour: 09:00–10:00")
    status = json.loads((refined / "_AudioRefine_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete"


def test_ten_minute_segments_inside_an_hour(tmp_path, model_dir, monkeypatch):
    """Same code path with 10-second 'segments' so the test is fast: 23 s of recordings -> 10 + 10 + 3 s."""
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 10)
    from audiorefine import hourtext

    monkeypatch.setattr(hourtext, "SEGMENT_SECONDS", 10)
    orig = segmod.plan_segments
    monkeypatch.setattr(pipeline.segments, "plan_segments", lambda total: orig(total, 10 * MASTER_SR))
    root = tmp_path / "f"
    root.mkdir()
    da, sr = speech("da_dk")
    write_audio(root / "20251125_100500.wav", da[: sr * 12], sr)
    write_audio(root / "20251125_102000.wav", da[sr * 12: sr * 23], sr)
    out = run(root, model_dir)
    assert out.status == "success", out.message
    h = root / "refined" / "2025-11-25" / "10-00"
    frames = [sf.info(str(h / f"Audio_{i:03d}.flac")).frames for i in (1, 2, 3)]
    assert frames == [10 * MASTER_SR, 10 * MASTER_SR, 3 * MASTER_SR]   # no silence inserted, nothing lost
    assert not (h / "Audio_004.flac").exists()


def test_cancel_during_hour_then_resume(multiday, model_dir):
    root = multiday
    token = CancelToken()
    events = []

    def emit(e):
        events.append(e)
        if e.get("type") == "progress" and e.get("current_hour") == "09:00–10:00" and e.get("label") == "Transcribing speech":
            token.cancel()

    out = pipeline.process_folder(root, emit, token, pipeline.Options(model_dir=model_dir))
    assert out.status == "cancelled"
    refined = root / "refined"
    assert (refined / "2025-11-25" / "08-00" / HOUR_MANIFEST).is_file()       # finished hour kept
    assert not (refined / "2025-11-25" / "09-00").exists()                     # unfinished hour not left behind
    assert not [p for p in refined.rglob(".incomplete-*")]
    txt = master_text(root)
    assert "INTERRUPTED" in txt and "HOUR: 08:00–09:00" in txt and "HOUR: 09:00–10:00" not in txt
    # the recording crossing 09:00 is NOT archived although its first part is finished
    assert (root / "20251125_085930.wav").exists()
    assert (root / "processed" / "2025-11-25" / "08-00" / "20251125_080400.m4a").exists()
    assert (root / "20251125_112000.flac").exists()                             # untouched, not processed yet
    first_08 = (refined / "2025-11-25" / "08-00" / HOUR_TRANSCRIPT).read_text(encoding="utf-8")
    out2 = run(root, model_dir)                                                  # resume
    assert out2.status == "success", out2.message
    assert "2025-11-25 08" not in out2.hours_done                                # not redone
    assert (refined / "2025-11-25" / "08-00" / HOUR_TRANSCRIPT).read_text(encoding="utf-8") == first_08
    assert (root / "processed" / "2025-11-25" / "08-00" / "20251125_085930.wav").exists()
    txt = master_text(root)
    assert "Status: COMPLETE" in txt and len(entry_lines(txt)) == len(set(entry_lines(txt)))
    assert len(re.findall(r"HOUR: ", txt)) == 6


def test_late_recordings_and_older_date(multiday, model_dir):
    root = multiday
    assert run(root, model_dir).status == "success"
    h8 = root / "refined" / "2025-11-25" / "08-00"
    m_old = json.loads((h8 / HOUR_MANIFEST).read_text(encoding="utf-8"))
    old_audio = np.concatenate([sf.read(str(h8 / s["audio"]), dtype="int16")[0] for s in m_old["segments"]])
    old_pos = {p["source"]: (p["out_start"], p["out_end"]) for p in m_old["pieces"]}
    before = entry_lines(master_text(root))
    en, sr = speech("en_us")
    write_audio(root / "20251125_084500.m4a", rec(30, en[sr * 25: sr * 40]), sr)   # into the finished 08-00 hour
    write_audio(root / "20251124_150000.wav", rec(25, en[: sr * 20]), sr)          # older date
    out = run(root, model_dir)
    assert out.status == "success", out.message
    assert out.hours_done == ["2025-11-24 15", "2025-11-25 08"]                     # chronological, only affected hours
    m = json.loads((h8 / HOUR_MANIFEST).read_text(encoding="utf-8"))
    assert [p["source"] for p in m["pieces"]] == ["20251125_080400.m4a", "20251125_083000.mp3",
                                                  "20251125_084500.m4a", "20251125_085930.wav"]
    new_audio = np.concatenate([sf.read(str(h8 / s["audio"]), dtype="int16")[0] for s in m["segments"]])
    for p in m["pieces"]:
        if p["source"] in old_pos:      # earlier audio reused bit-exactly, not re-enhanced
            a, b = old_pos[p["source"]]
            assert np.array_equal(new_audio[p["out_start"]:p["out_end"]], old_audio[a:b])
    txt = master_text(root)
    heads = re.findall(r"DATE: \S+", txt)
    assert heads == ["DATE: 2025-11-24", "DATE: 2025-11-25", "DATE: 2025-11-26"]
    after = entry_lines(txt)
    assert len(after) == len(set(after))
    assert set(before) <= set(after) | {l for l in before if "08:" in l}            # earlier entries are still there
    assert (root / "processed" / "2025-11-25" / "08-00" / "20251125_084500.m4a").exists()


def test_repeat_without_new_recordings_and_duplicate_copy(multiday, model_dir):
    root = multiday
    assert run(root, model_dir).status == "success"
    first = master_text(root)
    out = run(root, model_dir)
    assert out.status == "success" and "Nothing new" in out.message
    assert entry_lines(master_text(root)) == entry_lines(first)
    # a copy of an already processed recording is put back into the folder
    shutil.copy2(root / "processed" / "2025-11-25" / "08-00" / "20251125_080400.m4a", root / "copy again.m4a")
    out = run(root, model_dir)
    assert out.status == "success" and "Nothing new" in out.message
    assert any("already processed" in w for w in out.warnings)
    assert (root / "copy again.m4a").exists()
    assert entry_lines(master_text(root)) == entry_lines(first)


def test_repairs_combined_transcripts_and_archives_pending(multiday, model_dir):
    root = multiday
    assert run(root, model_dir).status == "success"
    refined = root / "refined"
    reference = entry_lines(master_text(root))
    # simulate a crash after an hour was finalized but before day/master update and archiving
    (refined / "2025-11-26" / DAY_TRANSCRIPT).unlink()
    (refined / MASTER_TRANSCRIPT).write_text("half written", encoding="utf-8")
    p = root / "processed" / "2025-11-26" / "07-00" / "20251126_071200.m4a"
    shutil.move(str(p), str(root / p.name))
    jpath = refined / "_AudioRefine_archive.json"           # the crash happened before the archive record
    j = json.loads(jpath.read_text(encoding="utf-8"))
    j["archived"] = [r for r in j["archived"] if r["name"] != p.name]
    jpath.write_text(json.dumps(j), encoding="utf-8")
    out = run(root, model_dir)
    assert out.status == "success", out.message
    assert (refined / "2025-11-26" / DAY_TRANSCRIPT).is_file()
    assert entry_lines(master_text(root)) == reference
    assert p.exists() and not (root / p.name).exists()      # archived instead of processed again
    assert out.hours_done == []


def test_recording_time_without_timestamp_is_estimated(tmp_path, model_dir):
    root = tmp_path / "f"
    root.mkdir()
    da, sr = speech("da_dk")
    write_audio(root / "Voice 001.m4a", rec(20, da[: sr * 15]), sr)
    t = time.mktime((2025, 11, 26, 14, 32, 40, 0, 0, -1))
    os.utime(root / "Voice 001.m4a", (t, t))
    out = run(root, model_dir)
    assert out.status == "success", out.message
    h = root / "refined" / "2025-11-26" / "14-00"
    txt = (h / HOUR_TRANSCRIPT).read_text(encoding="utf-8")
    assert "≈2025-11-26 14:32:" in txt and "estimated" in txt
    assert (root / "processed" / "2025-11-26" / "14-00" / "Voice 001.m4a").exists()


def test_failure_keeps_everything(multiday, model_dir, monkeypatch):
    root = multiday
    before = {p.name: sha(p) for p in root.iterdir() if p.is_file()}
    monkeypatch.setenv("AUDIOREFINE_ARFILTER", str(Path(root) / "missing-arfilter"))
    out = run(root, model_dir)
    assert out.status == "failed" and "enhancement component is missing" in out.message
    assert {p.name: sha(p) for p in root.iterdir() if p.is_file()} == before
    assert not (root / "processed").exists()
    assert not list((root / "refined").glob("*/*-00"))


def test_insufficient_disk_space_fails_early(multiday, model_dir, monkeypatch):
    monkeypatch.setattr(pipeline, "_disk_free", lambda p: 1000)
    out = run(multiday, model_dir)
    assert out.status == "failed" and "disk space" in out.message
    assert not (multiday / "processed").exists()


def test_unicode_folder_and_file_names(tmp_path, model_dir):
    root = tmp_path / "Optagelser æøå ÆØÅ"
    root.mkdir()
    da, sr = speech("da_dk")
    write_audio(root / "Møde på kontoret – 20251125_101500.m4a", da[: sr * 15], sr)
    out = run(root, model_dir)
    assert out.status == "success", out.message
    assert "Møde på kontoret – 20251125_101500.m4a" in master_text(root)
    assert (root / "processed" / "2025-11-25" / "10-00" / "Møde på kontoret – 20251125_101500.m4a").exists()


def test_engine_killed_mid_hour_then_retry(multiday, model_dir):
    """Simulates a crash / power loss while an hour is being processed."""
    import subprocess
    import sys

    root = multiday
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), AUDIOREFINE_MODEL_DIR=str(model_dir))
    proc = subprocess.Popen([sys.executable, "-m", "audiorefine.cli", "--json", "process", str(root)],
                            stdout=subprocess.PIPE, stdin=subprocess.PIPE, env=env, text=True, encoding="utf-8")
    t0 = time.time()
    for line in proc.stdout:
        if '"current_hour": "09:00' in line and "Enhancing" in line or time.time() - t0 > 300:
            break
    proc.kill()
    proc.wait()
    refined = root / "refined"
    assert (refined / "2025-11-25" / "08-00" / HOUR_MANIFEST).is_file()
    assert (root / "20251125_085930.wav").exists()                  # spans 08 and 09: not archived yet
    from audiorefine import cli

    assert cli.main(["--json", "scan", str(root)]) == 0              # scan tidies up the killed hour
    assert not list(refined.rglob(".incomplete-*"))
    out = run(root, model_dir)
    assert out.status == "success", out.message
    assert out.hours_done[0] == "2025-11-25 09"
    assert len(entry_lines(master_text(root))) == len(set(entry_lines(master_text(root))))
