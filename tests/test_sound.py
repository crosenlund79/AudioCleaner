"""Sound analysis (Analyze sounds / Sound Explorer).

Pipeline tests use a small fixed test vocabulary (beeps and music, which the bundled model
recognises reliably in synthetic audio), so they do not depend on the tuned thresholds of the
shipped vocabulary; ``test_shipped_vocabulary`` checks that file separately.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiorefine.hours import HourKey, read_json
from audiorefine.progress import CancelToken
from audiorefine.sound import activities, events as ev_mod, explore, reports, store
from audiorefine.sound import vocab as vocab_mod
from audiorefine.sound.model import SR, Frontend, StreamingMel
from audiorefine.sound.player import OUT_SR, Player, decode_from
from audiorefine.sound.runner import analyze_folder
from conftest import write_audio

TEST_VOCAB = {
    "version": "test", "media_context": 0.5, "media_ids": ["music"],
    "categories": [
        {"id": "beep", "display": "Beeps", "group": "Entertainment and electronics", "kind": "impact",
         "labels": ["Beep, bleep"], "moderate": 0.3, "high": 0.45, "min_windows": 1, "merge_gap": 3.0, "enabled": True},
        {"id": "music", "display": "Music", "group": "Entertainment and electronics", "kind": "media",
         "labels": ["Music"], "moderate": 0.5, "high": 0.7, "min_windows": 2, "merge_gap": 10.0, "enabled": True},
    ],
}


@pytest.fixture
def test_vocab(tmp_path, monkeypatch):
    p = tmp_path / "vocab_test.json"
    p.write_text(json.dumps(TEST_VOCAB))
    monkeypatch.setenv("AUDIOREFINE_SOUND_VOCAB", str(p))
    return p


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


rng = np.random.default_rng(3)


def room(seconds: float) -> np.ndarray:
    return (rng.standard_normal(int(seconds * SR)) * 10 ** (-60 / 20)).astype(np.float32)


def beeps(seconds: float = 3.0) -> np.ndarray:
    x = np.zeros(int(seconds * SR), np.float32)
    n = int(0.15 * SR)
    for k in range(int(seconds / 0.5)):
        a = int(k * 0.5 * SR)
        x[a:a + n] += 0.3 * np.sin(2 * np.pi * 1000 * np.arange(n) / SR)
    return x


def piano(seconds: float = 8.0) -> np.ndarray:
    x = np.zeros(int(seconds * SR), np.float32)
    notes = [261.6, 329.6, 392.0, 523.3, 392.0, 329.6] * 4
    for k, f0 in enumerate(notes):
        a = int(k * 0.5 * SR)
        if a >= len(x):
            break
        n = len(x) - a
        e = np.exp(-np.arange(n) / SR / 0.6)
        x[a:] += (0.08 * e * sum(np.sin(2 * np.pi * f0 * h * np.arange(n) / SR) / h for h in range(1, 7))).astype(np.float32)
    return x


def place(base: np.ndarray, sig: np.ndarray, at: float) -> np.ndarray:
    a = int(at * SR)
    base[a:a + len(sig)] += sig[: max(0, len(base) - a)]
    return base


# ------------------------------------------------------------------------------ front-end
def test_streaming_mel_equals_whole_signal():
    fe = Frontend()
    x = (rng.standard_normal(SR * 3 + 777) * 0.1).astype(np.float32)
    ref = fe.logmel(x)
    for bs in (1000, 4096, SR, 10 ** 7):
        sm = StreamingMel(fe)
        got = np.concatenate([sm.push(x[i:i + bs]) for i in range(0, len(x), bs)] +
                             [sm.push(np.zeros(0, np.float32), final=True)], axis=1)
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < 1e-4


# ------------------------------------------------------------------------------ events
def _vocab(cats, media=("music",), ctx=0.5):
    out = []
    for c in cats:
        out.append(vocab_mod.Category(c["id"], c["id"].title(), "g", c["kind"], ("x",), c["mod"], c["high"],
                                      c.get("k", 1), c.get("gap", 3.0), True, "", (0,)))
    return vocab_mod.Vocabulary("t", tuple(out), ctx, tuple(media), "d")


def test_events_hysteresis_merge_evidence_and_confidence():
    v = _vocab([{"id": "knock", "kind": "impact", "mod": 0.4, "high": 0.7},
                {"id": "dryer", "kind": "sustained", "mod": 0.4, "high": 0.7, "k": 3, "gap": 5.0},
                {"id": "music", "kind": "media", "mod": 0.5, "high": 0.8, "k": 2, "gap": 10.0}])
    n = 120
    starts = np.arange(n, dtype=float)       # 2 s windows every 1 s
    S = np.zeros((n, 3), np.float32)
    S[10, 0] = 0.75                          # knock, high
    S[12, 0] = 0.5                           # second knock 2 s later -> merged, 2 hits
    S[40, 0] = 0.45                          # separate knock, moderate
    S[60:75, 1] = 0.6                        # dryer 15 windows, with a weak dip
    S[66, 1] = 0.25                          # dip above OFF level (0.2) -> still one event
    S[90, 1] = 0.9                           # single strong window: not enough evidence for a sustained sound
    evs, media = ev_mod.build_events(S, starts, v, 2.0, 1.0, float(n + 1))
    k = [e for e in evs if e["category"] == "knock"]
    assert len(k) == 2
    assert k[0]["hits"] == 2 and k[0]["confidence"] == "high"
    assert abs(k[0]["start"] - 10.5) <= 0.5 and k[0]["end"] <= 14.0
    assert k[1]["confidence"] == "moderate"
    d = [e for e in evs if e["category"] == "dryer"]
    assert len(d) == 1 and d[0]["start"] == pytest.approx(60.5, abs=0.6) and d[0]["end"] == pytest.approx(75.5, abs=0.6)
    assert d[0]["confidence"] == "moderate"
    assert media == []


def test_very_frequent_moderate_detections_are_suppressed():
    v = _vocab([{"id": "click", "kind": "impact", "mod": 0.4, "high": 0.7}])
    n = 1800                                  # 30 minutes
    S = np.zeros((n, 1), np.float32)
    S[10:n:60, 0] = 0.5                       # a moderate detection every minute: 60/h
    S[905, 0] = 0.9                           # one clear one
    evs, _ = ev_mod.build_events(S, np.arange(n, dtype=float), v, 2.0, 1.0, float(n + 1))
    assert len(evs) == 1 and evs[0]["confidence"] == "high" and evs[0]["frequent"]


def test_events_during_media_are_flagged_and_not_high():
    v = _vocab([{"id": "laugh", "kind": "vocal", "mod": 0.4, "high": 0.6},
                {"id": "music", "kind": "media", "mod": 0.5, "high": 0.8, "k": 2, "gap": 10.0}])
    n = 60
    S = np.zeros((n, 2), np.float32)
    S[10:40, 1] = 0.7                        # music (TV) playing
    S[20, 0] = 0.9                           # laughter during it
    S[50, 0] = 0.9                           # laughter afterwards
    evs, media = ev_mod.build_events(S, np.arange(n, dtype=float), v, 2.0, 1.0, 61.0)
    laughs = [e for e in evs if e["category"] == "laugh"]
    assert laughs[0]["during_media"] and laughs[0]["confidence"] == "moderate"
    assert not laughs[1]["during_media"] and laughs[1]["confidence"] == "high"
    assert len(media) == 1


# ------------------------------------------------------------------------------ activities
def _ev(cat, t0, t1, conf="moderate", **kw):
    base = datetime(2025, 11, 25, 17, 0, 0).astimezone()
    return {"id": f"{cat}-{t0}", "category": cat, "start": (base + timedelta(seconds=t0)).isoformat(),
            "end": (base + timedelta(seconds=t1)).isoformat(), "confidence": conf, **kw}


def test_activity_needs_several_different_sounds_and_ignores_media():
    a = activities.infer([_ev("frying", 0, 60), _ev("chopping", 120, 130), _ev("dishes", 300, 302, "high")])
    assert [x["category"] for x in a] == ["cooking"]
    assert a[0]["supporting_categories"] == ["chopping", "dishes", "frying"]
    assert activities.infer([_ev("frying", 0, 60), _ev("frying", 100, 160)]) == []          # one kind only
    assert activities.infer([_ev("frying", 0, 60), _ev("chopping", 900, 910)]) == []        # too far apart
    assert activities.infer([_ev("frying", 0, 60), _ev("chopping", 30, 40, during_media=True)]) == []
    w = activities.infer([_ev("tap_water", 0, 40), _ev("dishes", 20, 25)])
    assert [x["category"] for x in w] == ["washing_up"]


# ------------------------------------------------------------------------------ playback
class FakeOut:
    def __init__(self, cb):
        self.cb = cb

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.mark.parametrize("ext,tol_ms", [(".wav", 0.1), (".flac", 0.1), (".m4a", 30), (".mp3", 30)])
def test_player_starts_at_the_requested_position(tmp_path, ext, tol_ms):
    x = (rng.standard_normal(SR * 20) * 0.1).astype(np.float32)
    p = write_audio(tmp_path / f"noise{ext}", x, SR)
    ref = np.concatenate(list(decode_from(p, 0.0)))[:, 0]
    pl = Player(output_factory=FakeOut)
    assert pl.open(p) == pytest.approx(20.0, abs=0.1)
    for start in (0.0, 7.31, 15.0):
        pl.play(start)
        buf = np.zeros((2048, 2), np.float32)
        got = []
        for _ in range(20):
            t_wait = time.time()
            while pl._q.empty() and len(pl._buf) < len(buf) and not pl._eof and time.time() - t_wait < 5:
                time.sleep(0.005)                # under load the decoder may lag; a device would hear silence
            pl._fill(buf)
            got.append(buf[:, 0].copy())
        a = np.concatenate(got)[:OUT_SR // 2]
        s0 = int(start * OUT_SR)
        lo = max(0, s0 - 4800)
        c = np.correlate(ref[lo:s0 + OUT_SR // 2 + 4800], a, mode="valid")
        lag = (int(np.argmax(c)) - (s0 - lo)) / OUT_SR * 1000
        assert abs(lag) <= tol_ms, (ext, start, lag)
        assert pl.position() == pytest.approx(start + 20 * 2048 / OUT_SR, abs=0.01)
    pl.stop()


# ------------------------------------------------------------------------------ whole folder
@pytest.fixture
def sound_folder(tmp_path, test_vocab):
    root = tmp_path / "Lyde ø"
    root.mkdir()
    a = room(70.0)
    place(a, piano(8.0), 10.0)                      # 08:59:40 – 08:59:48
    place(a, beeps(4.0), 35.0)                      # 09:00:05 – 09:00:09 (after the hour boundary)
    sf.write(root / "20251125_085930.wav", a, SR)
    b = place(room(20.0), beeps(3.0), 5.0)          # 10:15:05
    write_audio(root / "20251125_101500.flac", b, SR)
    arch = root / "processed" / "2025-11-24" / "22-00"
    arch.mkdir(parents=True)
    write_audio(arch / "20251124_221000.m4a", place(room(15.0), beeps(3.0), 3.0), SR)   # archived original
    (root / "notes.txt").write_text("x")
    return root


def run(root: Path, token: CancelToken | None = None, events: list | None = None):
    evs = events if events is not None else []
    return analyze_folder(root, evs.append, token or CancelToken())


def master(root: Path) -> dict:
    return read_json(root / "analysis" / store.MASTER_JSON)


def test_analyze_folder_hours_boundaries_and_safety(sound_folder):
    root = sound_folder
    before = {p: sha(p) for p in root.rglob("*") if p.is_file()}
    evs = []
    out = run(root, events=evs)
    assert out.status == "success", out.message
    hours = sorted(f"{d.name}/{h.name}" for d in (root / "analysis").iterdir() if d.is_dir() and d.name[0] == "2"
                   for h in d.iterdir() if h.is_dir())
    assert hours == ["2025-11-24/22-00", "2025-11-25/08-00", "2025-11-25/09-00", "2025-11-25/10-00"]
    assert [e["hour"] for e in evs if e["type"] == "hour_done"] == ["22:00–23:00", "08:00–09:00", "09:00–10:00",
                                                                     "10:00–11:00"]
    # originals: unchanged and not moved; nothing written outside 'analysis'
    after = {p: sha(p) for p in root.rglob("*") if p.is_file() and "analysis" not in p.parts}
    assert after == before
    m = master(root)
    by_hour = {}
    for e in m["events"]:
        by_hour.setdefault((e["date"], e["hour"]), []).append(e)
    h8 = by_hour[("2025-11-25", 8)]
    assert [e["category"] for e in h8] == ["music"]
    # media (music/TV) is recognised from 10 s context windows: start accurate to about ±5 s
    assert "08:59:33" <= datetime.fromisoformat(h8[0]["start"]).strftime("%H:%M:%S") <= "08:59:42"
    h9 = by_hour[("2025-11-25", 9)]
    assert [e["category"] for e in h9] == ["beep"]                    # once, in the hour it starts in
    t9 = datetime.fromisoformat(h9[0]["start"])
    assert "09:00:04" <= t9.strftime("%H:%M:%S") <= "09:00:07"
    assert h9[0]["src_start"] == pytest.approx(35.5, abs=1.0)
    assert h9[0]["source_name"] == "20251125_085930.wav"
    assert [e["category"] for e in by_hour[("2025-11-25", 10)]] == ["beep"]
    arch = by_hour[("2025-11-24", 22)]
    assert arch[0]["source_name"] == "20251124_221000.m4a"
    ids = [e["id"] for e in m["events"]]
    assert len(ids) == len(set(ids))
    assert [e["start"] for e in m["events"]] == sorted(e["start"] for e in m["events"])
    txt = (root / "analysis" / store.MASTER_TXT).read_text(encoding="utf-8")
    assert "Status: COMPLETE" in txt and "Detected sound: Beeps" in txt and "DATE: 2025-11-24" in txt
    assert txt.index("DATE: 2025-11-24") < txt.index("DATE: 2025-11-25")
    assert (root / "analysis" / "2025-11-25" / store.DAY_TXT).is_file()
    assert (root / "analysis" / "2025-11-25" / "09-00" / store.HOUR_TXT).is_file()
    # playback target: the original, a little before the event
    res = explore.load(root)
    tgt = explore.play_target(res, res.by_id[h9[0]["id"]])
    assert tgt.path == root / "20251125_085930.wav"
    assert tgt.start == pytest.approx(h9[0]["src_start"] - explore.PRE_ROLL_S, abs=0.01)


def test_repeat_run_moved_original_and_new_recording(sound_folder):
    root = sound_folder
    assert run(root).status == "success"
    m1 = master(root)
    created = {k: r["created"] for k, r in store.hour_records(root).items()}
    out = run(root)
    assert out.status == "success" and "Nothing new" in out.message
    assert [e["id"] for e in master(root)["events"]] == [e["id"] for e in m1["events"]]
    # transcription archived a recording meanwhile: still found, not analysed again
    dest = root / "processed" / "2025-11-25" / "10-00"
    dest.mkdir(parents=True)
    shutil.move(str(root / "20251125_101500.flac"), dest / "20251125_101500.flac")
    out = run(root)
    assert "Nothing new" in out.message
    res = explore.load(root)
    ev10 = [e for e in res.events if e["hour"] == 10][0]
    assert explore.play_target(res, ev10).path == dest / "20251125_101500.flac"
    # a new recording in an existing hour: only that hour is redone, earlier ids unchanged
    sf.write(root / "20251125_104000.wav", place(room(12.0), beeps(3.0), 4.0), SR)
    out = run(root)
    assert out.hours_done == ["2025-11-25 10"]
    recs = store.hour_records(root)
    assert recs[HourKey("2025-11-25", 8)]["created"] == created[HourKey("2025-11-25", 8)]
    m3 = master(root)
    assert {e["id"] for e in m1["events"]} <= {e["id"] for e in m3["events"]}
    assert len([e for e in m3["events"] if e["hour"] == 10]) == 2


def test_reviews_hide_incorrect_and_survive_reanalysis(sound_folder, test_vocab):
    root = sound_folder
    run(root)
    res = explore.load(root)
    target = [e for e in res.events if e["date"] == "2025-11-24"][0]
    explore.set_review(res, target, "incorrect")
    conf = [e for e in res.events if e["hour"] == 10][0]
    explore.set_review(res, conf, "confirmed")
    v = vocab_mod.load()
    reports.write_combined(root, store.fingerprint(v), v, "COMPLETE")
    txt = (root / "analysis" / store.MASTER_TXT).read_text(encoding="utf-8")
    assert "DATE: 2025-11-24" in txt and "20251124_221000.m4a" not in txt   # hidden from the report
    assert "Reviewed: confirmed by the user" in txt
    res = explore.load(root)
    labels = dict((k, n) for k, n, _ in explore.counts(res))
    assert labels["Beeps"] == 2                                              # incorrect one not counted
    assert dict((k, n) for k, n, _ in explore.counts(res, show_incorrect=True))["Beeps"] == 3
    assert res.by_id[target["id"]]["review"] == "incorrect"                  # machine label kept
    assert res.by_id[target["id"]]["category"] == "beep"
    # a different vocabulary version re-analyses everything, never mixing results
    d = json.loads(test_vocab.read_text())
    d["version"] = "test-2"
    d["categories"][0]["moderate"] = 0.31
    test_vocab.write_text(json.dumps(d))
    out = run(root)
    assert len(out.hours_done) == 4
    fp = master(root)["fingerprint"]
    assert all(store.same_fingerprint(r["fingerprint"], fp) for r in store.hour_records(root).values())
    res = explore.load(root)
    assert any(e["review"] == "incorrect" for e in res.events)               # review found again by time


def test_cancel_after_first_hour_then_resume(sound_folder):
    root = sound_folder
    token = CancelToken()
    evs = []

    def emit(e):
        evs.append(e)
        if e["type"] == "hour_done":
            token.cancel()

    out = analyze_folder(root, emit, token)
    assert out.status == "cancelled" and out.hours_done == ["2025-11-24 22"]
    assert "INTERRUPTED" in (root / "analysis" / store.MASTER_TXT).read_text(encoding="utf-8")
    assert not (root / "analysis" / store.LOCK).exists()
    out = run(root)
    assert out.status == "success" and len(out.hours_done) == 3
    assert "Status: COMPLETE" in (root / "analysis" / store.MASTER_TXT).read_text(encoding="utf-8")


def test_refuses_while_folder_is_being_transcribed(sound_folder):
    root = sound_folder
    lock = root / "refined" / ".audiorefine.lock"
    lock.parent.mkdir()
    import socket
    lock.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "time": 0}))
    out = run(root)
    assert out.status == "failed" and "being processed" in out.message
    assert not (root / "analysis").exists()


def test_annotated_transcript_combines_without_changing_text(sound_folder):
    root = sound_folder
    run(root)
    # a transcription result for 09:00–10:00 (as written by the transcription pipeline)
    a_sha = sha(root / "20251125_085930.wav")
    d = root / "refined" / "2025-11-25" / "09-00"
    d.mkdir(parents=True)
    sf.write(d / "Audio_001.flac", np.zeros(48000 * 40, np.float32), 48000)
    start9 = datetime(2025, 11, 25, 9, 0, 0).astimezone()
    m = {"app": "AudioRefine", "state": "complete", "date": "2025-11-25", "hour": 9,
         "pieces": [{"source": "20251125_085930.wav", "sha256": a_sha, "size": 1, "source_samples": 70 * 48000,
                     "src_start": 30 * 48000, "src_end": 70 * 48000, "out_start": 0, "out_end": 40 * 48000,
                     "start": start9.isoformat(), "time_basis": "file name", "estimated": False}],
         "segments": [{"audio": "Audio_001.flac", "start_sample": 0, "end_sample": 40 * 48000, "samples": 40 * 48000}],
         "entries": [{"t0": 2.0, "t1": 4.0, "text": "Godmorgen, kan du høre det?"},
                     {"t0": 20.0, "t1": 22.0, "text": "Ja, nu er det stille."}],
         "total_samples": 40 * 48000}
    (d / "_AudioRefine_hour.json").write_text(json.dumps(m), encoding="utf-8")
    v = vocab_mod.load()
    data = reports.write_combined(root, store.fingerprint(v), v, "COMPLETE")
    p = reports.write_annotated(root, data)
    txt = p.read_text(encoding="utf-8")
    i1, i2, i3 = txt.index("Godmorgen, kan du høre det?"), txt.index("[Detected sound: Beeps"), txt.index("Ja, nu er det stille.")
    assert i1 < i2 < i3                                   # 09:00:02 speech, 09:00:05 beeps, 09:00:20 speech
    assert "DATE: 2025-11-24" not in txt                  # only hours with a transcript
    # the refined audio reference is known for the event in that hour
    ev9 = [e for e in data["events"] if e["hour"] == 9][0]
    assert ev9["refined"]["file"].endswith("2025-11-25/09-00/Audio_001.flac")
    assert ev9["refined"]["offset"] == pytest.approx(ev9["src_start"] - 30.0, abs=0.01)
    assert not (root / "refined" / "Transcript_Master_Annotated.txt").exists()


def test_shipped_vocabulary_is_consistent():
    os.environ.pop("AUDIOREFINE_SOUND_VOCAB", None)
    v = vocab_mod._load(str(vocab_mod.vocabulary_file()))
    assert v.version != "draft"
    en = v.enabled
    assert len(en) >= 15
    for c in v.categories:
        assert 0.0 < c.moderate < c.high <= 1.0, c.id
        assert c.kind in ("impact", "vocal", "sustained", "media")
        assert c.index, c.id
    assert "speech" not in {c.id for c in en}
    assert set(v.media_ids) <= {c.id for c in v.categories}
