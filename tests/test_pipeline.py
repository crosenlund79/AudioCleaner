"""End-to-end pipeline tests on real speech (FLEURS excerpts) with a small Whisper model."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiorefine import batch, pipeline
from audiorefine.constants import DETAILS_NAME, MASTER_SR
from audiorefine.paths import arfilter_path
from audiorefine.progress import CancelToken
from conftest import speech, tone, write_audio

pytestmark = [pytest.mark.model, pytest.mark.skipif(not arfilter_path().is_file(), reason="arfilter not built")]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_folder(tmp_path: Path) -> Path:
    root = tmp_path / "Audio dumps"
    root.mkdir()
    da, sr = speech("da_dk")
    en, _ = speech("en_us")
    write_audio(root / "Recording001.m4a", da[: sr * 30], sr)
    write_audio(root / "Recording002.mp3", np.stack([da[sr * 30:], da[sr * 30:]], 1), sr)
    write_audio(root / "Recording010.wav", en, sr)
    (root / "notes.txt").write_text("unrelated", encoding="utf-8")
    return root


def run(root: Path, model_dir: Path, token: CancelToken | None = None, events=None):
    from audiorefine import log as logsetup

    ev = events if events is not None else []
    log_file = logsetup.setup("process")
    return pipeline.process_folder(root, ev.append, token or CancelToken(), pipeline.Options(model_dir=model_dir),
                                   log_file=log_file)


def test_full_run_outputs_and_archive(tmp_path, model_dir):
    root = make_folder(tmp_path)
    originals = {p.name: sha(p) for p in root.iterdir() if p.suffix != ".txt"}
    out = run(root, model_dir)
    assert out.status == "success", out.message
    bdir = Path(out.refined_dir)
    names = sorted(p.name for p in bdir.iterdir())
    assert names == ["Audio_001.flac", "Audio_001.srt", "Audio_001.txt", "Transcript_Complete.txt",
                     "_AudioRefine_details.json", "_AudioRefine_log.txt"]
    det = json.loads((bdir / DETAILS_NAME).read_text(encoding="utf-8"))
    assert det["state"] == "complete"
    total = sum(s["decoded_samples"] for s in det["sources"])
    info = sf.info(str(bdir / "Audio_001.flac"))
    assert info.frames == total and info.samplerate == MASTER_SR
    # sources are stitched back to back in the right order
    assert [s["name"] for s in det["sources"]] == ["Recording001.m4a", "Recording002.mp3", "Recording010.wav"]
    for a, b in zip(det["sources"], det["sources"][1:]):
        assert a["end_sample"] == b["start_sample"]
    # originals archived unchanged; unrelated files untouched
    pdir = root / "processed" / out.batch
    for n, h in originals.items():
        assert not (root / n).exists()
        assert sha(pdir / n) == h
    assert (root / "notes.txt").read_text(encoding="utf-8") == "unrelated"
    txt = (bdir / "Transcript_Complete.txt").read_text(encoding="utf-8")
    assert "Recording010.wav begins at" in txt
    assert not any(p.name.startswith(".incomplete") for p in (root / "refined").iterdir())
    # the log never contains transcript text
    log = (bdir / "_AudioRefine_log.txt").read_text(encoding="utf-8")
    first_line = [l for l in txt.split("\n\n") if l.startswith("[00:")][0].split("\n", 1)[1]
    assert first_line not in log


def test_repeat_processing_only_new_files(tmp_path, model_dir):
    root = make_folder(tmp_path)
    first = run(root, model_dir)
    assert first.status == "success"
    snapshot = {p: sha(p) for p in (root / "refined" / first.batch).iterdir()}
    da, sr = speech("da_dk")
    write_audio(root / "NewRecording001.wav", da[: sr * 12], sr)
    second = run(root, model_dir)
    assert second.status == "success" and second.batch != first.batch
    det = json.loads((Path(second.refined_dir) / DETAILS_NAME).read_text(encoding="utf-8"))
    assert [s["name"] for s in det["sources"]] == ["NewRecording001.wav"]
    for p, h in snapshot.items():
        assert sha(p) == h  # previous batch untouched
    assert (root / "processed" / second.batch / "NewRecording001.wav").exists()
    assert len(list((root / "processed" / first.batch).iterdir())) == 3
    third = run(root, model_dir)
    assert third.status == "failed" and "No readable audio" in third.message


def test_cancellation_leaves_everything_untouched(tmp_path, model_dir):
    root = make_folder(tmp_path)
    before = {p.name: sha(p) for p in root.iterdir() if p.is_file()}
    token = CancelToken()
    events = []

    def emit(e):
        events.append(e)
        if e.get("stage") == "transcribe":
            token.cancel()

    out = pipeline.process_folder(root, emit, token, pipeline.Options(model_dir=model_dir))
    assert out.status == "cancelled"
    assert {p.name: sha(p) for p in root.iterdir() if p.is_file()} == before
    assert not (root / "processed").exists()
    refined = root / "refined"
    assert [p.name for p in refined.iterdir() if not p.name.startswith(".audiorefine")] == []


def test_failure_is_reported_and_nothing_moved(tmp_path, model_dir, monkeypatch):
    root = make_folder(tmp_path)
    before = {p.name: sha(p) for p in root.iterdir() if p.is_file()}
    monkeypatch.setenv("AUDIOREFINE_ARFILTER", str(tmp_path / "missing-arfilter"))
    out = run(root, model_dir)
    assert out.status == "failed" and "enhancement component is missing" in out.message
    assert {p.name: sha(p) for p in root.iterdir() if p.is_file()} == before
    assert not (root / "processed").exists()
    assert list((root / "refined").glob("2*")) == []


def test_model_failure_is_reported(tmp_path):
    root = make_folder(tmp_path)
    bad = tmp_path / "bad-model"
    bad.mkdir()
    (bad / "model.bin").write_bytes(b"not a model")
    out = run(root, bad)
    assert out.status == "failed"
    assert not (root / "processed").exists()


def test_insufficient_disk_space_fails_early(tmp_path, model_dir, monkeypatch):
    root = make_folder(tmp_path)
    monkeypatch.setattr(pipeline, "_disk_free", lambda p: 1000)
    out = run(root, model_dir)
    assert out.status == "failed" and "disk space" in out.message
    assert not (root / "processed").exists()


def test_crash_leftovers_are_cleaned_and_retry_is_safe(tmp_path, model_dir):
    root = make_folder(tmp_path)
    st = batch.create_staging(root, "2020-01-01_00-00-00")
    (st / "Audio_001.flac").write_bytes(b"half written")
    out = run(root, model_dir)
    assert out.status == "success"
    assert not st.exists()


def test_locked_original_completed_on_next_run(tmp_path, model_dir, monkeypatch):
    root = make_folder(tmp_path)
    real = os.rename

    def flaky(src, dst):
        if Path(src).name == "Recording002.mp3":
            raise PermissionError(32, "in use")
        return real(src, dst)

    monkeypatch.setattr(batch.os, "rename", flaky)
    monkeypatch.setattr(batch.time, "sleep", lambda s: None)
    out = run(root, model_dir)
    assert out.status == "archive_incomplete"
    assert (root / "Recording002.mp3").exists()
    monkeypatch.setattr(batch.os, "rename", real)
    from audiorefine.discovery import scan_folder

    scan = scan_folder(root)
    assert scan.sources == [] and scan.pending
    write_audio(root / "Later.wav", tone(3.0, 16000), 16000)
    out2 = run(root, model_dir)
    assert out2.status == "success"
    assert (root / "processed" / out.batch / "Recording002.mp3").exists()
    det2 = json.loads((Path(out2.refined_dir) / DETAILS_NAME).read_text(encoding="utf-8"))
    assert [s["name"] for s in det2["sources"]] == ["Later.wav"]


@pytest.mark.slow
def test_long_recording_over_ten_minutes(tmp_path, model_dir):
    """A 21-minute recording: three segments of 10, 10 and ~1 minutes, nothing lost."""
    root = tmp_path / "long"
    root.mkdir()
    da, sr = speech("da_dk")
    n = 21 * 60 * sr + 1234
    x = np.zeros(n, np.float32)
    for start in range(0, n - len(da), 5 * 60 * sr):
        x[start:start + len(da)] = da
    x[600 * sr - len(da) // 2: 600 * sr + (len(da) - len(da) // 2)] = da  # speech across the 10:00 boundary
    write_audio(root / "long.flac", x, sr)
    out = run(root, model_dir)
    assert out.status == "success", out.message
    bdir = Path(out.refined_dir)
    frames = [sf.info(str(bdir / f"Audio_{i:03d}.flac")).frames for i in (1, 2, 3)]
    assert frames[:2] == [600 * MASTER_SR] * 2
    assert sum(frames) == n * 3
    t2 = (bdir / "Audio_002.txt").read_text(encoding="utf-8")
    assert t2.startswith("Recording timeline: 00:10:00 – 00:20:00")
