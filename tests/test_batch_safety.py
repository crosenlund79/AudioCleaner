import json
import os
import socket
from datetime import datetime
from pathlib import Path

import pytest

from audiorefine import batch
from audiorefine.constants import DETAILS_NAME, LOCK_NAME, STAGING_MARKER


def make_root(tmp_path, names=("a.wav", "b.wav")):
    root = tmp_path / "root"
    root.mkdir()
    for n in names:
        (root / n).write_bytes(os.urandom(5000))
    return root


def make_details(root, batch_name, names):
    final = root / "refined" / batch_name
    final.mkdir(parents=True)
    srcs = [{"name": n, "size": (root / n).stat().st_size, "sha256": batch.sha256_of(root / n)} for n in names]
    det = {"app": "AudioRefine", "batch": batch_name, "state": "archiving", "sources": srcs}
    batch.write_json_atomic(final / DETAILS_NAME, det)
    return final / DETAILS_NAME, srcs


def test_batch_name_collision_safe(tmp_path):
    root = make_root(tmp_path)
    now = datetime(2026, 1, 2, 3, 4, 5)
    n1 = batch.new_batch_name(root, now)
    assert n1 == "2026-01-02_03-04-05"
    (root / "refined" / n1).mkdir(parents=True)
    n2 = batch.new_batch_name(root, now)
    assert n2 == "2026-01-02_03-04-05_2"
    (root / "processed" / n2).mkdir(parents=True)
    assert batch.new_batch_name(root, now) == "2026-01-02_03-04-05_3"


def test_staging_commit_and_no_overwrite(tmp_path):
    root = make_root(tmp_path)
    st = batch.create_staging(root, "B1")
    assert (st / STAGING_MARKER).is_file() and st.name.startswith(".incomplete-")
    (st / "Audio_001.flac").write_bytes(b"x")
    final = root / "refined" / "B1"
    batch.commit_staging(st, final)
    assert (final / "Audio_001.flac").is_file() and not st.exists()
    assert not (final / STAGING_MARKER).exists()
    st2 = batch.create_staging(root, "B2")
    (root / "refined" / "B2").mkdir()
    with pytest.raises(batch.SafetyError):
        batch.commit_staging(st2, root / "refined" / "B2")


def test_cleanup_only_removes_marked_staging(tmp_path):
    root = make_root(tmp_path)
    st = batch.create_staging(root, "crashed")
    (st / "Audio_001.flac").write_bytes(b"partial")
    user_dir = root / "refined" / ".incomplete-not-ours"
    user_dir.mkdir()
    (user_dir / "keep.txt").write_text("user data")
    removed = batch.cleanup_stale_staging(root)
    assert removed == [st.name]
    assert not st.exists() and (user_dir / "keep.txt").exists()


def test_archive_moves_and_verifies(tmp_path):
    root = make_root(tmp_path, ("Recording001.m4a", "Recording002.m4a"))
    hashes = {n: batch.sha256_of(root / n) for n in ("Recording001.m4a", "Recording002.m4a")}
    det, _ = make_details(root, "B", list(hashes))
    (root / "unrelated.txt").write_text("keep me")
    r = batch.archive_sources(root, "B", det)
    assert r.complete and sorted(r.moved) == sorted(hashes)
    for n, h in hashes.items():
        assert not (root / n).exists()
        assert batch.sha256_of(root / "processed" / "B" / n) == h
    assert (root / "unrelated.txt").read_text() == "keep me"
    info = json.loads(det.read_text())
    assert info["state"] == "complete" and info["archive"]["state"] == "complete"
    assert batch.find_pending_archives(root) == []


def test_archive_never_overwrites_existing(tmp_path):
    root = make_root(tmp_path, ("a.wav",))
    det, _ = make_details(root, "B", ["a.wav"])
    (root / "processed" / "B").mkdir(parents=True)
    (root / "processed" / "B" / "a.wav").write_bytes(b"older file")
    r = batch.archive_sources(root, "B", det)
    assert not r.complete
    assert (root / "a.wav").exists()
    assert (root / "processed" / "B" / "a.wav").read_bytes() == b"older file"


def test_partial_archive_failure_and_recovery(tmp_path, monkeypatch):
    root = make_root(tmp_path, ("a.wav", "b.wav", "c.wav"))
    det, _ = make_details(root, "B", ["a.wav", "b.wav", "c.wav"])
    real_rename = os.rename

    def locked(src, dst):
        if Path(src).name == "b.wav":
            raise PermissionError(32, "The process cannot access the file because it is being used by another process")
        return real_rename(src, dst)

    monkeypatch.setattr(batch.os, "rename", locked)
    monkeypatch.setattr(batch.time, "sleep", lambda s: None)
    r = batch.archive_sources(root, "B", det)
    assert sorted(r.moved) == ["a.wav", "c.wav"] and r.failed[0][0] == "b.wav"
    assert "in use" in r.failed[0][1]
    info = json.loads(det.read_text())
    assert info["state"] == "archive_incomplete"
    pend = batch.find_pending_archives(root)
    assert len(pend) == 1 and pend[0].names == ["b.wav"]
    # the pending file is not offered as a new recording
    from audiorefine.discovery import scan_folder

    assert "b.wav" not in [s.name for s in scan_folder(root).sources]
    monkeypatch.setattr(batch.os, "rename", real_rename)
    r2 = batch.archive_sources(root, "B", det)
    assert r2.complete and (root / "processed" / "B" / "b.wav").exists()
    assert batch.find_pending_archives(root) == []


def test_original_changed_after_processing_is_not_moved(tmp_path):
    root = make_root(tmp_path, ("a.wav",))
    det, _ = make_details(root, "B", ["a.wav"])
    data = bytearray((root / "a.wav").read_bytes())
    data[100] ^= 0xFF
    (root / "a.wav").write_bytes(bytes(data))  # same size, different content
    r = batch.archive_sources(root, "B", det)
    assert not r.complete and (root / "a.wav").exists()


def test_archive_resumes_after_crash_between_move_and_record(tmp_path):
    root = make_root(tmp_path, ("a.wav",))
    det, _ = make_details(root, "B", ["a.wav"])
    (root / "processed" / "B").mkdir(parents=True)
    os.rename(root / "a.wav", root / "processed" / "B" / "a.wav")  # moved, but not yet recorded
    r = batch.archive_sources(root, "B", det)
    assert r.complete


def test_folder_lock(tmp_path):
    root = make_root(tmp_path)
    with batch.FolderLock(root):
        assert batch.folder_busy(root)
        with pytest.raises(batch.SafetyError):
            batch.FolderLock(root).acquire()
    assert not batch.folder_busy(root)


def test_stale_lock_is_recovered(tmp_path):
    root = make_root(tmp_path)
    (root / "refined").mkdir()
    (root / "refined" / LOCK_NAME).write_text(json.dumps({"pid": 999999999, "host": socket.gethostname(), "time": 0}))
    assert not batch.folder_busy(root)
    with batch.FolderLock(root):
        pass


def test_stale_work_dirs_removed(tmp_path):
    wr = tmp_path / "work"
    (wr / "dead").mkdir(parents=True)
    batch.write_json_atomic(wr / "dead" / "owner.json", {"pid": 999999999, "host": socket.gethostname()})
    (wr / "dead" / "master48.f32").write_bytes(b"x" * 100)
    (wr / "alive").mkdir()
    batch.write_json_atomic(wr / "alive" / "owner.json", {"pid": os.getpid(), "host": socket.gethostname()})
    assert batch.cleanup_stale_work(wr) == ["dead"]
    assert (wr / "alive").exists() and not (wr / "dead").exists()


def test_scan_command_cleans_after_killed_run(tmp_path):
    root = make_root(tmp_path)
    st = batch.create_staging(root, "killed")
    (root / "refined" / LOCK_NAME).write_text(json.dumps({"pid": 999999999, "host": socket.gethostname()}))
    from audiorefine import cli

    assert cli.main(["--json", "scan", str(root)]) == 0
    assert not st.exists() and not (root / "refined" / LOCK_NAME).exists()
