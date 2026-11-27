"""Recording time detection, hour splitting and hour-folder recovery (no speech model needed)."""
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from audiorefine import hourbatch, rectime
from audiorefine.constants import MASTER_SR, STAGING_MARKER
from audiorefine.hours import HOUR_MANIFEST, HourKey, hour_key_of, split_into_hours, write_json_atomic


def info(name="x.m4a", duration=60.0, mtime=None, birthtime=None, meta=None):
    return SimpleNamespace(name=name, path=Path(name), duration=duration,
                           mtime=mtime if mtime is not None else time.time(), birthtime=birthtime,
                           meta_datetime=meta, meta_datetime_tag="creation_time",
                           name_datetime=rectime.parse_filename_datetime(Path(name).stem))


def test_priority_filename_then_metadata():
    meta = datetime(2025, 11, 25, 7, 0, 0, tzinfo=timezone.utc)
    r = rectime.detect(info("20251125_080412.m4a", meta=meta))
    assert r.basis == rectime.BASIS_NAME and not r.estimated
    assert rectime.local(r.start).strftime("%Y-%m-%d %H:%M:%S") == "2025-11-25 08:04:12"
    r = rectime.detect(info("Voice 001.m4a", meta=meta))
    assert r.basis == rectime.BASIS_META and not r.estimated
    assert r.start == meta                                   # UTC instant kept, shown in local time


def test_filesystem_fallbacks_are_estimated():
    m = time.mktime((2025, 11, 26, 14, 32, 40, 0, 0, -1))
    r = rectime.detect(info("Voice 001.m4a", duration=20, mtime=m))
    assert r.basis == rectime.BASIS_MODIFIED and r.estimated
    assert rectime.local(r.start).strftime("%H:%M:%S") == "14:32:20"
    # creation time consistent with modification time minus duration -> used
    r = rectime.detect(info("Voice 002.m4a", duration=600, mtime=m, birthtime=m - 601))
    assert r.basis == rectime.BASIS_CREATED and r.estimated
    # creation time set by copying (later than modification) -> ignored
    r = rectime.detect(info("Voice 003.m4a", duration=600, mtime=m, birthtime=m + 86400))
    assert r.basis == rectime.BASIS_MODIFIED


def test_implausible_metadata_ignored():
    r = rectime.detect(info("Voice.m4a", meta=datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)))
    assert r.basis == rectime.BASIS_MODIFIED
    r = rectime.detect(info("Voice.m4a", meta=datetime.now(timezone.utc) + timedelta(days=30)))
    assert r.basis == rectime.BASIS_MODIFIED


def test_split_at_hour_boundaries_exact():
    start = datetime(2025, 11, 25, 8, 59, 30).astimezone()
    ps = split_into_hours(start, 90 * MASTER_SR)
    assert [str(p.key) for p in ps] == ["2025-11-25 08", "2025-11-25 09"]
    assert ps[0].src_end == 30 * MASTER_SR and ps[1].src_start == 30 * MASTER_SR and ps[1].src_end == 90 * MASTER_SR
    # several hours, midnight, nothing lost
    start = datetime(2025, 11, 25, 22, 45, 0).astimezone()
    ps = split_into_hours(start, int(2.5 * 3600 * MASTER_SR) + 7)
    assert [str(p.key) for p in ps] == ["2025-11-25 22", "2025-11-25 23", "2025-11-26 00", "2025-11-26 01"]
    assert ps[0].src_start == 0 and ps[-1].src_end == int(2.5 * 3600 * MASTER_SR) + 7
    assert all(a.src_end == b.src_start for a, b in zip(ps, ps[1:]))
    assert ps[1].samples == 3600 * MASTER_SR
    # entirely inside one hour
    ps = split_into_hours(datetime(2025, 11, 25, 8, 4).astimezone(), 60 * MASTER_SR)
    assert len(ps) == 1 and ps[0].key == HourKey("2025-11-25", 8)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset to switch time zone")
def test_daylight_saving_transitions(monkeypatch):
    monkeypatch.setenv("TZ", "Europe/Copenhagen")
    time.tzset()
    try:
        # autumn: 02:00-03:00 happens twice; a recording across it stays in one '02-00' folder
        start = datetime(2025, 10, 26, 0, 30, tzinfo=timezone.utc)      # 02:30 CEST
        ps = split_into_hours(start, 3600 * MASTER_SR)                  # until 02:30 CET
        assert [str(p.key) for p in ps] == ["2025-10-26 02"] and ps[0].samples == 3600 * MASTER_SR
        # spring: 02:00-03:00 does not exist; 01:30 + 1 h -> 01-00 then 03-00
        start = datetime(2025, 3, 30, 0, 30, tzinfo=timezone.utc)       # 01:30 CET
        ps = split_into_hours(start, 3600 * MASTER_SR)
        assert [str(p.key) for p in ps] == ["2025-03-30 01", "2025-03-30 03"]
        assert ps[0].samples == 1800 * MASTER_SR
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def _hour(root, date, folder, complete=True, marker=False, name=None):
    d = root / "refined" / date / (name or folder)
    d.mkdir(parents=True)
    write_json_atomic(d / HOUR_MANIFEST, {"app": "AudioRefine", "state": "complete" if complete else "writing",
                                          "date": date, "hour": int(folder[:2]), "segments": [], "pieces": []})
    if marker:
        (d / STAGING_MARKER).write_text("x")
    return d


def test_recover_interrupted_replacement(tmp_path):
    root = tmp_path
    date = root / "refined" / "2025-11-25"
    # crash after the old folder was moved aside and before the new one was renamed: new is complete -> used
    _hour(root, "2025-11-25", "08-00", name=".replaced-08-00")
    _hour(root, "2025-11-25", "08-00", name=".incomplete-08-00")
    write_json_atomic(date / ".replace-08-00.json", {"staging": ".incomplete-08-00", "old": ".replaced-08-00", "final": "08-00"})
    notes = hourbatch.recover(root)
    assert (date / "08-00" / HOUR_MANIFEST).is_file() and not (date / ".replaced-08-00").exists()
    assert not (date / ".replace-08-00.json").exists() and notes
    # crash in the same place, but the new folder was not finished -> old one restored
    _hour(root, "2025-11-25", "09-00", name=".replaced-09-00")
    _hour(root, "2025-11-25", "09-00", complete=False, marker=True, name=".incomplete-09-00")
    write_json_atomic(date / ".replace-09-00.json", {"staging": ".incomplete-09-00", "old": ".replaced-09-00", "final": "09-00"})
    hourbatch.recover(root)
    assert (date / "09-00" / HOUR_MANIFEST).is_file()
    assert not list(date.glob(".*"))
    # an abandoned new hour is removed
    _hour(root, "2025-11-25", "10-00", complete=False, marker=True, name=".incomplete-10-00")
    hourbatch.recover(root)
    assert not (date / ".incomplete-10-00").exists()


def test_archive_never_overwrites(tmp_path):
    root = tmp_path
    src = root / "Rec.m4a"
    src.write_bytes(os.urandom(2000))
    key = HourKey("2025-11-25", 8)
    dest = key.processed_dir(root)
    dest.mkdir(parents=True)
    (dest / "Rec.m4a").write_bytes(b"an older recording with the same name")
    from audiorefine.batch import sha256_of

    ok, msg = hourbatch.archive_original(root, src, sha256_of(src), src.stat().st_size, key)
    assert ok and msg.replace("\\", "/").endswith("2025-11-25/08-00/Rec (2).m4a")
    assert (dest / "Rec.m4a").read_bytes() == b"an older recording with the same name"
    assert hourbatch.archived_shas(root)
