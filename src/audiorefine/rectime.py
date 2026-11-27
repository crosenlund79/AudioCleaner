"""When was a recording made? (start date and time of the original recording)

Priority, first usable source wins:

1. **Date and time in the file name** (e.g. ``20251125_080412.m4a``, ``2025-11-25 08.04.12.m4a``).
   Written by the recording device, survives copying, and is local wall-clock time.
2. **Recording time stored inside the file** (QuickTime/MP4 ``creationdate``/``creation_time``,
   BWF ``origination``, ID3 ``TDRC`` …) when it has a full date and time, is not in the future
   and not a placeholder (1904/1970).
3. **File creation time** (Windows), only when it is consistent with the recording: it must be
   earlier than the modification time by about the recording's duration (the file was created
   when recording started and last written when it stopped). Copying a file normally sets the
   creation time to the copy time, which fails this test and is then ignored.
4. **File modification time minus the duration** (the file is last written when the recording
   stops). Copying from a phone may change this time, so it is marked as *estimated*.

Sources 3 and 4 are always reported as *estimated* ("≈" in the transcripts). The time at which
AudioRefine processes the files is never used.

All times become timezone-aware instants. Times without a time zone (file names, some
metadata) are interpreted as local time of this PC; times in UTC or with an offset are converted
to this PC's local time for display and grouping. Daylight-saving transitions are therefore
handled by the operating system's rules; during the repeated hour in autumn both occurrences
end up in the same hour folder, ordered by their true instant.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .ordering import parse_filename_datetime

BASIS_NAME = "file name"
BASIS_META = "recording metadata"
BASIS_CREATED = "file creation time"
BASIS_MODIFIED = "file modification time minus duration"


@dataclass(frozen=True)
class RecTime:
    start: datetime          # timezone-aware
    basis: str
    estimated: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "basis": self.basis, "estimated": self.estimated, "note": self.note}

    @staticmethod
    def from_dict(d: dict) -> "RecTime":
        return RecTime(datetime.fromisoformat(d["start"]), d["basis"], bool(d["estimated"]), d.get("note", ""))


def aware(dt: datetime) -> datetime:
    """Naive values are local wall-clock time of this PC."""
    return dt if dt.tzinfo is not None else dt.astimezone()


def local(dt: datetime) -> datetime:
    """Local wall-clock time of this PC (timezone-aware)."""
    return aware(dt).astimezone()


def _plausible(dt: datetime, now: datetime) -> bool:
    a = aware(dt)
    return datetime(1995, 1, 1, tzinfo=timezone.utc) <= a <= now + timedelta(days=1)


def detect(info, now: datetime | None = None) -> RecTime:
    """Recording start time for a probed source (``probe.SourceInfo``)."""
    now = now or datetime.now(timezone.utc)
    dur = max(0.0, float(info.duration or 0.0))
    notes = []
    name_dt = info.name_datetime or parse_filename_datetime(info.path.stem if hasattr(info, "path") else info.name)
    meta_dt = info.meta_datetime
    if name_dt is not None and _plausible(name_dt, now):
        if meta_dt is not None and _plausible(meta_dt, now):
            diff = abs(aware(meta_dt).timestamp() - aware(name_dt).timestamp())
            if diff > 3 * 3600:
                notes.append(f"stored recording time differs by {diff / 3600:.1f} h; file name used")
        return RecTime(aware(name_dt), BASIS_NAME, False, "; ".join(notes))
    if meta_dt is not None and _plausible(meta_dt, now) and (meta_dt.hour or meta_dt.minute or meta_dt.second):
        return RecTime(aware(meta_dt), BASIS_META, False, f"tag {info.meta_datetime_tag}")
    mtime = float(info.mtime)
    bt = getattr(info, "birthtime", None)
    if bt:
        span = mtime - float(bt)
        if span >= 0 and abs(span - dur) <= max(60.0, 0.1 * dur):
            return RecTime(datetime.fromtimestamp(float(bt), tz=timezone.utc).astimezone(), BASIS_CREATED, True,
                           "creation and modification times match the recording length")
    start = datetime.fromtimestamp(mtime - dur, tz=timezone.utc).astimezone()
    return RecTime(start, BASIS_MODIFIED, True, "no recording time in file name or metadata")


def fmt_clock(dt: datetime, estimated: bool = False, with_date: bool = True) -> str:
    l = local(dt)
    s = l.strftime("%Y-%m-%d %H:%M:%S") if with_date else l.strftime("%H:%M:%S")
    return ("≈" + s) if estimated else s


def describe(rt: RecTime) -> str:
    return f"{fmt_clock(rt.start)} ({rt.basis}{', estimated' if rt.estimated else ''})"


def local_tz_name() -> str:
    try:
        return time.strftime("%Z") or "local time"
    except Exception:
        return "local time"
