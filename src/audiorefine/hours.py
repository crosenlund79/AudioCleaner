"""Clock-hour groups: keys, splitting recordings at hour boundaries, hour manifests and coverage.

Output layout::

    refined/
        Transcript_Master_Complete.txt
        2026-11-25/
            Transcript_Day_Complete.txt
            08-00/  Audio_001.flac/.txt/.srt … Transcript_Hour_Complete.txt  _AudioRefine_hour.json
    processed/
        2026-11-25/08-00/<originals that *started* in that hour>

A recording is split *virtually* (sample positions only) at every local clock-hour boundary it
crosses; each part ("piece") is processed with the hour it belongs to. The original file is
never cut or modified.

``_AudioRefine_hour.json`` in each finished hour folder is the authoritative record that the
hour is complete: which pieces of which recordings (by SHA-256 and sample range) it contains,
the audio files with their checksums, and the transcript entries. Daily and master transcripts
are regenerated from these manifests, never by running speech recognition again.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .constants import MASTER_SR, REFINED_DIR
from .rectime import local

HOUR_MANIFEST = "_AudioRefine_hour.json"
WORDS_FILE = "_AudioRefine_words.json"
HOUR_TRANSCRIPT = "Transcript_Hour_Complete.txt"
DAY_TRANSCRIPT = "Transcript_Day_Complete.txt"
MASTER_TRANSCRIPT = "Transcript_Master_Complete.txt"
STATUS_FILE = "_AudioRefine_status.json"
ARCHIVE_JOURNAL = "_AudioRefine_archive.json"
MANIFEST_VERSION = 2


@dataclass(frozen=True, order=True)
class HourKey:
    date: str      # YYYY-MM-DD (local)
    hour: int      # 0..23 (local)

    @property
    def folder_name(self) -> str:
        return f"{self.hour:02d}-00"

    @property
    def label(self) -> str:
        return f"{self.hour:02d}:00–{(self.hour + 1) % 24:02d}:00" if self.hour < 23 else "23:00–24:00"

    @property
    def text(self) -> str:
        return f"{self.date} {self.label}"

    def dir(self, root: Path) -> Path:
        return root / REFINED_DIR / self.date / self.folder_name

    def processed_dir(self, root: Path) -> Path:
        return root / "processed" / self.date / self.folder_name

    @staticmethod
    def parse(s: str) -> "HourKey":
        d, h = s.split(" ")
        return HourKey(d, int(h[:2]))

    def __str__(self) -> str:
        return f"{self.date} {self.hour:02d}"


def hour_key_of(instant: datetime) -> HourKey:
    l = local(instant)
    return HourKey(l.strftime("%Y-%m-%d"), l.hour)


@dataclass
class Piece:
    """Part of one recording that falls into one clock hour (48 kHz sample positions in the source)."""
    key: HourKey
    src_start: int
    src_end: int
    start: datetime          # instant of src_start

    @property
    def samples(self) -> int:
        return self.src_end - self.src_start


def split_into_hours(start: datetime, total_samples: int, sr: int = MASTER_SR) -> list[Piece]:
    """Split [start, start + total/sr) at local clock-hour boundaries (DST-safe)."""
    if total_samples <= 0:
        return []
    start = start if start.tzinfo else start.astimezone()
    end = start + timedelta(seconds=total_samples / sr)
    # candidate boundaries: every quarter hour in UTC (covers zones with 30/45-minute offsets)
    s_utc = start.astimezone(timezone.utc)
    c = s_utc.replace(minute=(s_utc.minute // 15) * 15, second=0, microsecond=0) + timedelta(minutes=15)
    cuts = []
    while c < end:
        if hour_key_of(c) != hour_key_of(c - timedelta(microseconds=1)):
            cuts.append(c)
        c += timedelta(minutes=15)
    pieces = []
    a = 0
    for cut in cuts:
        b = min(total_samples, max(a, int(round((cut - start).total_seconds() * sr))))
        if b > a:
            pieces.append(Piece(hour_key_of(start + timedelta(seconds=a / sr)), a, b, start + timedelta(seconds=a / sr)))
        a = b
    if total_samples > a:
        pieces.append(Piece(hour_key_of(start + timedelta(seconds=a / sr)), a, total_samples,
                            start + timedelta(seconds=a / sr)))
    return pieces


# ------------------------------------------------------------------------------ manifests
def read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json_atomic(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def hour_dirs(root: Path) -> list[Path]:
    """Finished-looking hour folders, sorted chronologically."""
    refined = root / REFINED_DIR
    out = []
    if not refined.is_dir():
        return out
    for d in sorted(refined.iterdir()):
        if not d.is_dir() or len(d.name) != 10 or d.name[4] != "-" or d.name[7] != "-":
            continue
        for h in sorted(d.iterdir()):
            if h.is_dir() and len(h.name) == 5 and h.name.endswith("-00") and h.name[:2].isdigit():
                out.append(h)
    return out


def load_complete_manifests(root: Path) -> dict[HourKey, dict]:
    """Hour manifests that record a completed hour. A folder without a valid manifest is not complete."""
    res = {}
    for h in hour_dirs(root):
        m = read_json(h / HOUR_MANIFEST)
        if not m or m.get("app") != "AudioRefine" or m.get("state") != "complete":
            continue
        if not all((h / s["audio"]).is_file() and (h / s["audio"]).stat().st_size > 0 for s in m.get("segments", [])):
            continue
        res[HourKey(m["date"], int(m["hour"]))] = m
    return res


@dataclass
class Coverage:
    """Which sample ranges of each recording (by SHA-256) are already in completed hours."""
    ranges: dict[str, list[tuple[int, int, HourKey]]] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    last_word_end: dict[tuple[str, int], float] = field(default_factory=dict)   # (sha, src_end) -> seconds in source

    @staticmethod
    def build(manifests: dict[HourKey, dict]) -> "Coverage":
        cov = Coverage()
        for key, m in manifests.items():
            for p in m.get("pieces", []):
                cov.ranges.setdefault(p["sha256"], []).append((int(p["src_start"]), int(p["src_end"]), key))
                cov.totals[p["sha256"]] = int(p["source_samples"])
                if p.get("last_word_end_src") is not None:
                    cov.last_word_end[(p["sha256"], int(p["src_end"]))] = float(p["last_word_end_src"])
        for v in cov.ranges.values():
            v.sort()
        return cov

    def covered(self, sha: str, a: int, b: int) -> bool:
        return any(x <= a and b <= y for x, y, _ in self.ranges.get(sha, []))

    def fully_covered(self, sha: str) -> bool:
        if sha not in self.ranges:
            return False
        pos = 0
        for a, b, _ in self.ranges[sha]:
            if a > pos:
                return False
            pos = max(pos, b)
        return pos >= self.totals.get(sha, 1 << 62)

    def hours_of(self, sha: str) -> list[HourKey]:
        return sorted({k for _, _, k in self.ranges.get(sha, [])})


def piece_record(p: Piece, name: str, sha: str, size: int, source_samples: int, out_start: int, rt,
                 last_word_end_src: float | None = None) -> dict:
    return {
        "source": name, "sha256": sha, "size": size, "source_samples": source_samples,
        "src_start": p.src_start, "src_end": p.src_end, "out_start": out_start,
        "out_end": out_start + p.samples, "start": p.start.isoformat(),
        "time_basis": rt.basis, "estimated": rt.estimated, "last_word_end_src": last_word_end_src,
    }


def clock_at(pieces: list[dict], out_seconds: float) -> tuple[datetime | None, bool, int]:
    """Clock time (aware) of a position on an hour's audio timeline, its estimated flag, piece index."""
    pos = out_seconds * MASTER_SR
    for i, p in enumerate(pieces):
        if p["out_start"] <= pos < p["out_end"] or (i == len(pieces) - 1 and pos >= p["out_start"]):
            st = datetime.fromisoformat(p["start"])
            return st + timedelta(seconds=(pos - p["out_start"]) / MASTER_SR), bool(p["estimated"]), i
    return None, True, -1
