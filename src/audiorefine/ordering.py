"""Deterministic ordering of recordings.

Default strategy (documented in README.md / TECHNICAL_DECISIONS.md):

1. If *every* recording has a full date and time in its file name (e.g. ``20240315_143022``,
   ``2024-03-15 14.30.22``, ``Recording 2024-03-15 at 14.30.22``), order chronologically by it.
   Such names are written by the recording device and survive copying.
2. Otherwise order by file name using *natural* sorting (as Windows Explorer does), so that
   ``Recording2`` comes before ``Recording10``.

Embedded metadata timestamps are only used to *warn* when they contradict the chosen order:
recorders write them inconsistently (encode time vs. start time, UTC vs. local time, unset
clocks). File-system timestamps are never used for ordering because copying changes them.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Sequence

_NUM = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple:
    parts = _NUM.split(name.casefold())
    key = []
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            key.append((0, int(p), len(p)))
        else:
            key.append((1, p.strip(), 0))
    return (tuple(key), name.casefold(), name)


def _valid(y, mo, d, h=0, mi=0, s=0) -> datetime | None:
    try:
        if not (1990 <= y <= 2100):
            return None
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


# Patterns with full date and time. Separators: '-', '_', '.', ' ', 'T', ':' and ' at '.
_PATTERNS = [
    # 2024-03-15 14.30.22 | 2024-03-15_14-30-22 | 2024-03-15T14:30:22 | 2024-03-15 at 14.30.22
    re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})(?:[ _T-]|\s+at\s+|\s*kl\.?\s*)(\d{2})[.:_-]?(\d{2})(?:[.:_-]?(\d{2}))?(?!\d)", re.I),
    # 20240315_143022 | 20240315-143022 | 20240315T143022 | 20240315 143022 | 20240315143022
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[ _T-]?(\d{2})(\d{2})(\d{2})?(?!\d)"),
]
# Day-first (European) dates: 15-03-2024 14.30.22, 15.03.2024_14-30
_PATTERN_DMY = re.compile(r"(?<!\d)(\d{2})[-_.](\d{2})[-_.](\d{4})(?:[ _T-]|\s+kl\.?\s*)(\d{2})[.:_-]?(\d{2})(?:[.:_-]?(\d{2}))?(?!\d)", re.I)


def parse_filename_datetime(stem: str) -> datetime | None:
    """Full date+time found in a file name, or None."""
    for pat in _PATTERNS:
        m = pat.search(stem)
        if m:
            y, mo, d, h, mi = (int(m.group(i)) for i in range(1, 6))
            s = int(m.group(6)) if m.group(6) else 0
            dt = _valid(y, mo, d, h, mi, s)
            if dt and h < 24 and mi < 60 and s < 60:
                return dt
    m = _PATTERN_DMY.search(stem)
    if m:
        d, mo, y, h, mi = (int(m.group(i)) for i in range(1, 6))
        s = int(m.group(6)) if m.group(6) else 0
        dt = _valid(y, mo, d, h, mi, s)
        if dt:
            return dt
    return None


_META_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M%z", "%Y-%m-%d %H:%M",
)


def parse_metadata_datetime(value: str) -> datetime | None:
    v = value.strip().replace("Z", "+0000")
    if re.search(r"[+-]\d{2}:\d{2}$", v):
        v = v[:-3] + v[-2:]
    for fmt in _META_FORMATS:
        try:
            dt = datetime.strptime(v, fmt)
        except ValueError:
            continue
        if dt.year < 1990 or dt.year > 2100:
            return None
        return dt   # timezone-aware when the value carries an offset/UTC marker, else local wall-clock
    return None


def to_instant(dt: datetime) -> float:
    """POSIX timestamp of a datetime; naive values are interpreted as local wall-clock time."""
    return dt.timestamp()


def determine_order(items: Sequence) -> tuple[list, str, list[str]]:
    """Order readable recordings. ``items`` need ``name``, ``name_datetime``, ``meta_datetime``.

    Returns (ordered items, human readable basis, warnings).
    """
    items = list(items)
    warnings: list[str] = []
    if items and all(i.name_datetime is not None for i in items):
        ordered = sorted(items, key=lambda i: (i.name_datetime, natural_key(i.name)))
        basis = "date and time in the file names"
        times = [i.name_datetime for i in ordered]
        if len(set(times)) != len(times):
            warnings.append("Some file names contain the same date and time; those files are ordered by name.")
    else:
        ordered = sorted(items, key=lambda i: natural_key(i.name))
        basis = "file name (natural order, as in Windows Explorer)"
    if len(ordered) > 1 and all(i.meta_datetime is not None for i in ordered):
        by_meta = sorted(ordered, key=lambda i: (to_instant(i.meta_datetime), natural_key(i.name)))
        if [i.name for i in by_meta] != [i.name for i in ordered]:
            warnings.append(
                "The recording times stored inside the files suggest a different order. "
                "Check the list; rename the files if the order is wrong.")
    return ordered, basis, warnings
