"""Human-readable transcripts for the hourly layout.

* ``Audio_NNN.txt``                – one 10-minute audio file; times are recording clock time plus
                                     the position inside that audio file.
* ``Audio_NNN.srt``                – subtitles relative to the audio file (unchanged format).
* ``Transcript_Hour_Complete.txt`` – everything said in one clock hour.
* ``Transcript_Day_Complete.txt``  – all completed hours of one date.
* ``Transcript_Master_Complete.txt`` – all completed hours of all dates (in ``refined``).

Day and master transcripts are generated only from the hour manifests (``hours.py``), never by
running speech recognition again. All files are UTF-8 and replaced atomically.
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .constants import MASTER_SR, SEGMENT_SECONDS
from .hours import HourKey, clock_at
from .rectime import fmt_clock
from .transcript import Piece as TextPiece

UNCERTAIN = "[uncertain]"
RULE_DATE = "=" * 50
RULE_HOUR = "-" * 30


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fmt_offset(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def audio_ref(t: float) -> tuple[str, float]:
    """(Audio_NNN, offset in that file) for a position on the hour's audio timeline."""
    idx = int(t // SEGMENT_SECONDS)
    return f"Audio_{idx + 1:03d}", t - idx * SEGMENT_SECONDS


def _entry_text(e: dict) -> str:
    t = e["text"]
    if e.get("cont_prev"):
        t = "… " + t
    if e.get("cont_next"):
        t = t + " …"
    return t


def _clock(m: dict, t: float) -> tuple[str, bool]:
    dt, est, _ = clock_at(m["pieces"], t)
    if dt is None:
        return "?", True
    return fmt_clock(dt, est), est


def _source_lines(m: dict) -> list[tuple[float, str]]:
    """'── X begins … ──' markers at the start of every recording piece in the hour."""
    out = []
    for p in m["pieces"]:
        st = datetime.fromisoformat(p["start"])
        what = "begins" if p["src_start"] == 0 else "continues"
        out.append((p["out_start"] / MASTER_SR,
                    f"── {p['source']} {what} ({fmt_clock(st, p['estimated'])}) ──"))
    return out


def any_estimated(m: dict) -> bool:
    return any(p["estimated"] for p in m["pieces"])


def estimated_note() -> str:
    return ("Times marked ≈ are estimated from the file's dates because neither the file name nor the "
            "file itself contains the recording time.")


# ------------------------------------------------------------------------------ per 10-minute file
def render_segment_txt(m: dict, seg: dict, pieces: list[TextPiece]) -> str:
    key = HourKey(m["date"], int(m["hour"]))
    seg_start = seg["start_sample"] / MASTER_SR
    seg_end = seg["end_sample"] / MASTER_SR
    buf = io.StringIO()
    buf.write(f"Recording date: {key.date}\nRecording hour: {key.label}\n")
    buf.write(f"Audio file: {seg['audio']} ({fmt_offset(seg_start)}–{fmt_offset(seg_end)} of this hour's audio)\n")
    names = [p["source"] for p in m["pieces"] if p["out_start"] < seg["end_sample"] and p["out_end"] > seg["start_sample"]]
    if names:
        buf.write("Source recordings: " + ", ".join(dict.fromkeys(names)) + "\n")
    if any_estimated(m):
        buf.write(estimated_note() + "\n")
    buf.write("\n")
    stem = Path(seg["audio"]).stem
    events: list[tuple[float, int, str]] = []
    for t, line in _source_lines(m):
        if seg_start <= t < seg_end:
            events.append((t, 0, line))
    for p in pieces:
        clock, _ = _clock(m, p.start)
        head = f"[{clock} | {stem} {fmt_offset(p.start - seg_start)}]"
        if p.uncertain and not p.marker:
            head += f" {UNCERTAIN}"
        txt = p.text
        if p.cont_prev:
            txt = "… " + txt
        if p.cont_next:
            txt = txt + " …"
        events.append((p.start, 1, head + "\n" + txt))
    if not pieces:
        events.append((seg_end, 2, "(No speech was detected in this part of the recording.)"))
    events.sort(key=lambda e: (e[0], e[1]))
    buf.write("\n\n".join(e[2] for e in events) + "\n")
    return buf.getvalue()


# ------------------------------------------------------------------------------ hour
def hour_entries(m: dict) -> list[tuple[float, int, str]]:
    """(time on the hour timeline, order, text block) for the hour, day and master transcripts."""
    ev: list[tuple[float, int, str]] = []
    for t, line in _source_lines(m):
        ev.append((t, 0, line))
    for e in m.get("entries", []):
        clock, _ = _clock(m, e["t0"])
        stem, off = audio_ref(e["t0"])
        head = f"[{clock} | {stem} {fmt_offset(off)}]"
        if e.get("uncertain") and not e.get("marker"):
            head += f" {UNCERTAIN}"
        ev.append((e["t0"], 1, head + "\n" + _entry_text(e)))
    ev.sort(key=lambda x: (x[0], x[1]))
    return ev


def render_hour(m: dict) -> str:
    key = HourKey(m["date"], int(m["hour"]))
    buf = io.StringIO()
    buf.write("AudioRefine – hour transcript\n")
    buf.write(f"Recording date: {key.date}\nRecording hour: {key.label}\n")
    total = m["total_samples"] / MASTER_SR
    files = ", ".join(s["audio"] for s in m["segments"])
    buf.write(f"Audio: {files} ({fmt_offset(total)} of recorded audio in this hour)\n")
    buf.write("Recordings in this hour:\n")
    for p in m["pieces"]:
        st = datetime.fromisoformat(p["start"])
        part = "" if (p["src_start"] == 0 and p["src_end"] == p["source_samples"]) else " (part)"
        buf.write(f"  {p['source']}{part}  from {fmt_clock(st, p['estimated'])}  "
                  f"[time from {p['time_basis']}]\n")
    if any_estimated(m):
        buf.write(estimated_note() + "\n")
    buf.write("Automatic transcription; [uncertain] marks passages that may contain recognition errors, "
              "[unclear speech] marks speech that could not be recognised.\n\n")
    ev = hour_entries(m)
    if not any(o == 1 for _, o, _ in ev):
        ev.append((total, 2, "(No speech was detected in this hour.)"))
    buf.write("\n\n".join(e[2] for e in ev) + "\n")
    return buf.getvalue()


# ------------------------------------------------------------------------------ day and master
def _hour_section(m: dict, master: bool) -> str:
    key = HourKey(m["date"], int(m["hour"]))
    head = f"HOUR: {key.label}" if master else key.label
    ev = hour_entries(m)
    body = "\n\n".join(e[2] for e in ev) if any(o == 1 for _, o, _ in ev) else "(No speech was detected in this hour.)"
    return f"{RULE_HOUR}\n{head}\n{RULE_HOUR}\n\n{body}\n"


def render_day(date: str, manifests: Iterable[dict]) -> str:
    ms = sorted(manifests, key=lambda m: int(m["hour"]))
    buf = io.StringIO()
    buf.write(f"{RULE_DATE}\nDATE: {date}\n{RULE_DATE}\n\n")
    buf.write(f"Hours included: {len(ms)} ({', '.join(HourKey(date, int(m['hour'])).label for m in ms)})\n")
    if any(any_estimated(m) for m in ms):
        buf.write(estimated_note() + "\n")
    buf.write("Each entry: [recording date and time | audio file in that hour's folder, position].\n\n")
    for m in ms:
        buf.write(_hour_section(m, master=False) + "\n")
    return buf.getvalue()


def _span(manifests: list[dict]) -> tuple[str, str]:
    first = last = None
    for m in manifests:
        for p in m["pieces"]:
            st = datetime.fromisoformat(p["start"])
            en, est, _ = clock_at([p], (p["out_end"] - 1) / MASTER_SR)
            if first is None or st < first[0]:
                first = (st, p["estimated"])
            if en is not None and (last is None or en > last[0]):
                last = (en, est)
    f = fmt_clock(*first) if first else "–"
    l_ = fmt_clock(*last) if last else "–"
    return f, l_


def write_master(path: Path, manifests: dict[HourKey, dict], status_line: str) -> None:
    """Stream the master transcript to a temporary file, then replace atomically."""
    keys = sorted(manifests)
    ms = [manifests[k] for k in keys]
    dates = sorted({k.date for k in keys})
    first, last = _span(ms)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(f"{RULE_DATE}\nAUDIOREFINE — MASTER TRANSCRIPT\n{RULE_DATE}\n\n")
        f.write(f"Status: {status_line}\n")
        f.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"First recording: {first}\nLast recording: {last}\n\n")
        f.write(f"Dates included: {len(dates)}\nHourly groups included: {len(keys)}\n\n")
        f.write("About this transcript: produced automatically from the recordings, not checked by a person, "
                "and not summarised, translated or rewritten. [uncertain] marks passages that may contain "
                "recognition errors; [unclear speech] marks speech that could not be recognised. "
                "Each entry shows the recording date and time and the audio file (in that hour's folder "
                "under refined\\<date>\\<hour>) with the position inside it.")
        if any(any_estimated(m) for m in ms):
            f.write(" " + estimated_note())
        f.write("\n\n")
        for d in dates:
            f.write(f"{RULE_DATE}\nDATE: {d}\n{RULE_DATE}\n\n")
            for k in keys:
                if k.date == d:
                    f.write(_hour_section(manifests[k], master=True) + "\n")
        if not keys:
            f.write("(No hours have been completed yet.)\n\n")
        f.write(f"{RULE_DATE}\nEND OF TRANSCRIPT\n{RULE_DATE}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
