"""Transcript data model and the TXT / SRT / complete-transcript writers.

Timestamp conventions
* ``.txt`` files and ``Transcript_Complete.txt`` use the continuous timeline of the stitched
  recording (they never restart at zero).
* ``.srt`` files are relative to their own audio file, so subtitles play in sync with it.
* A spoken segment that crosses a 10-minute file boundary is split at the word level: words
  that start before the boundary belong to the earlier file, the rest to the later file. The
  pieces are marked with an ellipsis (…) in the per-file transcripts; the complete transcript
  keeps the segment whole. No word is dropped or duplicated.
"""
from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

UNCERTAIN_TAG = "[uncertain]"


@dataclass
class TWord:
    start: float
    end: float
    text: str
    prob: float = 1.0


@dataclass
class TSegment:
    start: float
    end: float
    text: str
    words: list[TWord] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0
    temperature: float = 0.0
    language: str = ""
    uncertain: bool = False
    marker: bool = False      # "[unclear speech]" placeholder (no words were recognised)
    note: str = ""


@dataclass
class Piece:
    start: float
    end: float
    text: str
    uncertain: bool
    marker: bool
    cont_prev: bool = False   # continues a sentence from the previous audio file
    cont_next: bool = False   # continues in the next audio file


@dataclass
class SourceSpan:
    name: str
    start: float
    end: float


@dataclass
class SegmentFile:
    index: int
    stem: str
    audio_name: str
    start: float
    end: float
    start_sample: int = 0
    end_sample: int = 0


# ------------------------------------------------------------------------------ formatting
def fmt_ts(t: float, sep: str = ".") -> str:
    ms_total = int(round(max(0.0, t) * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_srt(t: float) -> str:
    return fmt_ts(t, ",")


def fmt_hms(t: float) -> str:
    """HH:MM:SS, with milliseconds only when the value is not a whole second."""
    ms_total = int(round(max(0.0, t) * 1000))
    if ms_total % 1000 == 0:
        s_total = ms_total // 1000
        return f"{s_total // 3600:02d}:{s_total % 3600 // 60:02d}:{s_total % 60:02d}"
    return fmt_ts(t)


def _words_text(words: list[TWord]) -> str:
    return "".join(w.text for w in words).strip()


# ------------------------------------------------------------------------------ splitting
def split_into_files(segs: list[TSegment], files: list[SegmentFile]) -> list[list[Piece]]:
    """Distribute transcript segments over the audio files (see module docstring)."""
    out: list[list[Piece]] = [[] for _ in files]
    if not files:
        return out
    bounds = [f.start for f in files[1:]]

    def file_of(t: float) -> int:
        lo, hi = 0, len(bounds)
        while lo < hi:
            mid = (lo + hi) // 2
            if t >= bounds[mid]:
                lo = mid + 1
            else:
                hi = mid
        return lo

    for seg in segs:
        if seg.marker or not seg.words:
            fi, fj = file_of(seg.start), file_of(max(seg.start, seg.end - 1e-6))
            for k in range(fi, fj + 1):
                s = max(seg.start, files[k].start)
                e = min(seg.end, files[k].end)
                if e <= s and k != fi:
                    continue
                out[k].append(Piece(s, max(e, s), seg.text, seg.uncertain, seg.marker,
                                    cont_prev=k > fi, cont_next=k < fj))
            continue
        groups: dict[int, list[TWord]] = {}
        for w in seg.words:
            groups.setdefault(file_of(w.start), []).append(w)
        keys = sorted(groups)
        for n, k in enumerate(keys):
            ws = groups[k]
            s = max(ws[0].start, files[k].start)
            e = min(max(ws[-1].end, s), files[k].end)
            text = seg.text if len(keys) == 1 else _words_text(ws)
            out[k].append(Piece(s, e, text, seg.uncertain, False, cont_prev=n > 0, cont_next=n < len(keys) - 1))
    for lst in out:
        lst.sort(key=lambda p: (p.start, p.end))
    return out


# ------------------------------------------------------------------------------ writers
def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _piece_text(p: Piece) -> str:
    t = p.text
    if p.cont_prev:
        t = "… " + t
    if p.cont_next:
        t = t + " …"
    return t


def _source_markers(sources: list[SourceSpan], start: float, end: float) -> list[tuple[float, str]]:
    """'── X begins at … ──' lines for recordings starting inside [start, end)."""
    return [(s.start, f"── {s.name} begins at {fmt_ts(s.start)} ──") for s in sources if start <= s.start < end]


def render_segment_txt(sf: SegmentFile, pieces: list[Piece], sources: list[SourceSpan]) -> str:
    buf = io.StringIO()
    buf.write(f"Recording timeline: {fmt_hms(sf.start)} – {fmt_hms(sf.end)}\n")
    buf.write(f"Audio file: {sf.audio_name}\n")
    overlapping = [s.name for s in sources if s.start < sf.end and s.end > sf.start]
    if overlapping:
        buf.write("Source recordings: " + ", ".join(overlapping) + "\n")
    buf.write("\n")
    events: list[tuple[float, int, str]] = []
    for t, line in _source_markers(sources, sf.start, sf.end):
        events.append((t, 0, line))
    for p in pieces:
        head = f"[{fmt_ts(p.start)} – {fmt_ts(p.end)}]"
        if p.uncertain and not p.marker:
            head += f" {UNCERTAIN_TAG}"
        events.append((p.start, 1, head + "\n" + _piece_text(p)))
    events.sort(key=lambda e: (e[0], e[1]))
    if not pieces:
        events.append((sf.end, 2, "(No speech was detected in this part of the recording.)"))
        events.sort(key=lambda e: (e[0], e[1]))
    buf.write("\n\n".join(e[2] for e in events))
    buf.write("\n")
    return buf.getvalue()


def _wrap_cue(text: str, width: int = 42) -> str:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    if len(lines) > 2:  # rebalance into two lines
        mid = len(text) // 2
        cut = text.rfind(" ", 0, mid + 1)
        if cut <= 0:
            cut = text.find(" ", mid)
        if cut > 0:
            return text[:cut].strip() + "\n" + text[cut:].strip()
    return "\n".join(lines)


def _cues_for_piece(p: Piece, words: list[TWord] | None, max_chars: int = 84, max_dur: float = 7.0) -> list[tuple[float, float, str]]:
    """Split a transcript piece into subtitle cues of at most two 42-character lines / 7 seconds.

    Pieces are divided into the smallest number of cues that satisfies the limits, with cue
    boundaries chosen at word boundaries so that cues have similar lengths (no orphan words);
    sentence ends are preferred as boundaries.
    """
    text = _piece_text(p)
    if not words or (len(text) <= max_chars and p.end - p.start <= max_dur):
        return [(p.start, p.end, text)]
    dur = max(words[-1].end - words[0].start, 0.0)
    n = max(1, -(-len(text) // max_chars), int(math.ceil(dur / max_dur)))
    groups: list[list[TWord]] = []
    total_chars = sum(len(w.text) for w in words)
    target = total_chars / n
    cur: list[TWord] = []
    acc = 0
    for i, w in enumerate(words):
        cur.append(w)
        acc += len(w.text)
        remaining_groups = n - len(groups) - 1
        if remaining_groups <= 0 or i == len(words) - 1:
            continue
        sentence_end = w.text.rstrip().endswith((".", "?", "!", ","))
        if acc >= target * (0.8 if sentence_end else 1.0):
            groups.append(cur)
            cur, acc = [], 0
    if cur:
        groups.append(cur)
    out = []
    for i, c in enumerate(groups):
        t = _words_text(c)
        if i == 0 and p.cont_prev:
            t = "… " + t
        if i == len(groups) - 1 and p.cont_next:
            t = t + " …"
        out.append((max(c[0].start, p.start), min(max(c[-1].end, c[0].start), p.end), t))
    return out


def render_segment_srt(sf: SegmentFile, pieces: list[Piece], piece_words: list[list[TWord] | None]) -> str:
    cues: list[tuple[float, float, str]] = []
    for p, ws in zip(pieces, piece_words):
        for s, e, t in _cues_for_piece(p, ws):
            if p.uncertain and not p.marker:
                t = t + " (?)"
            cues.append((s - sf.start, e - sf.start, t))
    cues.sort(key=lambda c: c[0])
    dur = sf.end - sf.start
    fixed = []
    for i, (s, e, t) in enumerate(cues):
        s = max(0.0, min(s, dur))
        e = max(s + 0.3, e)                     # minimum display time
        if i + 1 < len(cues):
            e = min(e, max(s + 0.05, cues[i + 1][0] - 0.001))
        e = min(e, dur)
        if e <= s:
            e = min(dur, s + 0.05)
        fixed.append((s, e, t))
    buf = io.StringIO()
    for i, (s, e, t) in enumerate(fixed, 1):
        buf.write(f"{i}\n{fmt_srt(s)} --> {fmt_srt(e)}\n{_wrap_cue(t)}\n\n")
    return buf.getvalue()


def words_for_pieces(segs: list[TSegment], files: list[SegmentFile]) -> list[list[list[TWord] | None]]:
    """Word lists matching ``split_into_files`` pieces (same order), used for SRT cue timing."""
    out: list[list[tuple[float, list[TWord] | None]]] = [[] for _ in files]
    bounds = [f.start for f in files[1:]]

    def file_of(t: float) -> int:
        k = 0
        while k < len(bounds) and t >= bounds[k]:
            k += 1
        return k

    for seg in segs:
        if seg.marker or not seg.words:
            fi, fj = file_of(seg.start), file_of(max(seg.start, seg.end - 1e-6))
            for k in range(fi, fj + 1):
                s = max(seg.start, files[k].start)
                e = min(seg.end, files[k].end)
                if e <= s and k != fi:
                    continue
                out[k].append((s, None))
            continue
        groups: dict[int, list[TWord]] = {}
        for w in seg.words:
            groups.setdefault(file_of(w.start), []).append(w)
        for k, ws in groups.items():
            out[k].append((max(ws[0].start, files[k].start), ws))
    res = []
    for lst in out:
        lst.sort(key=lambda x: x[0])
        res.append([w for _, w in lst])
    return res


def render_complete(segs: list[TSegment], sources: list[SourceSpan], files: list[SegmentFile], meta: dict) -> str:
    buf = io.StringIO()
    buf.write("AudioRefine – complete transcript\n")
    buf.write(f"Batch: {meta.get('batch', '')}\n")
    buf.write(f"Total duration: {fmt_ts(meta.get('duration', 0.0))}\n")
    langs = meta.get("languages") or []
    if langs:
        buf.write("Language(s) detected: " + ", ".join(langs) + "\n")
    buf.write(f"Created: {meta.get('created', '')} by AudioRefine {meta.get('version', '')} "
              f"(speech recognition: {meta.get('model', '')})\n\n")
    buf.write("Source recordings (position in the continuous timeline):\n")
    w = max([len(s.name) for s in sources] + [10])
    for i, s in enumerate(sources, 1):
        buf.write(f"  {i:>2}. {s.name:<{w}}  {fmt_ts(s.start)} – {fmt_ts(s.end)}\n")
    buf.write("\nAudio files:\n")
    for f in files:
        buf.write(f"  {f.audio_name}  {fmt_hms(f.start)} – {fmt_hms(f.end)}\n")
    buf.write(
        "\nAbout this transcript: it was produced automatically and has not been checked by a person.\n"
        "Timestamps refer to the continuous recording formed by joining the source recordings in the\n"
        "order above. Passages marked [uncertain] may contain recognition errors; [unclear speech]\n"
        "marks places where speech was detected but no words could be recognised. Nothing has been\n"
        "summarised, translated or rewritten.\n\n"
    )
    buf.write("=" * 72 + "\n\n")
    events: list[tuple[float, int, str]] = []
    for s in sources:
        events.append((s.start, 0, f"── {s.name} begins at {fmt_ts(s.start)} ──"))
    for f in files[1:]:
        events.append((f.start, 1, f"── {fmt_hms(f.start)} · start of {f.audio_name} ──"))
    for seg in segs:
        head = f"[{fmt_ts(seg.start)} – {fmt_ts(seg.end)}]"
        if seg.uncertain and not seg.marker:
            head += f" {UNCERTAIN_TAG}"
        events.append((seg.start, 2, head + "\n" + seg.text))
    events.sort(key=lambda e: (e[0], e[1]))
    if not segs:
        events.append((0.0, 3, "(No speech was detected in the recordings.)"))
    buf.write("\n\n".join(e[2] for e in events))
    buf.write("\n")
    return buf.getvalue()


def write_transcripts(out_dir: Path, segs: list[TSegment], sources: list[SourceSpan], files: list[SegmentFile],
                      meta: dict) -> list[Path]:
    written = []
    pieces = split_into_files(segs, files)
    pwords = words_for_pieces(segs, files)
    for f, ps, pw in zip(files, pieces, pwords):
        p_txt = out_dir / f"{f.stem}.txt"
        _atomic_write_text(p_txt, render_segment_txt(f, ps, sources))
        p_srt = out_dir / f"{f.stem}.srt"
        _atomic_write_text(p_srt, render_segment_srt(f, ps, pw))
        written += [p_txt, p_srt]
    p_all = out_dir / "Transcript_Complete.txt"
    _atomic_write_text(p_all, render_complete(segs, sources, files, meta))
    written.append(p_all)
    return written
