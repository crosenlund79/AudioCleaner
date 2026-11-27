"""Analyze sounds: the independent sound analysis of a folder, one clock hour at a time.

1. Lock the folder for analysis (never while it is being transcribed).
2. Find the original recordings: those directly in the folder and those already archived in
   ``processed`` (any depth). Identify each by SHA-256 (cached by size and modification time).
3. Place each recording on the clock with the transcription's rules (``rectime``; for recordings
   already transcribed, the exact start time of their hour manifests) and split it into hours.
4. Hours whose results are missing, from another model/vocabulary version, or whose set of
   recordings changed, are (re)done – oldest date first, oldest hour first. For each hour every
   recording in it is analysed completely (once; a recording crossing into the next hour is not
   analysed twice), the events that *start* in the hour are written to the hour record, and the
   day and master reports are regenerated before the next hour begins.
5. Originals are only read. Transcripts and enhanced audio are not touched.
"""
from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .. import __version__
from ..batch import SafetyError, folder_busy, sha256_of
from ..constants import AUDIO_EXTENSIONS, MASTER_SR, PROCESSED_DIR
from ..discovery import candidate_files
from ..hours import HourKey, hour_key_of, load_complete_manifests, split_into_hours, write_json_atomic
from ..probe import probe
from ..progress import CancelledError, CancelToken
from ..rectime import RecTime, detect, fmt_clock
from . import reports, store
from .analyze import AudioReadError, analyze_file
from .model import SoundModel
from .vocab import load as load_vocab

log = logging.getLogger("audiorefine.sound")


@dataclass
class SoundOutcome:
    status: str                       # success | failed | cancelled
    message: str
    warnings: list[str] = field(default_factory=list)
    hours_done: list[str] = field(default_factory=list)
    report: str | None = None
    log_file: str | None = None


@dataclass
class Source:
    sha: str
    path: Path
    rel: str
    name: str
    size: int
    duration: float
    rt: RecTime
    hours: list[HourKey] = field(default_factory=list)


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def discover(root: Path, cancel: Callable[[], None]) -> tuple[list[Path], list[Path]]:
    top, _ = candidate_files(root)
    archived = []
    pdir = root / PROCESSED_DIR
    if pdir.is_dir():
        for dp, dns, fns in os.walk(pdir):
            dns[:] = [d for d in dns if not d.startswith(".")]
            for fn in fns:
                if not fn.startswith((".", "~$")) and Path(fn).suffix.lower() in AUDIO_EXTENSIONS:
                    archived.append(Path(dp) / fn)
            cancel()
    return sorted(top), sorted(archived)


def manifest_times(root: Path) -> dict[str, RecTime]:
    """Recording start times already used by the transcription (so both agree exactly)."""
    out: dict[str, RecTime] = {}
    for m in load_complete_manifests(root).values():
        for p in m.get("pieces", []):
            sha = p["sha256"]
            if sha in out:
                continue
            st = datetime.fromisoformat(p["start"]) - timedelta(seconds=int(p["src_start"]) / MASTER_SR)
            out[sha] = RecTime(st, p.get("time_basis", ""), bool(p.get("estimated")), "as transcribed")
    return out


class Progress:
    def __init__(self, emit, work: dict[HourKey, float]):
        self.emit, self.work = emit, dict(work)
        self.done = 0.0
        self.key: HourKey | None = None
        self.frac = 0.0
        self.hours_done = 0
        self.dates_done: set[str] = set()
        self.t0 = time.monotonic()
        self._last = 0.0
        self.detail = ""

    def start(self, key: HourKey):
        self.key, self.frac = key, 0.0
        self.send(force=True)

    def set(self, frac: float, detail: str | None = None):
        self.frac = max(0.0, min(1.0, frac))
        if detail is not None:
            self.detail = detail
        self.send()

    def finish(self):
        self.done += self.work.get(self.key, 0.0)
        self.hours_done += 1
        if self.key.date not in {k.date for k in self.work if k > self.key}:
            self.dates_done.add(self.key.date)
        self.frac = 0.0
        self.send(force=True)

    def send(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last < 0.25:
            return
        self._last = now
        total = max(1e-6, sum(self.work.values()))
        p = min(1.0, (self.done + self.work.get(self.key, 0.0) * self.frac) / total) if self.key else 0.0
        el = now - self.t0
        eta = el * (1 - p) / p if p > 0.03 and el > 10 else None
        self.emit({"type": "progress", "stage": "hour", "label": "Analysing sounds", "detail": self.detail,
                   "fraction": round(p, 4), "hour_fraction": round(self.frac, 4),
                   "current_date": self.key.date if self.key else None,
                   "current_hour": self.key.label if self.key else None,
                   "hours_done": self.hours_done, "hours_total": len(self.work),
                   "days_done": len(self.dates_done), "days_total": len({k.date for k in self.work}),
                   "elapsed": round(el, 1), "eta": round(eta, 0) if eta is not None else None})


def plan_sources(root: Path, emit, chk, warnings: list[str]) -> tuple[list[Source], dict]:
    index = store.load_index(root)
    top, archived = discover(root, chk)
    mtimes = manifest_times(root)
    sources: dict[str, Source] = {}
    files = {}
    n = len(top) + len(archived)
    for i, p in enumerate(top + archived):
        chk()
        emit({"type": "progress", "stage": "prepare", "label": "Finding recordings", "detail": p.name,
              "fraction": 0.0, "hour_fraction": i / max(1, n), "hours_done": 0, "hours_total": 0})
        rel = _rel(root, p)
        try:
            st = p.stat()
        except OSError:
            continue
        cached = index["files"].get(rel)
        if cached and cached.get("size") == st.st_size and cached.get("mtime_ns") == st.st_mtime_ns and cached.get("sha256"):
            sha = cached["sha256"]
        else:
            sha = None
        info = probe(p, check_decode=True)
        if not info.ok:
            if p.parent == root:
                warnings.append(f"Not analysed (cannot be read): {p.name}")
            continue
        if sha is None:
            sha = sha256_of(p, chk)
        files[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
        if sha in sources:
            continue
        rt = mtimes.get(sha) or detect(info)
        sources[sha] = Source(sha, p, rel, p.name, st.st_size, float(info.duration or 0.0), rt)
    index["files"] = files
    for s in sources.values():
        prev = index["sources"].get(s.sha, {})
        index["sources"][s.sha] = {"name": s.name, "path": s.rel, "size": s.size,
                                   "rec_start": s.rt.start.isoformat(), "estimated": s.rt.estimated,
                                   "time_basis": s.rt.basis, "duration": prev.get("duration", round(s.duration, 3))}
    store.save_index(root, index)
    out = sorted(sources.values(), key=lambda s: (s.rt.start.timestamp(), s.name))
    for s in out:
        s.hours = [p.key for p in split_into_hours(s.rt.start, max(1, int(round(s.duration * MASTER_SR))))]
    return out, index


def hour_needs_work(rec: dict | None, shas: set[str], fp: dict) -> bool:
    if not rec or not store.same_fingerprint(rec.get("fingerprint"), fp):
        return True
    return {s["sha256"] for s in rec.get("sources", [])} != shas


def build_hour_record(key: HourKey, srcs: list[Source], results: dict[str, dict], fp: dict, vocab) -> dict:
    cats = {c.id: c for c in vocab.categories}
    events, src_entries = [], []
    for s in srcs:
        r = results[s.sha]
        dur = float(r["duration"])
        pieces = [p for p in split_into_hours(s.rt.start, max(1, int(round(dur * MASTER_SR)))) if p.key == key]
        if not pieces:
            continue
        a, b = pieces[0].src_start / MASTER_SR, pieces[-1].src_end / MASTER_SR
        src_entries.append({"sha256": s.sha, "name": s.name, "path": s.rel, "from": round(a, 3), "to": round(b, 3),
                            "hour_start": (s.rt.start + timedelta(seconds=a)).isoformat(),
                            "rec_start": s.rt.start.isoformat(), "estimated": s.rt.estimated,
                            "time_basis": s.rt.basis})
        for e in r["events"]:
            st = s.rt.start + timedelta(seconds=e["start"])
            if hour_key_of(st) != key:
                continue
            c = cats.get(e["category"])
            if c is None or not c.enabled:
                continue
            en = s.rt.start + timedelta(seconds=e["end"])
            events.append({
                "id": f"{s.sha[:12]}-{e['category']}-{int(round(e['start'] * 10))}",
                "kind": "sound", "category": c.id, "display": c.display, "group": c.group, "sound_kind": c.kind,
                "start": st.isoformat(timespec="milliseconds"), "end": en.isoformat(timespec="milliseconds"),
                "date": key.date, "hour": key.hour, "confidence": e["confidence"],
                "score_peak": e["score_peak"], "score_mean": e["score_mean"], "windows": e["windows"],
                "hits": e["hits"], "during_media": e["during_media"], "frequent": bool(e.get("frequent")),
                "source_sha256": s.sha, "source_name": s.name, "src_start": e["start"], "src_end": e["end"],
                "estimated": s.rt.estimated, "time_basis": s.rt.basis,
            })
    events.sort(key=lambda e: (e["start"], e["category"]))
    return {"app": "AudioRefine", "kind": "sound-hour", "state": "complete", "date": key.date, "hour": key.hour,
            "fingerprint": fp, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "sources": src_entries, "events": events}


def analyze_folder(root: Path, emit: Callable[[dict], None], cancel: CancelToken,
                   log_file: Path | None = None) -> SoundOutcome:
    root = Path(root).resolve()
    lock = store.AnalysisLock(root)
    warnings: list[str] = []
    done: list[HourKey] = []
    status_written = False
    finished = False
    fp: dict = {}
    vocab = None
    t0 = time.time()

    def chk():
        cancel.check()

    try:
        log.info("AudioRefine %s - sound analysis of %s", __version__, root)
        if not root.is_dir():
            return SoundOutcome("failed", "The selected folder does not exist.")
        if folder_busy(root):
            return SoundOutcome("failed", "This folder is being processed (transcription) right now. Analyse the "
                                          "sounds when that has finished.")
        vocab = load_vocab()
        fp = store.fingerprint(vocab)
        log.info("Sound model %s, vocabulary %s (%s), %d enabled categories", fp["model"], vocab.version,
                 vocab.digest, len(vocab.enabled))
        lock.acquire()
        for r in store.cleanup_temp(root):
            log.info("Removed an unfinished temporary file: %s", r)
        sources, index = plan_sources(root, emit, chk, warnings)
        if not sources:
            return SoundOutcome("failed", "No readable audio recordings were found in the folder or in its "
                                          f"'{PROCESSED_DIR}' folder.", warnings)
        for s in sources:
            log.info("  %s | %s | starts %s [%s%s] | %.1f s | hours %s", s.name, s.sha[:12], fmt_clock(s.rt.start),
                     s.rt.basis, ", estimated" if s.rt.estimated else "", s.duration, [str(k) for k in s.hours])
        by_hour: dict[HourKey, list[Source]] = {}
        for s in sources:
            for k in s.hours:
                by_hour.setdefault(k, []).append(s)
        recs = store.hour_records(root)
        todo = sorted(k for k, ss in by_hour.items() if hour_needs_work(recs.get(k), {s.sha for s in ss}, fp))
        # combined reports always reflect what exists (repairs an interrupted update)
        reports.write_combined(root, fp, vocab, "IN PROGRESS" if todo else "COMPLETE")
        status_written = True
        if not todo:
            data = reports.write_combined(root, fp, vocab, "COMPLETE")
            reports.write_annotated(root, data)
            finished = True
            return SoundOutcome("success", "Nothing new to analyse – all recordings in this folder have already "
                                           "been analysed. The reports were refreshed.", warnings,
                                report=str(store.adir(root) / store.MASTER_TXT))
        log.info("Hours to analyse: %s", [str(k) for k in todo])
        emit({"type": "progress", "stage": "model", "label": "Loading the sound model", "detail": "",
              "fraction": 0.0, "hours_done": 0, "hours_total": len(todo)})
        model = SoundModel()
        results: dict[str, dict] = {}
        work = {k: sum(max(1.0, min(3600.0, s.duration / max(1, len(s.hours)))) for s in by_hour[k]) for k in todo}
        prog = Progress(emit, work)
        i = 0
        while i < len(todo):
            chk()
            key = todo[i]
            prog.start(key)
            srcs = sorted(by_hour[key], key=lambda s: (s.rt.start.timestamp(), s.name))
            need = [s for s in srcs if s.sha not in results]
            for n_s, s in enumerate(need):
                r = store.load_source_result(root, s.sha, fp)
                if r is None:
                    log.info("Analysing %s", s.name)
                    t1 = time.time()
                    base, span = n_s / max(1, len(need)), 1 / max(1, len(need))
                    try:
                        res = analyze_file(s.path, model, vocab, chk,
                                           lambda f: prog.set(base + span * f, f"{s.name}"), s.duration)
                    except AudioReadError as e:
                        warnings.append(f"Not analysed: {s.name} ({e})")
                        log.warning("Not analysed: %s: %s", s.name, e)
                        res = {"duration": s.duration, "events": [], "media_spans": [], "error": str(e)}
                    r = {"app": "AudioRefine", "kind": "sound-source", "sha256": s.sha, "name": s.name,
                         "fingerprint": fp, "analysed": datetime.now().astimezone().isoformat(timespec="seconds"),
                         "seconds": round(time.time() - t1, 1), **res}
                    store.save_source_result(root, s.sha, r)
                    log.info("  %s: %.1f s audio in %.1f s, %d events", s.name, float(res["duration"]),
                             time.time() - t1, len(res["events"]))
                results[s.sha] = r
                # exact duration may move the end of a recording into an hour that was not planned
                exact = [p.key for p in split_into_hours(s.rt.start, max(1, int(round(float(r["duration"]) * MASTER_SR))))]
                index["sources"][s.sha]["duration"] = r["duration"]
                for k in exact:
                    if k not in by_hour:
                        by_hour[k] = [s]
                    elif s not in by_hour[k]:
                        by_hour[k].append(s)
                    if k > key and k not in todo and hour_needs_work(recs.get(k), {x.sha for x in by_hour[k]}, fp):
                        todo.append(k)
                        prog.work[k] = 1.0
            todo.sort()
            prog.set(0.97, "Writing results")
            srcs = sorted(by_hour[key], key=lambda s: (s.rt.start.timestamp(), s.name))
            for s in srcs:
                if s.sha not in results:
                    results[s.sha] = store.load_source_result(root, s.sha, fp) or {"duration": s.duration, "events": []}
            rec = build_hour_record(key, srcs, results, fp, vocab)
            d = store.hour_dir(root, key)
            d.mkdir(parents=True, exist_ok=True)
            write_json_atomic(d / store.HOUR_JSON, rec)
            reports.write_hour_txt(root, key, rec)
            reports.write_combined(root, fp, vocab, "IN PROGRESS", dates={key.date})
            done.append(key)
            prog.finish()
            emit({"type": "hour_done", "date": key.date, "hour": key.label, "folder": str(d),
                  "report": str(store.adir(root) / store.MASTER_TXT), "hours_done": len(done), "hours_total": len(todo),
                  "events": len(rec["events"])})
            log.info("Hour %s: %d event(s)", key, len(rec["events"]))
            # release analyses that no later hour needs
            for sha in [x for x in results if all(k <= key for k in [kk for kk, ss in by_hour.items()
                                                                     if any(s.sha == x for s in ss)])]:
                results.pop(sha, None)
            i = todo.index(key) + 1
        store.save_index(root, index)
        data = reports.write_combined(root, fp, vocab, "COMPLETE")
        ann = reports.write_annotated(root, data)
        finished = True
        log.info("Sound analysis finished: %d hour(s) in %.0f s; annotated transcript: %s", len(done),
                 time.time() - t0, ann)
        n_ev = len(reports.visible(data["events"]))
        return SoundOutcome("success", f"Done. {len(done)} hour(s) on {len({k.date for k in done})} day(s) were "
                                       f"analysed; {n_ev} sound event(s) in total. Open Sound Explorer to review them.",
                            warnings, [str(k) for k in done], str(store.adir(root) / store.MASTER_TXT))
    except CancelledError:
        log.warning("Sound analysis cancelled")
        return SoundOutcome("cancelled", f"Sound analysis was cancelled. {len(done)} finished hour(s) are kept; "
                                         "analyse the folder again to continue.", warnings, [str(k) for k in done],
                            str(store.adir(root) / store.MASTER_TXT))
    except SafetyError as e:
        return SoundOutcome("failed", str(e), warnings)
    except Exception as e:
        log.error("Sound analysis failed: %s\n%s", e, traceback.format_exc())
        return SoundOutcome("failed", f"Sound analysis failed ({type(e).__name__}: {e}). Finished hours are kept.",
                            warnings, [str(k) for k in done])
    finally:
        if lock.held:
            if status_written and vocab is not None and not finished:
                try:
                    reports.write_combined(root, fp, vocab, "INTERRUPTED – analyse the folder again to continue")
                except Exception as e:
                    log.error("Could not update the report status: %s", e)
            lock.release()
