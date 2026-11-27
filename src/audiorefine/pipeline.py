"""Chronological, hour-by-hour processing of one folder.

For every clock hour that contains recordings, oldest first:

  collect the hour's pieces → enhance → stitch → exact 10-minute FLAC segments → transcribe
  (with context from the same recording across the hour boundary) → hour transcript → manifest
  → validate → finalize the hour folder → regenerate the day transcript → regenerate the master
  transcript → archive originals whose every piece is finished → next hour.

The speech model is loaded once and reused for all hours. Results of an hour are complete and
usable as soon as it is finalized; an interruption only loses the hour being worked on.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import socket
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, analysis, batch, enhance, hardware, hourbatch, hours, hourtext, segments, transcript, vad
from .constants import ACTIVE_MODEL, ASR_SR, DECIMATION, MASTER_SR, PROCESSED_DIR, REFINED_DIR, SEGMENT_SECONDS
from .decode import DecodeError, decode_to_raw
from .discovery import ScanResult, scan_folder
from .downloads import gpu_runtime_installed, gpu_runtime_path, model_installed, model_path
from .hours import (DAY_TRANSCRIPT, HOUR_MANIFEST, HOUR_TRANSCRIPT, MASTER_TRANSCRIPT, MANIFEST_VERSION, STATUS_FILE,
                    WORDS_FILE, Coverage, HourKey, hour_key_of, load_complete_manifests, split_into_hours)
from .paths import work_root
from .planner import Plan, PlannedSource
from .progress import CancelledError, CancelToken
from .rawaudio import Decimator3, RawWriter, raw_length, read_raw
from .rectime import fmt_clock
from .transcriber import Transcriber
from .transcript import SegmentFile, TSegment, TWord

log = logging.getLogger("audiorefine.pipeline")

CTX_BEFORE_S = 30.0      # recognition context taken from the same recording before an hour piece
CTX_AFTER_S = 15.0       # … and after it


@dataclass
class Outcome:
    status: str                   # success | failed | cancelled | archive_incomplete
    message: str
    batch: str | None = None      # kept for compatibility: last finished hour
    refined_dir: str | None = None
    warnings: list[str] = field(default_factory=list)
    log_file: str | None = None
    hours_done: list[str] = field(default_factory=list)
    master: str | None = None


@dataclass
class Options:
    allow_gpu: bool = True
    model_dir: Path | None = None   # override (tests); otherwise the installed ACTIVE_MODEL
    model_name: str = ""
    workers: int = 0


def resolve_model(opts: Options) -> tuple[Path, str]:
    env = os.environ.get("AUDIOREFINE_MODEL_DIR")
    if opts.model_dir:
        return Path(opts.model_dir), opts.model_name or Path(opts.model_dir).name
    if env:
        return Path(env), opts.model_name or Path(env).name
    if not model_installed(ACTIVE_MODEL):
        raise RuntimeError("MODEL_MISSING")
    return model_path(ACTIVE_MODEL), "Whisper large-v3"


def _disk_free(p: Path) -> int:
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return 1 << 62


def _fmt_dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ------------------------------------------------------------------------------ progress
class HourProgress:
    """Overall progress weighted by the amount of audio of each hour."""

    STAGES = {"prepare": (0.0, 0.08), "enhance": (0.08, 0.30), "transcribe": (0.30, 0.94), "write": (0.94, 1.0)}

    def __init__(self, emit: Callable[[dict], None], hour_work: dict[HourKey, float]):
        self.emit = emit
        self.work = dict(hour_work)
        self.done_work = 0.0
        self.hours_done = 0
        self.dates_total = len({k.date for k in self.work})
        self.dates_done: set[str] = set()
        self.key: HourKey | None = None
        self.t0 = time.monotonic()
        self._last = 0.0
        self.hour_frac = 0.0
        self.label = ""
        self.detail = ""

    def add_hour(self, key: HourKey, work: float):
        self.work[key] = self.work.get(key, 0.0) + work
        self.dates_total = len({k.date for k in self.work})

    def total(self) -> float:
        return max(1e-6, sum(self.work.values()))

    def start_hour(self, key: HourKey):
        self.key = key
        self.hour_frac = 0.0
        self.stage("prepare", 0.0, "")

    def stage(self, name: str, frac: float, detail: str | None = None, label: str | None = None):
        a, b = self.STAGES[name]
        self.hour_frac = a + (b - a) * max(0.0, min(1.0, frac))
        self.label = label or {"prepare": "Preparing recordings", "enhance": "Enhancing speech",
                               "transcribe": "Transcribing speech", "write": "Writing and checking results"}[name]
        if detail is not None:
            self.detail = detail
        self.send()

    def finish_hour(self):
        self.done_work += self.work.get(self.key, 0.0)
        self.hours_done += 1
        remaining_dates = {k.date for k in self.work if k > self.key}
        if self.key.date not in remaining_dates:
            self.dates_done.add(self.key.date)
        self.hour_frac = 0.0
        self.send(force=True)

    def fraction(self) -> float:
        cur = self.work.get(self.key, 0.0) * self.hour_frac if self.key else 0.0
        return min(1.0, (self.done_work + cur) / self.total())

    def send(self, force: bool = False, label: str | None = None):
        now = time.monotonic()
        if not force and now - self._last < 0.25:
            return
        self._last = now
        el = now - self.t0
        p = self.fraction()
        eta = el * (1 - p) / p if p > 0.03 and el > 20 else None
        self.emit({
            "type": "progress", "stage": "hour", "label": label or self.label, "detail": self.detail,
            "fraction": round(p, 4), "hour_fraction": round(self.hour_frac, 4),
            "current_date": self.key.date if self.key else None,
            "current_hour": self.key.label if self.key else None,
            "hours_done": self.hours_done, "hours_total": len(self.work),
            "days_done": len(self.dates_done), "days_total": self.dates_total,
            "elapsed": round(el, 1), "eta": round(eta, 0) if eta is not None else None,
        })


# ------------------------------------------------------------------------------ per-recording cache
@dataclass
class SourceCache:
    raw48: Path
    raw16: Path
    samples: int
    probs: np.ndarray
    regions16: list[tuple[int, int]]
    plan: analysis.EnhancementPlan
    speech_frames: np.ndarray
    pieces: list[hours.Piece]
    warnings: list[str]


class Run:
    def __init__(self, root: Path, emit, cancel: CancelToken, opts: Options, log_file: Path | None):
        self.root = root
        self.emit = emit
        self.cancel = cancel
        self.opts = opts
        self.log_file = log_file
        self.warnings: list[str] = []
        self.work: Path | None = None
        self.cache: dict[str, SourceCache] = {}
        self.transcriber: Transcriber | None = None
        self.lm = None
        self.model_label = ""
        self.hours_done: list[HourKey] = []
        self.run_last_word: dict[tuple[str, int], float] = {}
        self.workers = opts.workers or max(1, os.cpu_count() or 2)
        self.status_started = datetime.now().strftime("%Y-%m-%d %H:%M")

    def chk(self):
        self.cancel.check()

    # -------------------------------------------------------------- combined transcripts / status
    def write_combined(self, state: str, planned_total: int, dates: set[str] | None = None) -> Path:
        manifests = load_complete_manifests(self.root)
        refined = self.root / REFINED_DIR
        refined.mkdir(parents=True, exist_ok=True)
        by_date: dict[str, list[dict]] = {}
        for k, m in manifests.items():
            by_date.setdefault(k.date, []).append(m)
        for d, ms in by_date.items():
            if dates is None or d in dates:
                hourtext.atomic_write_text(refined / d / DAY_TRANSCRIPT, hourtext.render_day(d, ms))
        done = len(self.hours_done)
        if state == "processing":
            line = (f"PROCESSING IN PROGRESS (started {self.status_started}) – {done} of {planned_total} hour(s) of "
                    "this run finished so far. This file is updated after every finished hour. If AudioRefine is "
                    "no longer running, the run was interrupted: process the folder again to continue.")
        elif state == "interrupted":
            line = (f"INTERRUPTED ({datetime.now().strftime('%Y-%m-%d %H:%M')}) – {done} of {planned_total} hour(s) "
                    "of the last run were finished. Process the folder again to continue with the rest.")
        else:
            line = f"COMPLETE – all recordings found in the folder have been processed ({len(manifests)} hour(s))."
        master = refined / MASTER_TRANSCRIPT
        hourtext.write_master(master, manifests, line)
        hours.write_json_atomic(refined / STATUS_FILE, {
            "app": "AudioRefine", "version": __version__, "state": state, "updated": datetime.now().isoformat(timespec="seconds"),
            "hours_completed_total": len(manifests), "hours_planned_this_run": planned_total,
            "hours_finished_this_run": [str(k) for k in self.hours_done]})
        return master

    # -------------------------------------------------------------- recordings
    def prepare_source(self, ps: PlannedSource, prog: HourProgress) -> SourceCache:
        if ps.sha in self.cache:
            return self.cache[ps.sha]
        name = ps.info.name
        prog.stage("prepare", 0.1, f"Decoding {name}")
        raw48 = self.work / f"src_{ps.sha[:16]}.f32"
        raw16 = self.work / f"src_{ps.sha[:16]}_16k.f32"
        dec = Decimator3()
        try:
            with RawWriter(raw48) as w, RawWriter(raw16) as w16:
                r = decode_to_raw(ps.info.path, w, on_samples=lambda x: w16.write(dec.process(x)), cancel=self.chk)
                w16.write(dec.flush())
        except CancelledError:
            raise
        except DecodeError as e:
            raise RuntimeError(f"{name} could not be decoded: {e}") from e
        except Exception as e:
            raise RuntimeError(f"{name} could not be decoded ({type(e).__name__}: {e})") from e
        for w_ in r.warnings:
            self.warnings.append(f"{name}: {w_}")
            log.warning("%s: %s", name, w_)
        prog.stage("prepare", 0.5, f"Detecting speech in {name}")
        probs = vad.speech_probabilities(raw16, cancel=self.chk)
        regions = vad.speech_regions(probs, raw_length(raw16))
        prog.stage("prepare", 0.8, f"Analysing {name}")
        a, sp = analysis.analyse_source(raw48, probs, 0)
        plan = analysis.plan_enhancement(a)
        pieces = split_into_hours(ps.rt.start, r.samples)
        log.info("Prepared %s: %d samples (%.1f s, %d Hz/%d ch, downmix %s); starts %s [%s%s]; hours %s; "
                 "speech %.1f s; analysis %s; plan: %s", name, r.samples, r.samples / MASTER_SR, r.native_rate,
                 r.native_channels, r.downmix, fmt_clock(ps.rt.start), ps.rt.basis,
                 ", estimated" if ps.rt.estimated else "", [str(p.key) for p in pieces],
                 sum(b - x for x, b in regions) / ASR_SR,
                 {k: (round(v, 1) if isinstance(v, float) else v) for k, v in a.__dict__.items()}, "; ".join(plan.reasons))
        c = SourceCache(raw48, raw16, r.samples, probs, regions, plan, sp, pieces, r.warnings)
        self.cache[ps.sha] = c
        return c

    def release_source(self, sha: str):
        c = self.cache.pop(sha, None)
        if c:
            for p in (c.raw48, c.raw16):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # -------------------------------------------------------------- recognition of one piece
    def transcribe_piece(self, ps: PlannedSource, c: SourceCache, piece: hours.Piece, out_start: int,
                         cov: Coverage, prog: HourProgress, frac0: float, frac1: float) -> tuple[list[dict], list, float | None]:
        lo = piece.src_start / MASTER_SR
        hi = piece.src_end / MASTER_SR
        n16 = raw_length(c.raw16)
        ctx_a = max(0, int((lo - CTX_BEFORE_S) * ASR_SR))
        ctx_b = min(n16, int(math.ceil((hi + CTX_AFTER_S) * ASR_SR)))
        regs = [(max(a, ctx_a), min(b, ctx_b)) for a, b in c.regions16 if b > ctx_a and a < ctx_b]
        regs = [(a, b) for a, b in regs if b > a]
        prev = self.run_last_word.get((ps.sha, piece.src_start), cov.last_word_end.get((ps.sha, piece.src_start)))
        lo_eff = max(lo - 1.0, prev - 0.05) if prev is not None and piece.src_start > 0 else lo
        segs: list[TSegment] = []
        if regs:
            segs = self.transcriber.transcribe(
                c.raw16, regs, probs=c.probs, cancel=self.chk,
                progress=lambda f: prog.stage("transcribe", frac0 + (frac1 - frac0) * f, f"Transcribing {ps.info.name}"))
        off = out_start / MASTER_SR
        out_end = off + piece.samples / MASTER_SR
        to_out = lambda t: min(out_end, max(off, off + t - lo))
        entries, words_out = [], []
        last_end = None
        for s in segs:
            if s.marker or not s.words:
                if lo_eff <= s.start < hi:
                    entries.append({"t0": round(to_out(s.start), 3), "t1": round(to_out(min(s.end, hi)), 3),
                                    "text": s.text, "uncertain": s.uncertain, "marker": s.marker, "lang": s.language,
                                    "cont_prev": False, "cont_next": False})
                    words_out.append([])
                continue
            kept = [w for w in s.words if lo_eff <= w.start < hi]
            if not kept:
                continue
            first_i = s.words.index(kept[0])
            last_i = s.words.index(kept[-1])
            text = "".join(w.text for w in kept).strip() if len(kept) != len(s.words) else s.text
            entries.append({"t0": round(to_out(kept[0].start), 3), "t1": round(to_out(max(kept[-1].end, kept[0].start)), 3),
                            "text": text, "uncertain": s.uncertain, "marker": False, "lang": s.language,
                            "cont_prev": first_i > 0, "cont_next": last_i < len(s.words) - 1})
            words_out.append([[round(to_out(w.start), 3), round(to_out(max(w.end, w.start)), 3), w.text, round(w.prob, 3)]
                              for w in kept])
            last_end = max(last_end or 0.0, kept[-1].end)
        if last_end is None and prev is not None:
            last_end = prev
        if last_end is not None:
            self.run_last_word[(ps.sha, piece.src_end)] = last_end
        return entries, words_out, last_end

    # -------------------------------------------------------------- one hour
    def process_hour(self, key: HourKey, plan: Plan, prog: HourProgress, planned_total: int) -> None:
        root = self.root
        existing = plan.manifests.get(key)
        cov = Coverage.build(plan.manifests)
        prog.start_hour(key)
        log.info("=== Hour %s (%s): %d new recording piece(s)%s", key, key.label, len(plan.hours.get(key, [])),
                 f", updating the finished hour ({len(existing['pieces'])} existing piece(s))" if existing else "")
        # collect pieces: (start instant, name, kind, data)
        items = []
        for ps in plan.hours.get(key, []):
            c = self.prepare_source(ps, prog)
            for p in c.pieces:
                if p.key == key and not cov.covered(ps.sha, p.src_start, p.src_end):
                    items.append((p.start.timestamp(), ps.info.name, "new", (ps, c, p)))
        if existing:
            old_words = (hours.read_json(key.dir(root) / WORDS_FILE) or {}).get("words", [])
            for i, op in enumerate(existing["pieces"]):
                items.append((datetime.fromisoformat(op["start"]).timestamp(), op["source"], "old", (i, op)))
        if not any(kind == "new" for _, _, kind, _ in items):
            log.info("Hour %s: nothing new to add (recording timing changed after exact decoding)", key)
            return
        items.sort(key=lambda x: (x[0], x[1]))
        staging = hourbatch.create_hour_staging(root, key)
        hour_raw = self.work / f"hour_{key.date}_{key.hour:02d}.f32"
        piece_recs, entries, words, exact = [], [], [], []
        new_total = sum(it[3][2].samples for it in items if it[2] == "new") or 1
        new_done = 0
        try:
            with RawWriter(hour_raw) as hw:
                for _, name, kind, data in items:
                    self.chk()
                    out_start = hw.count
                    if kind == "old":
                        i, op = data
                        n = op["out_end"] - op["out_start"]
                        for blk in segments.read_int16(key.dir(root), existing["segments"], op["out_start"], op["out_end"]):
                            hw.write(blk.astype(np.float32) / 32767.0)
                        if hw.count - out_start != n:
                            raise RuntimeError(f"Existing audio of {key} is incomplete; the hour cannot be updated.")
                        exact.append((out_start, hw.count))
                        delta = (out_start - op["out_start"]) / MASTER_SR
                        rec = dict(op, out_start=out_start, out_end=out_start + n)
                        pi = len(piece_recs)
                        piece_recs.append(rec)
                        for e, w in zip(existing.get("entries", []), old_words or [[]] * len(existing.get("entries", []))):
                            if e.get("piece") == i:
                                entries.append(dict(e, t0=round(e["t0"] + delta, 3), t1=round(e["t1"] + delta, 3), piece=pi))
                                words.append([[round(a + delta, 3), round(b + delta, 3), t, pr] for a, b, t, pr in w])
                        continue
                    ps, c, p = data
                    f0 = new_done / new_total
                    f1 = (new_done + p.samples) / new_total
                    t0 = time.time()
                    rep = enhance.enhance_range(c.raw48, p.src_start, p.src_end, c.plan, c.speech_frames, self.work, hw,
                                                self.workers, cancel=self.chk,
                                                progress=lambda f, a=f0, b=f1, n=name: prog.stage("enhance", a + (b - a) * f, f"Enhancing {n}"))
                    if hw.count - out_start != p.samples:
                        raise RuntimeError("Internal error: enhanced audio length differs from the recording part.")
                    log.info("Enhanced %s [%d:%d] in %.1f s: %s", name, p.src_start, p.src_end, time.time() - t0, rep)
                    t0 = time.time()
                    ents, wds, last_end = self.transcribe_piece(ps, c, p, out_start, cov, prog, f0, f1)
                    log.info("Transcribed %s part (%.1f s) in %.1f s: %d entries", name, p.samples / MASTER_SR,
                             time.time() - t0, len(ents))
                    pi = len(piece_recs)
                    piece_recs.append(hours.piece_record(p, name, ps.sha, ps.info.size, c.samples, out_start, ps.rt,
                                                         round(last_end, 3) if last_end is not None else None))
                    for e in ents:
                        e["piece"] = pi
                    entries.extend(ents)
                    words.extend(wds)
                    new_done += p.samples
            total = raw_length(hour_raw)
            # ---------------------------------------------------------- outputs of the hour
            prog.stage("write", 0.1, "Writing audio files")
            files = segments.plan_segments(total)
            seg_res = segments.write_segments(hour_raw, staging, files, cancel=self.chk, exact_ranges=exact)
            order = sorted(range(len(entries)), key=lambda i: (entries[i]["t0"], entries[i]["t1"]))
            entries = [entries[i] for i in order]
            words = [words[i] for i in order]
            manifest = {
                "app": "AudioRefine", "kind": "hour", "manifest_version": MANIFEST_VERSION, "version": __version__,
                "date": key.date, "hour": key.hour, "state": "writing",
                "updated": datetime.now().isoformat(timespec="seconds"),
                "model": self.model_label, "sample_rate": MASTER_SR, "total_samples": total,
                "pieces": piece_recs,
                "segments": [{"audio": f.audio_name, "txt": f.stem + ".txt", "srt": f.stem + ".srt",
                              "start_sample": f.start_sample, "end_sample": f.end_sample, "samples": r["samples"],
                              "pcm_sha256": r["pcm_sha256"]} for f, r in zip(files, seg_res)],
                "entries": entries,
            }
            prog.stage("write", 0.5, "Writing transcripts")
            self.write_hour_texts(staging, manifest, words, files)
            hours.write_json_atomic(staging / WORDS_FILE, {"words": words})
            problems = self.validate_hour(staging, manifest, files)
            if problems:
                for pr in problems:
                    log.error("Validation %s: %s", key, pr)
                raise RuntimeError(f"The results for {key.text} failed validation: {problems[0]}")
            for it in items:
                if it[2] == "new":
                    ps = it[3][0]
                    st = ps.info.path.stat()
                    if st.st_size != ps.info.size or st.st_mtime != ps.info.mtime:
                        raise RuntimeError(f"{ps.info.name} changed while it was being processed. Process the folder again.")
            manifest["state"] = "complete"
            hours.write_json_atomic(staging / HOUR_MANIFEST, manifest)
            hourbatch.commit_hour(root, key, staging)
            staging = None
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            try:
                os.remove(hour_raw)
            except OSError:
                pass
        log.info("Hour %s finalized: %d audio file(s), %s of audio, %d transcript entries", key, len(files),
                 _fmt_dur(total / MASTER_SR), len(entries))
        self.hours_done.append(key)
        plan.manifests[key] = hours.read_json(key.dir(root) / HOUR_MANIFEST)
        prog.stage("write", 0.9, "Updating day and master transcripts")
        master = self.write_combined("processing", planned_total, dates={key.date})
        self.emit({"type": "hour_done", "date": key.date, "hour": key.label, "folder": str(key.dir(root)),
                   "master": str(master), "hours_done": len(self.hours_done), "hours_total": planned_total})
        # archive originals whose every part is now finished
        cov2 = Coverage.build(plan.manifests)
        for ps in {id(it[3][0]): it[3][0] for it in items if it[2] == "new"}.values():
            if cov2.fully_covered(ps.sha):
                ok, msg = hourbatch.archive_original(root, ps.info.path, ps.sha, ps.info.size, ps.start_key)
                if ok:
                    log.info("Archived %s → %s", ps.info.name, msg)
                    ps.status = "archived"
                else:
                    self.warnings.append(msg)
                    log.error("Archiving %s failed: %s", ps.info.name, msg)
                self.release_source(ps.sha)
        prog.finish_hour()

    def write_hour_texts(self, out: Path, m: dict, words: list, files: list[SegmentFile]) -> None:
        segs = []
        for e, w in zip(m["entries"], words):
            ws = [TWord(a, b, t, pr) for a, b, t, pr in w]
            text = e["text"]
            if e.get("cont_prev"):
                text = "… " + text
                if ws:
                    ws[0] = TWord(ws[0].start, ws[0].end, " …" + ws[0].text, ws[0].prob)
            if e.get("cont_next"):
                text = text + " …"
                if ws:
                    ws[-1] = TWord(ws[-1].start, ws[-1].end, ws[-1].text + " …", ws[-1].prob)
            segs.append(TSegment(e["t0"], e["t1"], text, ws, uncertain=e.get("uncertain", False),
                                 marker=e.get("marker", False), language=e.get("lang", "")))
        pieces = transcript.split_into_files(segs, files)
        pwords = transcript.words_for_pieces(segs, files)
        for f, ps, pw, sg in zip(files, pieces, pwords, m["segments"]):
            hourtext.atomic_write_text(out / f"{f.stem}.txt", hourtext.render_segment_txt(m, sg, ps))
            hourtext.atomic_write_text(out / f"{f.stem}.srt", transcript.render_segment_srt(f, ps, pw))
        hourtext.atomic_write_text(out / HOUR_TRANSCRIPT, hourtext.render_hour(m))

    def validate_hour(self, d: Path, m: dict, files: list[SegmentFile]) -> list[str]:
        problems = []
        total = 0
        for i, (f, sg) in enumerate(zip(files, m["segments"])):
            self.chk()
            n = f.end_sample - f.start_sample
            if i < len(files) - 1 and n != SEGMENT_SECONDS * MASTER_SR:
                problems.append(f"{f.audio_name}: not exactly {SEGMENT_SECONDS // 60} minutes")
            err = segments.verify_segment(d / f.audio_name, n, sg["pcm_sha256"])
            if err:
                problems.append(err)
            total += n
            for ext in (".txt", ".srt"):
                try:
                    (d / (f.stem + ext)).read_text(encoding="utf-8")
                except Exception as e:
                    problems.append(f"{f.stem}{ext}: unreadable ({e})")
        if total != m["total_samples"] or total != sum(p["out_end"] - p["out_start"] for p in m["pieces"]):
            problems.append("audio length does not equal the sum of the recording parts")
        pos = 0
        for p in m["pieces"]:
            if p["out_start"] != pos:
                problems.append("recording parts are not contiguous")
            pos = p["out_end"]
        try:
            if not (d / HOUR_TRANSCRIPT).read_text(encoding="utf-8").strip():
                problems.append("hour transcript is empty")
        except Exception as e:
            problems.append(f"{HOUR_TRANSCRIPT}: unreadable ({e})")
        return problems


# ------------------------------------------------------------------------------ entry point
def process_folder(root: Path, emit: Callable[[dict], None], cancel: CancelToken, opts: Options | None = None,
                   log_file: Path | None = None) -> Outcome:
    opts = opts or Options()
    root = Path(root).resolve()
    run = Run(root, emit, cancel, opts, log_file)
    lock = batch.FolderLock(root)
    planned_total = 0
    state = "processing"
    t_start = time.time()
    try:
        log.info("AudioRefine %s - processing folder %s (hourly mode)", __version__, root)
        log.info("Python %s on %s; local time zone %s", sys.version.split()[0], sys.platform, time.strftime("%Z"))
        if not root.is_dir():
            return Outcome("failed", "The selected folder does not exist.")
        model_dir, run.model_label = resolve_model(opts)
        hw = hardware.detect()
        log.info("Hardware: %s", hw.to_dict())
        log.info("Enhancement helper: %s", enhance.arfilter_version())
        emit({"type": "progress", "stage": "prepare", "label": "Checking the folder", "detail": "", "fraction": 0.0,
              "elapsed": 0.0, "eta": None})
        from .sound.store import analysis_busy
        if analysis_busy(root):
            return Outcome("failed", "The sounds in this folder are being analysed right now. Process the folder "
                                     "when the sound analysis has finished. Nothing was changed.")
        lock.acquire()
        for note in hourbatch.recover(root):
            log.info("Recovery: %s", note)
        batch.cleanup_stale_staging(root, have_lock=True)
        for pend in batch.find_pending_archives(root):    # batches of AudioRefine 1.0
            batch.archive_sources(root, pend.batch, pend.details_path)
        wr = work_root()
        for name in batch.cleanup_stale_work(wr):
            log.info("Removed temporary files of an interrupted run: %s", name)

        scan: ScanResult = scan_folder(root, sha_of=lambda info: batch.sha256_of(info.path, run.chk))
        plan: Plan = scan.plan
        for u in scan.unreadable:
            log.warning("Not processed (unreadable): %s - %s", u.name, u.error)
        if scan.unreadable:
            run.warnings.append(f"{len(scan.unreadable)} file(s) could not be read and were left untouched: "
                                + ", ".join(u.name for u in scan.unreadable[:5]) + ("…" if len(scan.unreadable) > 5 else ""))
        for w in scan.warnings:
            log.warning(w)
            if "already processed earlier" in w:
                run.warnings.append(w)
        for ps in plan.sources:
            log.info("  %s | %s | starts %s [%s%s] | %s | hours %s", ps.info.name, ps.status, fmt_clock(ps.rt.start),
                     ps.rt.basis, ", estimated" if ps.rt.estimated else "", f"{ps.info.duration:.1f} s",
                     [str(k) for k in (ps.todo_hours or ps.all_hours)])
        # originals that are completely processed but were not archived (interrupted earlier)
        archived_now = 0
        for ps in plan.sources:
            if ps.status == "processed":
                ok, msg = hourbatch.archive_original(root, ps.info.path, ps.sha, ps.info.size, ps.start_key)
                log.info("Archive of already processed %s: %s", ps.info.name, msg)
                archived_now += ok
                if not ok:
                    run.warnings.append(msg)
        keys = plan.hour_keys
        planned_total = len(keys)
        # combined transcripts are regenerated from the manifests (repairs an interrupted update)
        run.write_combined("processing" if keys else "complete", planned_total)
        if not keys:
            master = root / REFINED_DIR / MASTER_TRANSCRIPT
            if archived_now:
                return Outcome("success", f"Moved {archived_now} already processed original recording(s) to "
                                          f"'{PROCESSED_DIR}'. There were no new recordings to process.",
                               warnings=run.warnings, master=str(master) if master.exists() else None)
            if plan.sources or plan.manifests:
                return Outcome("success", "Nothing new to process – all recordings in this folder have already been "
                                          "processed.", warnings=run.warnings,
                               master=str(master) if master.exists() else None)
            return Outcome("failed", "No readable audio recordings were found directly in the folder.",
                           warnings=run.warnings)
        log.info("Plan: %d hour(s) on %d day(s): %s", planned_total, len({k.date for k in keys}), [str(k) for k in keys])
        # disk space: decoded recordings of the longest span + one hour of work files
        longest = max((ps.info.duration for ps in plan.sources if ps.status in ("new", "partial")), default=0.0)
        need_work = int(longest * 256_000 + 3600 * 230_000) + (300 << 20)
        if _disk_free(wr) < need_work:
            return Outcome("failed", f"Not enough free disk space for temporary files on {wr.anchor or wr} "
                                     f"(about {need_work / 2**30:.1f} GB needed). Nothing was changed.",
                           warnings=run.warnings)
        need_out = int(sum(ps.info.duration for ps in plan.sources if ps.status in ("new", "partial")) * 100_000) + (100 << 20)
        if _disk_free(root) < need_out:
            return Outcome("failed", f"Not enough free disk space in the selected folder (about "
                                     f"{need_out / 2**30:.1f} GB needed). Nothing was changed.", warnings=run.warnings)
        run.work = wr / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run.work.mkdir(parents=True)
        batch.write_json_atomic(run.work / "owner.json", {"pid": os.getpid(), "host": socket.gethostname(),
                                                          "folder": str(root)})
        # model, loaded once for all hours
        emit({"type": "progress", "stage": "model", "label": "Loading speech recognition", "detail": run.model_label,
              "fraction": 0.0, "elapsed": round(time.time() - t_start, 1), "eta": None,
              "hours_done": 0, "hours_total": planned_total})
        gpu_rt = gpu_runtime_path() if gpu_runtime_installed() else None
        run.lm = hardware.load_model(model_dir, hw, allow_gpu=opts.allow_gpu, gpu_runtime=gpu_rt)
        log.info("Speech recognition: %s on %s (load %.1f s)%s", run.model_label, run.lm.description,
                 run.lm.load_seconds, f"; GPU not used: {run.lm.gpu_error}" if run.lm.gpu_error else "")
        emit({"type": "info", "device": run.lm.description})
        run.transcriber = Transcriber(run.lm.model)
        # work per hour ~ seconds of new audio in it (estimate)
        work = {}
        for k in keys:
            secs = 0.0
            for ps in plan.hours[k]:
                secs += max(1.0, min(3600.0, ps.info.duration / max(1, len(ps.todo_hours))))
            work[k] = secs
        prog = HourProgress(emit, work)
        i = 0
        while i < len(keys):
            run.chk()
            k = keys[i]
            run.process_hour(k, plan, prog, planned_total)
            # exact decoding may reveal a part of a recording in an hour that was not planned
            for ps in plan.sources:
                c = run.cache.get(ps.sha or "")
                if not c:
                    continue
                for p in c.pieces:
                    if p.key > k and p.key not in plan.hours:
                        plan.hours[p.key] = [ps]
                        prog.add_hour(p.key, p.samples / MASTER_SR)
                        log.info("Added hour %s for the end of %s", p.key, ps.info.name)
                    elif p.key > k and ps not in plan.hours[p.key]:
                        plan.hours[p.key].append(ps)
            keys = sorted(plan.hours)
            planned_total = len(keys)
            i = keys.index(k) + 1
        state = "complete"
        master = run.write_combined("complete", planned_total)
        not_archived = [ps.info.name for ps in plan.sources if ps.status in ("new", "partial") and ps.info.path.exists()]
        log.info("Transcription statistics: %s; languages %s", run.transcriber.stats, run.transcriber.policy.summary())
        log.info("Finished %d hour(s) in %s", len(run.hours_done), _fmt_dur(time.time() - t_start))
        last = run.hours_done[-1] if run.hours_done else None
        if not_archived:
            return Outcome("archive_incomplete",
                           f"All {len(run.hours_done)} hour(s) were processed, but {len(not_archived)} original "
                           f"recording(s) could not be moved to '{PROCESSED_DIR}'. They are not processed again and will "
                           "be moved next time you click Process.", str(last) if last else None,
                           str(root / REFINED_DIR), run.warnings, hours_done=[str(k) for k in run.hours_done],
                           master=str(master))
        return Outcome("success", f"Done. {len(run.hours_done)} hour(s) on {len({k.date for k in run.hours_done})} "
                                  f"day(s) were processed. The complete master transcript is ready.",
                       str(last) if last else None, str(root / REFINED_DIR), run.warnings,
                       hours_done=[str(k) for k in run.hours_done], master=str(master))
    except CancelledError:
        log.warning("Cancelled by the user")
        state = "interrupted"
        done = len(run.hours_done)
        return Outcome("cancelled", f"Processing was cancelled. {done} finished hour(s) are kept and usable; the "
                                    "unfinished hour and all remaining recordings were left unchanged. Process the "
                                    "folder again to continue.", warnings=run.warnings,
                       hours_done=[str(k) for k in run.hours_done],
                       master=str(root / REFINED_DIR / MASTER_TRANSCRIPT))
    except batch.SafetyError as e:
        state = "interrupted"
        log.error("Safety stop: %s", e)
        return Outcome("failed", str(e), warnings=run.warnings)
    except RuntimeError as e:
        if str(e) == "MODEL_MISSING":
            state = "untouched"
            return Outcome("failed", "MODEL_MISSING")
        state = "interrupted"
        log.error("Failed: %s\n%s", e, traceback.format_exc())
        return Outcome("failed", f"{e} Finished hours are kept; the remaining recordings were not changed.",
                       warnings=run.warnings, hours_done=[str(k) for k in run.hours_done])
    except OSError as e:
        state = "interrupted"
        log.error("Failed: %s\n%s", e, traceback.format_exc())
        if getattr(e, "errno", None) == 28:
            msg = "The disk is full. Free some space and try again."
        elif isinstance(e, PermissionError):
            msg = (f"A file could not be accessed ({e.filename or e}). It may be open in another program, or "
                   "access to the folder is denied.")
        else:
            msg = f"A file operation failed ({e})."
        return Outcome("failed", msg + " Finished hours are kept; the remaining recordings were not changed.",
                       warnings=run.warnings, hours_done=[str(k) for k in run.hours_done])
    except Exception as e:
        state = "interrupted"
        log.error("Unexpected error: %s\n%s", e, traceback.format_exc())
        return Outcome("failed", f"Unexpected error ({type(e).__name__}: {e}). Finished hours are kept; the remaining "
                                 "recordings were not changed.", warnings=run.warnings,
                       hours_done=[str(k) for k in run.hours_done])
    finally:
        if state == "interrupted" and lock.held:
            try:
                hourbatch.recover(root)
                run.write_combined("interrupted", planned_total)
            except Exception as e:
                log.error("Could not update the master transcript status: %s", e)
        if run.work is not None:
            shutil.rmtree(run.work, ignore_errors=True)
        lock.release()
