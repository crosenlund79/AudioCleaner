"""The complete processing run for one folder."""
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

from . import __version__, analysis, batch, enhance, hardware, segments, transcript, vad
from .constants import (ACTIVE_MODEL, ASR_SR, BATCH_LOG_NAME, COMPLETE_TRANSCRIPT, DECIMATION, DETAILS_NAME, FRAME,
                        MASTER_SR, PROCESSED_DIR, REFINED_DIR, SEGMENT_SECONDS)
from .decode import DecodeError, decode_to_raw
from .discovery import ScanResult, scan_folder
from .downloads import gpu_runtime_installed, gpu_runtime_path, model_installed, model_path
from .paths import work_root
from .progress import CancelledError, CancelToken, Progress, default_weights
from .rawaudio import Decimator3, RawWriter, raw_length
from .transcriber import Transcriber

log = logging.getLogger("audiorefine.pipeline")

LANG_NAMES = {"da": "Danish", "en": "English", "sv": "Swedish", "no": "Norwegian", "de": "German"}


@dataclass
class Outcome:
    status: str                   # success | failed | cancelled | archive_incomplete
    message: str
    batch: str | None = None
    refined_dir: str | None = None
    warnings: list[str] = field(default_factory=list)
    log_file: str | None = None


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


def process_folder(root: Path, emit: Callable[[dict], None], cancel: CancelToken, opts: Options | None = None,
                   log_file: Path | None = None) -> Outcome:
    opts = opts or Options()
    root = Path(root).resolve()
    prog = Progress(emit)
    t_start = time.time()
    warnings: list[str] = []
    staging: Path | None = None
    work: Path | None = None
    lock = batch.FolderLock(root)
    batch_name = None
    final_dir: Path | None = None
    committed = False

    def chk():
        cancel.check()

    try:
        log.info("AudioRefine %s - processing folder %s", __version__, root)
        log.info("Python %s on %s", sys.version.split()[0], sys.platform)
        if not root.is_dir():
            return Outcome("failed", "The selected folder does not exist.")
        model_dir, model_label = resolve_model(opts)
        hw = hardware.detect()
        log.info("Hardware: %s", hw.to_dict())
        log.info("Enhancement helper: %s", enhance.arfilter_version())
        prog.start("prepare", "Scanning folder")
        lock.acquire()
        removed = batch.cleanup_stale_staging(root, have_lock=True)
        for name in removed:
            log.info("Removed incomplete output from an earlier interrupted run: %s", name)
        # finish archiving of earlier batches first
        finished_earlier = 0
        for pend in batch.find_pending_archives(root):
            log.info("Completing archiving of earlier batch %s (%d file(s))", pend.batch, len(pend.names))
            r = batch.archive_sources(root, pend.batch, pend.details_path)
            finished_earlier += len([n for n in r.moved if n in pend.names])
            if r.failed:
                warnings.append(f"Earlier batch {pend.batch}: {len(r.failed)} original(s) still could not be moved.")
        scan: ScanResult = scan_folder(root)
        if scan.unreadable:
            for u in scan.unreadable:
                log.warning("Not processed (unreadable): %s - %s", u.name, u.error)
            warnings.append(f"{len(scan.unreadable)} file(s) could not be read and were left untouched: "
                            + ", ".join(u.name for u in scan.unreadable[:5]) + ("..." if len(scan.unreadable) > 5 else ""))
        sources = scan.sources
        if not sources:
            if finished_earlier:
                return Outcome("success", f"Moved {finished_earlier} original recording(s) of an earlier batch to "
                                          f"'{PROCESSED_DIR}'. There were no new recordings to process.", warnings=warnings)
            return Outcome("failed", "No readable audio recordings were found directly in the folder.", warnings=warnings)
        log.info("Order basis: %s", scan.order_basis)
        for i, s in enumerate(sources, 1):
            log.info("  %2d. %s | %s/%s %d Hz %d ch | ~%.1f s | %d bytes | name time %s | meta time %s",
                     i, s.name, s.container, s.codec, s.sample_rate, s.channels, s.duration, s.size,
                     s.name_datetime, s.meta_datetime)
        for w in scan.warnings:
            log.warning(w)

        # space check. Work files (measured): decoded 48 kHz float sources 192 kB/s + 16 kHz copy 64 kB/s,
        # plus the filtered copy of the longest recording and parallel chunks (~230 kB/s of it).
        # Outputs: FLAC 16-bit mono <= 96 kB/s.
        dur = scan.total_duration
        longest = max((s.duration for s in sources), default=0.0)
        need_work = int(dur * 256_000 + longest * 230_000) + (300 << 20)
        need_out = int(dur * 100_000) + (100 << 20)
        wr = work_root()
        for name in batch.cleanup_stale_work(wr):
            log.info("Removed temporary files of an interrupted run: %s", name)
        if _disk_free(wr) < need_work:
            return Outcome("failed", f"Not enough free disk space for temporary files on {wr.anchor or wr} "
                                     f"(about {need_work / 2**30:.1f} GB needed). Nothing was changed.", warnings=warnings)
        if _disk_free(root) < need_out:
            return Outcome("failed", f"Not enough free disk space in the selected folder (about "
                                     f"{need_out / 2**30:.1f} GB needed). Nothing was changed.", warnings=warnings)

        batch_name = batch.new_batch_name(root)
        staging = batch.create_staging(root, batch_name)
        final_dir = root / REFINED_DIR / batch_name
        work = wr / f"{batch_name}_{uuid.uuid4().hex[:8]}"
        work.mkdir(parents=True)
        batch.write_json_atomic(work / "owner.json", {"pid": os.getpid(), "host": socket.gethostname(),
                                                      "folder": str(root)})
        log.info("Batch %s; staging %s; work %s", batch_name, staging, work)
        details_path = staging / DETAILS_NAME
        details = {
            "app": "AudioRefine", "version": __version__, "batch": batch_name, "state": "processing",
            "created": datetime.now().isoformat(timespec="seconds"), "folder": str(root),
            "order_basis": scan.order_basis, "order_warnings": scan.warnings,
            "sources": [], "hardware": hw.to_dict(),
        }

        # ---------------------------------------------------------------- hash originals
        total_bytes = sum(s.size for s in sources) or 1
        done_bytes = 0
        for s in sources:
            chk()
            prog.update(done_bytes / total_bytes, f"Checking {s.name}")
            sha = batch.sha256_of(s.path, chk)
            done_bytes += s.size
            details["sources"].append({"name": s.name, "size": s.size, "mtime": s.mtime, "sha256": sha,
                                       "container": s.container, "codec": s.codec, "sample_rate": s.sample_rate,
                                       "channels": s.channels, "bit_rate": s.bit_rate,
                                       "embedded_time": s.meta_datetime.isoformat() if s.meta_datetime else None,
                                       "name_time": s.name_datetime.isoformat() if s.name_datetime else None})
        batch.write_json_atomic(details_path, details)

        # ---------------------------------------------------------------- model
        prog.start("model", model_label)
        gpu_rt = gpu_runtime_path() if gpu_runtime_installed() else None
        lm = hardware.load_model(model_dir, hw, allow_gpu=opts.allow_gpu, gpu_runtime=gpu_rt)
        log.info("Speech recognition: %s on %s (load %.1f s)%s", model_label, lm.description, lm.load_seconds,
                 f"; GPU not used: {lm.gpu_error}" if lm.gpu_error else "")
        details["transcription"] = {"model": model_label, "model_dir": str(model_dir), "device": lm.device,
                                    "compute_type": lm.compute_type, "device_description": lm.description,
                                    "gpu_error": lm.gpu_error}
        prog.set_weights(default_weights(lm.device == "cuda"))
        emit({"type": "info", "device": lm.description})

        # ---------------------------------------------------------------- decode + stitch
        prog.start("decode")
        asr_raw = work / "asr16.f32"
        dec = Decimator3()
        offsets = []
        pos = 0
        with RawWriter(asr_raw) as asr_w:
            for i, (s, d) in enumerate(zip(sources, details["sources"])):
                chk()
                prog.update(i / len(sources), f"Decoding {s.name}")
                raw = work / f"src_{i:04d}.f32"
                try:
                    with RawWriter(raw) as w:
                        r = decode_to_raw(s.path, w, on_samples=lambda x: asr_w.write(dec.process(x)), cancel=chk)
                except CancelledError:
                    raise
                except DecodeError as e:
                    raise RuntimeError(f"{s.name} could not be decoded: {e}") from e
                except Exception as e:
                    raise RuntimeError(f"{s.name} could not be decoded ({type(e).__name__}: {e})") from e
                d.update({"decoded_samples": r.samples, "start_sample": pos, "end_sample": pos + r.samples,
                          "start": pos / MASTER_SR, "end": (pos + r.samples) / MASTER_SR, "downmix": r.downmix,
                          "decode_warnings": r.warnings})
                for w_ in r.warnings:
                    warnings.append(f"{s.name}: {w_}")
                    log.warning("%s: %s", s.name, w_)
                log.info("Decoded %s: %d samples (%.3f s) from %d Hz/%d ch, downmix=%s", s.name, r.samples,
                         r.samples / MASTER_SR, r.native_rate, r.native_channels, r.downmix)
                offsets.append(pos)
                pos += r.samples
            asr_w.write(dec.flush())
        total_samples = pos
        if total_samples == 0:
            raise RuntimeError("The recordings contain no audio.")
        log.info("Stitched timeline: %d samples = %s", total_samples, _fmt_dur(total_samples / MASTER_SR))

        # ---------------------------------------------------------------- VAD
        prog.start("vad")
        probs = vad.speech_probabilities(asr_raw, progress=lambda f: prog.update(f), cancel=chk)
        regions = vad.speech_regions(probs, raw_length(asr_raw))
        speech_s = sum(b - a for a, b in regions) / ASR_SR
        log.info("Voice activity: %d speech regions, %.1f s of %.1f s", len(regions), speech_s, total_samples / MASTER_SR)
        prog.set_weights(estimate_weights(prog, total_samples / MASTER_SR, speech_s, lm.device == "cuda",
                                          hardware.cpu_threads(), opts.workers or (os.cpu_count() or 2)))

        # ---------------------------------------------------------------- analysis + enhancement
        prog.start("analyse")
        plans = []
        for i, d in enumerate(details["sources"]):
            chk()
            prog.update(i / len(sources), f"Analysing {d['name']}")
            raw = work / f"src_{i:04d}.f32"
            win_off = int(round(offsets[i] / DECIMATION / vad.WINDOW))
            a, sp = analysis.analyse_source(raw, probs, win_off)
            plan = analysis.plan_enhancement(a)
            plans.append((plan, sp))
            d["analysis"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.__dict__.items()}
            d["enhancement"] = plan.to_dict()
            log.info("Analysis %s: %s", d["name"], d["analysis"])
            log.info("Plan %s: %s", d["name"], "; ".join(plan.reasons))
        prog.start("enhance")
        master = work / "master48.f32"
        workers = opts.workers or max(1, (os.cpu_count() or 2))
        with RawWriter(master) as mw:
            for i, d in enumerate(details["sources"]):
                chk()
                raw = work / f"src_{i:04d}.f32"
                frac0 = d["start_sample"] / total_samples
                span = (d["end_sample"] - d["start_sample"]) / total_samples
                t0 = time.time()
                rep = enhance.enhance_source(raw, plans[i][0], plans[i][1], work, mw, workers, i == 0,
                                             i == len(sources) - 1,
                                             progress=lambda f, a=frac0, b=span: prog.update(a + b * f, f"Enhancing {d['name']}"),
                                             cancel=chk)
                d["enhancement_report"] = rep
                log.info("Enhanced %s in %.1f s: %s", d["name"], time.time() - t0, rep)
                os.remove(raw)
        if raw_length(master) != total_samples:
            raise RuntimeError("Internal error: enhanced audio length does not match the decoded audio.")

        # ---------------------------------------------------------------- transcription
        prog.start("transcribe")
        t0 = time.time()
        tr = Transcriber(lm.model)
        junctions = [o // DECIMATION for o in offsets[1:]]
        segs = tr.transcribe(asr_raw, regions, probs=probs, progress=lambda f: prog.update(f), cancel=chk,
                             breaks=junctions)
        dt = time.time() - t0
        langs = [k for k, _ in sorted(tr.policy.counts.items(), key=lambda kv: -kv[1])]
        details["transcription"].update({"seconds": round(dt, 1), "speech_seconds": round(speech_s, 1),
                                         "realtime_factor": round(dt / max(speech_s, 1e-6), 3),
                                         "language_policy": tr.policy.summary(), "integrity": tr.stats})
        log.info("Transcription finished in %.1f s (%.2fx speech duration): %s; languages %s", dt,
                 dt / max(speech_s, 1e-6), tr.stats, tr.policy.summary())

        # ---------------------------------------------------------------- outputs
        prog.start("write_audio")
        seg_files = segments.plan_segments(total_samples)
        seg_results = segments.write_segments(master, staging, seg_files, progress=lambda f: prog.update(f), cancel=chk)
        prog.start("write_text")
        spans = [transcript.SourceSpan(d["name"], d["start"], d["end"]) for d in details["sources"]]
        meta = {"batch": batch_name, "duration": total_samples / MASTER_SR, "version": __version__,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model_label,
                "languages": [f"{LANG_NAMES.get(l, l)} ({l})" for l in langs]}
        transcript.write_transcripts(staging, segs, spans, seg_files, meta)
        details["outputs"] = {
            "segment_seconds": SEGMENT_SECONDS, "sample_rate": MASTER_SR, "format": "FLAC 16-bit mono",
            "total_samples": total_samples,
            "segments": [{"audio": f.audio_name, "txt": f.stem + ".txt", "srt": f.stem + ".srt", "start": f.start,
                          "end": f.end, "start_sample": f.start_sample, "end_sample": f.end_sample,
                          "samples": r["samples"], "pcm_sha256": r["pcm_sha256"]}
                         for f, r in zip(seg_files, seg_results)],
            "complete_transcript": COMPLETE_TRANSCRIPT,
        }

        # ---------------------------------------------------------------- validation
        prog.start("validate")
        problems = validate_outputs(staging, details, seg_files, chk)
        details["validation"] = {"ok": not problems, "problems": problems}
        if problems:
            for p in problems:
                log.error("Validation: %s", p)
            raise RuntimeError("The results failed validation: " + problems[0])
        log.info("Validation passed: %d audio files, %d samples total", len(seg_files), total_samples)
        prog.update(0.5, "Verifying original recordings are unchanged")
        for s, d in zip(sources, details["sources"]):
            chk()
            st = s.path.stat()
            if st.st_size != d["size"] or batch.sha256_of(s.path, chk) != d["sha256"]:
                raise RuntimeError(f"{s.name} changed while it was being processed. Nothing was archived; "
                                   "please process the folder again.")
        details["state"] = "archiving"
        batch.write_json_atomic(details_path, details)
        if log_file and log_file.is_file():
            shutil.copyfile(log_file, staging / BATCH_LOG_NAME)
        batch.commit_staging(staging, final_dir)
        committed = True
        staging = None
        log.info("Results committed to %s", final_dir)

        # ---------------------------------------------------------------- archive originals
        prog.start("archive")
        ar = batch.archive_sources(root, batch_name, final_dir / DETAILS_NAME)
        for name, err in ar.failed:
            warnings.append(err)
        log.info("Archive: %d moved, %d failed", len(ar.moved), len(ar.failed))
        prog.update(1.0)
        total_time = time.time() - t_start
        log.info("Stage timings (s): %s", prog.finish())
        log.info("Finished in %s", _fmt_dur(total_time))
        if ar.failed:
            return Outcome("archive_incomplete",
                           f"Processing finished, but {len(ar.failed)} original recording(s) could not be moved to "
                           f"'{PROCESSED_DIR}\\{batch_name}'. They stay in the folder, are not processed again, and "
                           "will be moved next time you click Process.", batch_name, str(final_dir), warnings)
        return Outcome("success", f"Done. {len(seg_files)} audio file(s) and transcripts were created in "
                                  f"'{REFINED_DIR}\\{batch_name}'.", batch_name, str(final_dir), warnings)
    except CancelledError:
        log.warning("Cancelled by the user")
        return Outcome("cancelled", "Processing was cancelled. Your recordings were not changed.", warnings=warnings)
    except batch.SafetyError as e:
        log.error("Safety stop: %s", e)
        return Outcome("failed", str(e), warnings=warnings)
    except RuntimeError as e:
        if str(e) == "MODEL_MISSING":
            return Outcome("failed", "MODEL_MISSING")
        log.error("Failed: %s\n%s", e, traceback.format_exc())
        return Outcome("failed", f"{e} Your recordings were not changed.", warnings=warnings)
    except OSError as e:
        log.error("Failed: %s\n%s", e, traceback.format_exc())
        if getattr(e, "errno", None) == 28:
            msg = "The disk is full. Free some space and try again. Your recordings were not changed."
        elif isinstance(e, PermissionError):
            msg = (f"A file could not be accessed ({e.filename or e}). It may be open in another program, or "
                   "access to the folder is denied. Your recordings were not changed.")
        else:
            msg = f"A file operation failed ({e}). Your recordings were not changed."
        return Outcome("failed", msg, warnings=warnings)
    except Exception as e:
        log.error("Unexpected error: %s\n%s", e, traceback.format_exc())
        return Outcome("failed", f"Unexpected error ({type(e).__name__}: {e}). Your recordings were not changed.",
                       warnings=warnings)
    finally:
        if staging is not None and not committed:
            batch.discard_staging(staging)
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        lock.release()
        if committed and log_file and final_dir and log_file.is_file():
            try:
                shutil.copyfile(log_file, final_dir / BATCH_LOG_NAME)
            except OSError:
                pass


def estimate_weights(prog: Progress, total_s: float, speech_s: float, gpu: bool, threads: int, workers: int) -> dict:
    """Stage weights in expected seconds (measured for finished stages) for a realistic time estimate."""
    done = prog.finish()
    asr_rtf = 0.12 if gpu else 8.0 / max(1, threads)       # measured: ~2x speech duration with 4 threads
    est = {
        "prepare": done.get("prepare", 1.0), "model": done.get("model", 5.0), "decode": done.get("decode", 1.0),
        "vad": max(done.get("vad", 0.0), 0.003 * total_s), "analyse": 0.004 * total_s,
        "enhance": total_s * 0.2 / max(1, workers) + 1.0, "transcribe": speech_s * asr_rtf * 1.1 + 2.0,
        "write_audio": 0.01 * total_s + 1.0, "write_text": 1.0, "validate": 0.006 * total_s + 1.0, "archive": 2.0,
    }
    return {k: max(0.01, v) for k, v in est.items()}


def validate_outputs(out_dir: Path, details: dict, seg_files: list, chk) -> list[str]:
    problems = []
    total = 0
    for i, (f, rec) in enumerate(zip(seg_files, details["outputs"]["segments"])):
        chk()
        expect = f.end_sample - f.start_sample
        if i < len(seg_files) - 1 and expect != SEGMENT_SECONDS * MASTER_SR:
            problems.append(f"{f.audio_name}: not exactly {SEGMENT_SECONDS // 60} minutes")
        err = segments.verify_segment(out_dir / f.audio_name, expect, rec["pcm_sha256"])
        if err:
            problems.append(err)
        total += expect
        for ext in (".txt", ".srt"):
            p = out_dir / (f.stem + ext)
            try:
                p.read_text(encoding="utf-8")
            except Exception as e:
                problems.append(f"{p.name}: unreadable ({e})")
    if total != details["outputs"]["total_samples"]:
        problems.append(f"segments contain {total} samples, expected {details['outputs']['total_samples']}")
    if total != sum(d["decoded_samples"] for d in details["sources"]):
        problems.append("segment samples do not equal decoded samples")
    try:
        txt = (out_dir / COMPLETE_TRANSCRIPT).read_text(encoding="utf-8")
        if not txt.strip():
            problems.append("complete transcript is empty")
    except Exception as e:
        problems.append(f"{COMPLETE_TRANSCRIPT}: unreadable ({e})")
    return problems
