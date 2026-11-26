"""AudioRefine engine (command line). The desktop window runs this as a child process.

Commands
  scan <folder>                 list recordings, order and durations (JSON)
  process <folder>              process the folder
  status                        installed components, hardware (JSON)
  download                      download the speech model (and GPU runtime when useful)
  selftest                      check that all bundled components load
  version

With ``--json`` every event is one JSON object per line on stdout; a line ``cancel`` on stdin
requests cancellation. Exit codes: 0 success, 1 failure, 2 cancelled, 3 success with archiving
incomplete, 4 speech model missing.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from . import __version__

log = logging.getLogger("audiorefine.cli")


def _out(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _human(ev: dict) -> None:
    if ev.get("type") == "progress":
        eta = f" ~{int(ev['eta'] // 60)}m{int(ev['eta'] % 60):02d}s left" if ev.get("eta") else ""
        sys.stdout.write(f"\r[{ev['fraction'] * 100:5.1f}%] {ev['label']}: {ev.get('detail', '')[:50]:<50}{eta}   ")
        sys.stdout.flush()
    elif ev.get("type") in ("info", "warning"):
        sys.stdout.write("\n" + json.dumps(ev, ensure_ascii=False) + "\n")


def _stdin_cancel_watcher(token) -> None:
    def run():
        for line in sys.stdin:
            if line.strip().lower() == "cancel":
                token.cancel()
                return

    threading.Thread(target=run, daemon=True).start()


def cmd_scan(args) -> int:
    from .discovery import scan_folder

    res = scan_folder(Path(args.folder))
    _out({"type": "scan", **res.to_dict()})
    return 0


def cmd_status(args) -> int:
    from . import hardware
    from .constants import ACTIVE_MODEL
    from .downloads import gpu_runtime_bytes, gpu_runtime_installed, missing_model_bytes, model_installed

    hw = hardware.detect()
    gpu = hardware.gpu_candidate(hw)
    env_model = os.environ.get("AUDIOREFINE_MODEL_DIR")
    installed = bool(env_model) or model_installed(ACTIVE_MODEL)
    _out({"type": "status", "version": __version__, "model": ACTIVE_MODEL.display,
          "model_installed": installed, "model_missing_bytes": 0 if installed else missing_model_bytes(ACTIVE_MODEL),
          "gpu": gpu.__dict__ if gpu else None, "gpu_runtime_installed": gpu_runtime_installed(),
          "gpu_runtime_bytes": gpu_runtime_bytes(), "hardware": hw.to_dict()})
    return 0


def cmd_download(args, emit) -> int:
    from . import hardware
    from .constants import ACTIVE_MODEL
    from .downloads import Cancelled, DownloadError, gpu_runtime_installed, install_gpu_runtime, install_model, model_installed
    from .progress import CancelToken

    token = CancelToken()
    if args.json:
        _stdin_cancel_watcher(token)

    def chk():
        if token.cancelled:
            raise Cancelled()

    t0 = time.time()
    try:
        if not model_installed(ACTIVE_MODEL):
            log.info("Downloading %s from %s@%s", ACTIVE_MODEL.display, ACTIVE_MODEL.repo, ACTIVE_MODEL.revision)

            def mp(done, total):
                emit({"type": "download", "component": "model", "label": "Speech recognition model",
                      "done": done, "total": total, "elapsed": round(time.time() - t0, 1)})

            install_model(ACTIVE_MODEL, mp, chk)
            log.info("Model installed and verified")
        hw = hardware.detect()
        if args.gpu and hardware.gpu_candidate(hw) and not gpu_runtime_installed():
            log.info("Downloading NVIDIA GPU runtime for %s", hardware.gpu_candidate(hw))

            def gp(done, total):
                emit({"type": "download", "component": "gpu", "label": "GPU acceleration components",
                      "done": done, "total": total, "elapsed": round(time.time() - t0, 1)})

            try:
                install_gpu_runtime(gp, chk)
            except DownloadError as e:  # GPU is optional
                log.warning("GPU runtime download failed: %s", e)
                emit({"type": "warning", "message": f"GPU components could not be downloaded ({e}); the CPU will be used."})
        emit({"type": "result", "status": "success", "message": "Download complete."})
        return 0
    except Cancelled:
        emit({"type": "result", "status": "cancelled", "message": "Download cancelled. It will resume where it stopped."})
        return 2
    except Exception as e:
        log.exception("Download failed")
        emit({"type": "result", "status": "failed", "message": f"Download failed: {e}. It will resume where it stopped when you try again."})
        return 1


def cmd_process(args, emit, log_file: Path) -> int:
    from .pipeline import Options, process_folder
    from .progress import CancelToken

    token = CancelToken()
    if args.json:
        _stdin_cancel_watcher(token)
    opts = Options(allow_gpu=not args.cpu, model_dir=Path(args.model_dir) if args.model_dir else None)
    out = process_folder(Path(args.folder), emit, token, opts, log_file=log_file)
    if out.message == "MODEL_MISSING":
        emit({"type": "result", "status": "model_missing", "message": "The speech recognition model is not installed."})
        return 4
    emit({"type": "result", "status": out.status, "message": out.message, "batch": out.batch,
          "refined_dir": out.refined_dir, "warnings": out.warnings, "log_file": str(log_file)})
    return {"success": 0, "cancelled": 2, "archive_incomplete": 3}.get(out.status, 1)


def cmd_selftest(args) -> int:
    results = {}
    try:
        from .enhance import arfilter_version

        results["arfilter"] = arfilter_version()
    except Exception as e:
        results["arfilter"] = f"ERROR {e}"
    for mod in ("numpy", "soundfile", "av", "ctranslate2", "faster_whisper", "onnxruntime"):
        try:
            m = __import__(mod)
            results[mod] = getattr(m, "__version__", "ok")
        except Exception as e:
            results[mod] = f"ERROR {e}"
    try:
        from .vad import StreamingSileroVAD
        import numpy as np

        v = StreamingSileroVAD()
        v.windows(np.zeros(512 * 4, dtype=np.float32))
        results["silero_vad"] = "ok"
    except Exception as e:
        results["silero_vad"] = f"ERROR {e}"
    ok = not any(str(v).startswith(("ERROR", "unavailable")) for v in results.values())
    _out({"type": "selftest", "ok": ok, "version": __version__, "components": results})
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="AudioRefineEngine", description="AudioRefine processing engine")
    ap.add_argument("--json", action="store_true", help="machine-readable events on stdout")
    ap.add_argument("--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan"); p.add_argument("folder")
    p = sub.add_parser("process"); p.add_argument("folder"); p.add_argument("--cpu", action="store_true")
    p.add_argument("--model-dir", default=None)
    sub.add_parser("status")
    p = sub.add_parser("download"); p.add_argument("--gpu", action="store_true")
    sub.add_parser("selftest")
    sub.add_parser("version")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if args.cmd == "version":
        print(__version__)
        return 0
    from . import log as logsetup

    log_file = logsetup.setup(args.cmd, verbose=args.verbose)
    log.info("AudioRefine engine %s: %s", __version__, " ".join(sys.argv[1:]))
    emit = _out if args.json else _human
    try:
        if args.cmd == "scan":
            return cmd_scan(args)
        if args.cmd == "status":
            return cmd_status(args)
        if args.cmd == "download":
            return cmd_download(args, emit)
        if args.cmd == "process":
            rc = cmd_process(args, emit, log_file)
            if not args.json:
                print()
            return rc
        if args.cmd == "selftest":
            return cmd_selftest(args)
    except Exception as e:  # last-resort guard: always report something
        log.exception("Engine failure")
        emit({"type": "result", "status": "failed", "message": f"Unexpected engine error: {e}", "log_file": str(log_file)})
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
