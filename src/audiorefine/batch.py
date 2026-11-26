"""File-safety core: batch naming, staging, commit, archiving of originals and recovery.

Guarantees (see README.md "How your files are handled"):
* Original recordings are only ever opened for reading, and are moved (never copied-and-deleted
  unless the move crosses volumes, in which case the copy is verified first) into
  ``processed/<batch>`` only after all outputs were written and validated.
* All outputs are first written to a hidden staging folder ``refined/.incomplete-<batch>`` and
  appear under their final name ``refined/<batch>`` with a single atomic rename.
* Nothing existing is overwritten: batch folders get a unique name, and moves fail instead of
  replacing an existing file.
* Every step is recorded in ``_AudioRefine_details.json`` inside the batch folder, so an
  interrupted archiving step is detected and completed on the next run, and those originals are
  never mistaken for new recordings.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .constants import DETAILS_NAME, LOCK_NAME, PROCESSED_DIR, REFINED_DIR, STAGING_MARKER, STAGING_PREFIX

log = logging.getLogger("audiorefine.batch")


class SafetyError(Exception):
    pass


# ------------------------------------------------------------------------------ helpers
def sha256_of(path: Path, cancel: Callable[[], None] | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(4 << 20)
            if not b:
                break
            h.update(b)
            if cancel:
                cancel()
    return h.hexdigest()


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def set_hidden(path: Path) -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x02)
        except Exception:
            pass


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ------------------------------------------------------------------------------ batch naming
def new_batch_name(root: Path, now: datetime | None = None) -> str:
    now = now or datetime.now()
    base = now.strftime("%Y-%m-%d_%H-%M-%S")
    name, k = base, 2
    while ((root / REFINED_DIR / name).exists() or (root / PROCESSED_DIR / name).exists()
           or (root / REFINED_DIR / (STAGING_PREFIX + name)).exists()):
        name = f"{base}_{k}"
        k += 1
    return name


# ------------------------------------------------------------------------------ locking
class FolderLock:
    """Prevents two AudioRefine processes from working on the same folder at the same time."""

    def __init__(self, root: Path):
        self.path = root / REFINED_DIR / LOCK_NAME
        self.held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    json.dump({"pid": os.getpid(), "host": socket.gethostname(), "time": time.time()}, f)
                set_hidden(self.path)
                self.held = True
                return
            except FileExistsError:
                info = read_json(self.path) or {}
                same_host = info.get("host") == socket.gethostname()
                if same_host and not pid_alive(int(info.get("pid", 0))):
                    log.warning("Removing stale lock left by process %s", info.get("pid"))
                    self.path.unlink(missing_ok=True)
                    continue
                raise SafetyError("This folder is already being processed by AudioRefine "
                                  "(possibly on another computer). Wait for it to finish and try again.")
        raise SafetyError("Could not lock the folder for processing.")

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def folder_busy(root: Path) -> bool:
    p = root / REFINED_DIR / LOCK_NAME
    if not p.exists():
        return False
    info = read_json(p) or {}
    if info.get("host") == socket.gethostname() and not pid_alive(int(info.get("pid", 0))):
        return False
    return True


# ------------------------------------------------------------------------------ staging
def create_staging(root: Path, batch: str) -> Path:
    refined = root / REFINED_DIR
    refined.mkdir(parents=True, exist_ok=True)
    d = refined / (STAGING_PREFIX + batch)
    d.mkdir(parents=False, exist_ok=False)
    (d / STAGING_MARKER).write_text("AudioRefine incomplete output - safe to delete\n", encoding="utf-8")
    set_hidden(d)
    return d


def cleanup_stale_staging(root: Path, have_lock: bool = False) -> list[str]:
    """Remove staging folders of runs that crashed or were cancelled (only folders with our marker).

    Call with ``have_lock=True`` while holding the folder lock: then no other run can own them.
    """
    removed = []
    refined = root / REFINED_DIR
    if not refined.is_dir() or (not have_lock and folder_busy(root)):
        return removed
    for d in refined.iterdir():
        if d.is_dir() and d.name.startswith(STAGING_PREFIX) and (d / STAGING_MARKER).is_file():
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                removed.append(d.name)
                log.info("Removed incomplete output of an interrupted run: %s", d.name)
    return removed


def discard_staging(staging: Path) -> None:
    if staging.is_dir() and (staging / STAGING_MARKER).is_file():
        shutil.rmtree(staging, ignore_errors=True)


def _retry(fn, attempts: int = 8, delay: float = 0.75):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except PermissionError as e:  # Windows: file briefly locked (antivirus, indexer, player)
            last = e
            time.sleep(delay * (1 + i))
    raise last


def commit_staging(staging: Path, final: Path) -> None:
    if final.exists():
        raise SafetyError(f"The output folder {final.name} already exists; nothing was overwritten.")
    (staging / STAGING_MARKER).unlink(missing_ok=True)
    try:
        _retry(lambda: os.rename(staging, final))
    except Exception:
        (staging / STAGING_MARKER).write_text("AudioRefine incomplete output - safe to delete\n", encoding="utf-8")
        raise
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetFileAttributesW(str(final), 0x80)  # FILE_ATTRIBUTE_NORMAL
        except Exception:
            pass


# ------------------------------------------------------------------------------ archiving
@dataclass
class ArchiveResult:
    moved: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.failed


def _move_verified(src: Path, dst: Path, size: int, sha: str) -> None:
    if dst.exists():
        raise SafetyError(f"{dst.name} already exists in {dst.parent.name}; it was not overwritten.")
    st = src.stat()
    if st.st_size != size:
        raise SafetyError(f"{src.name} changed while it was being processed (size differs); it was not moved.")
    try:
        _retry(lambda: os.rename(src, dst))
    except OSError as e:
        if getattr(e, "errno", None) != errno.EXDEV and getattr(e, "winerror", None) != 17:
            raise
        # different volume: copy, verify, then remove the original
        tmp = dst.with_name(dst.name + ".copying")
        shutil.copy2(src, tmp)
        if sha256_of(tmp) != sha:
            tmp.unlink(missing_ok=True)
            raise SafetyError(f"Verification of the copy of {src.name} failed; the original was kept.")
        os.rename(tmp, dst)
        _retry(lambda: os.remove(src))
    if dst.stat().st_size != size:
        raise SafetyError(f"{dst.name}: size changed during archiving.")


def archive_sources(root: Path, batch: str, details_path: Path, verify_hash: bool = True) -> ArchiveResult:
    """Move the batch's originals into processed/<batch>, recording every step in the details file."""
    details = read_json(details_path)
    if details is None:
        raise SafetyError("The batch record is missing or unreadable; originals were not moved.")
    dest_dir = root / PROCESSED_DIR / batch
    dest_dir.mkdir(parents=True, exist_ok=True)
    arch = details.setdefault("archive", {})
    arch["dir"] = f"{PROCESSED_DIR}/{batch}"
    files = arch.setdefault("files", {})
    result = ArchiveResult()
    for s in details["sources"]:
        name, size, sha = s["name"], int(s["size"]), s["sha256"]
        state = files.get(name, {}).get("state")
        src, dst = root / name, dest_dir / name
        try:
            if state == "moved" and dst.exists():
                result.moved.append(name)
                continue
            if not src.exists():
                if dst.exists() and dst.stat().st_size == size and (not verify_hash or sha256_of(dst) == sha):
                    files[name] = {"state": "moved", "time": datetime.now().isoformat(timespec="seconds")}
                    result.moved.append(name)
                    continue
                raise SafetyError(f"{name} is no longer in the folder and was not archived.")
            if verify_hash and sha256_of(src) != sha:
                raise SafetyError(f"{name} changed after it was processed; it was left in place.")
            _move_verified(src, dst, size, sha)
            if verify_hash and sha256_of(dst) != sha:
                raise SafetyError(f"{name}: checksum after archiving does not match.")
            files[name] = {"state": "moved", "time": datetime.now().isoformat(timespec="seconds")}
            result.moved.append(name)
        except Exception as e:
            msg = str(e) if isinstance(e, SafetyError) else _friendly_os_error(name, e)
            files[name] = {"state": "failed", "error": msg}
            result.failed.append((name, msg))
            log.error("Archiving %s failed: %s", name, msg)
        finally:
            arch["state"] = "in_progress"
            write_json_atomic(details_path, details)
    arch["state"] = "complete" if result.complete else "incomplete"
    details["state"] = "complete" if result.complete else "archive_incomplete"
    write_json_atomic(details_path, details)
    return result


def _friendly_os_error(name: str, e: Exception) -> str:
    if isinstance(e, PermissionError):
        return f"{name} could not be moved because it is in use or access was denied."
    if isinstance(e, OSError) and e.errno == errno.ENOSPC:
        return f"{name} could not be moved: the disk is full."
    return f"{name} could not be moved ({type(e).__name__}: {e})."


@dataclass
class PendingArchive:
    batch: str
    details_path: Path
    names: list[str]


def find_pending_archives(root: Path) -> list[PendingArchive]:
    """Batches whose outputs are complete but whose originals were not all moved yet."""
    out = []
    refined = root / REFINED_DIR
    if not refined.is_dir():
        return out
    for d in sorted(refined.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        det = d / DETAILS_NAME
        info = read_json(det) if det.is_file() else None
        if not info or info.get("app") != "AudioRefine":
            continue
        if info.get("state") == "complete":
            continue
        files = info.get("archive", {}).get("files", {})
        pending = [s["name"] for s in info.get("sources", []) if files.get(s["name"], {}).get("state") != "moved"
                   and (root / s["name"]).exists()]
        if pending:
            out.append(PendingArchive(d.name, det, pending))
    return out
