"""File safety for the hourly layout: staging, commit/replace of hour folders, recovery, archiving.

* An hour is built in ``refined/<date>/.incomplete-HH-00`` (hidden, with a marker file) and
  appears as ``refined/<date>/HH-00`` with one rename.
* Replacing an existing hour (new recordings for an hour that was already finished) uses a small
  journal ``refined/<date>/.replace-HH-00.json``: old folder → ``.replaced-HH-00``, new folder →
  ``HH-00``, old folder deleted. After a crash at any point, ``recover`` completes or rolls back
  the replacement, so there is always exactly one valid hour folder.
* Originals are moved to ``processed/<date>/<HH-00>`` of the hour in which the recording
  *started*, only when every part of the recording is in finished hours; SHA-256 is verified
  before and after; an existing file is never overwritten (a numbered name is used instead).
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from . import batch
from .constants import REFINED_DIR, STAGING_MARKER
from .hours import ARCHIVE_JOURNAL, HOUR_MANIFEST, HourKey, read_json, write_json_atomic

log = logging.getLogger("audiorefine.hourbatch")

STAGING = ".incomplete-"
REPLACED = ".replaced-"
REPLACE_JOURNAL = ".replace-{}.json"


def staging_dir(root: Path, key: HourKey) -> Path:
    return root / REFINED_DIR / key.date / (STAGING + key.folder_name)


def create_hour_staging(root: Path, key: HourKey) -> Path:
    d = staging_dir(root, key)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / STAGING_MARKER).write_text("AudioRefine incomplete hour - safe to delete\n", encoding="utf-8")
    batch.set_hidden(d)
    return d


def commit_hour(root: Path, key: HourKey, staging: Path) -> None:
    final = key.dir(root)
    (staging / STAGING_MARKER).unlink(missing_ok=True)
    if not final.exists():
        batch._retry(lambda: os.rename(staging, final))
        _unhide(final)
        return
    date_dir = final.parent
    old = date_dir / (REPLACED + key.folder_name)
    journal = date_dir / REPLACE_JOURNAL.format(key.folder_name)
    if old.exists():
        shutil.rmtree(old)
    write_json_atomic(journal, {"staging": staging.name, "old": old.name, "final": final.name,
                                "time": datetime.now().isoformat(timespec="seconds")})
    batch.set_hidden(journal)
    batch._retry(lambda: os.rename(final, old))
    batch._retry(lambda: os.rename(staging, final))
    _unhide(final)
    shutil.rmtree(old, ignore_errors=True)
    journal.unlink(missing_ok=True)


def _unhide(p: Path) -> None:
    import sys

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetFileAttributesW(str(p), 0x80)
        except Exception:
            pass


def recover(root: Path) -> list[str]:
    """Finish or roll back interrupted hour replacements and remove incomplete hour folders."""
    notes = []
    refined = root / REFINED_DIR
    if not refined.is_dir():
        return notes
    for date_dir in sorted(p for p in refined.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for j in sorted(date_dir.glob(".replace-*.json")):
            info = read_json(j) or {}
            final = date_dir / info.get("final", "")
            old = date_dir / info.get("old", "")
            stg = date_dir / info.get("staging", "")
            if final.name and not final.exists():
                m = read_json(stg / HOUR_MANIFEST) if stg.name and stg.exists() else None
                if m and m.get("state") == "complete":
                    os.rename(stg, final)
                    notes.append(f"completed the interrupted update of {date_dir.name}\\{final.name}")
                elif old.name and old.exists():
                    os.rename(old, final)
                    notes.append(f"restored {date_dir.name}\\{final.name} after an interrupted update")
            if old.name and old.exists() and final.exists():
                shutil.rmtree(old, ignore_errors=True)
            j.unlink(missing_ok=True)
        for d in date_dir.iterdir():
            if d.is_dir() and d.name.startswith((STAGING, REPLACED)):
                if d.name.startswith(REPLACED) or (d / STAGING_MARKER).is_file():
                    shutil.rmtree(d, ignore_errors=True)
                    notes.append(f"removed unfinished work {date_dir.name}\\{d.name}")
        # an empty date folder left by an interrupted first hour
        try:
            if not any(date_dir.iterdir()):
                date_dir.rmdir()
        except OSError:
            pass
    return notes


# ------------------------------------------------------------------------------ archiving
def archive_journal(root: Path) -> list[dict]:
    j = read_json(root / REFINED_DIR / ARCHIVE_JOURNAL)
    return j.get("archived", []) if isinstance(j, dict) else []


def archived_shas(root: Path) -> dict[str, dict]:
    return {r["sha256"]: r for r in archive_journal(root)}


def _record_archive(root: Path, rec: dict) -> None:
    path = root / REFINED_DIR / ARCHIVE_JOURNAL
    path.parent.mkdir(parents=True, exist_ok=True)
    j = read_json(path) or {"app": "AudioRefine", "archived": []}
    j["archived"] = [r for r in j.get("archived", []) if r.get("sha256") != rec["sha256"]] + [rec]
    write_json_atomic(path, j)


def free_name(dest_dir: Path, name: str) -> Path:
    p = dest_dir / name
    if not p.exists():
        return p
    stem, ext = os.path.splitext(name)
    k = 2
    while (dest_dir / f"{stem} ({k}){ext}").exists():
        k += 1
    return dest_dir / f"{stem} ({k}){ext}"


def archive_original(root: Path, src: Path, sha: str, size: int, start_key: HourKey) -> tuple[bool, str]:
    """Move a fully processed original to processed/<date>/<HH-00>. Returns (ok, message)."""
    try:
        if not src.exists():
            return False, f"{src.name} is no longer in the folder."
        if src.stat().st_size != size or batch.sha256_of(src) != sha:
            return False, f"{src.name} changed after it was processed; it was left in place."
        dest_dir = start_key.processed_dir(root)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dst = free_name(dest_dir, src.name)
        batch._move_verified(src, dst, size, sha)
        if batch.sha256_of(dst) != sha:
            return False, f"{src.name}: checksum after archiving does not match."
        rel = dst.relative_to(root)
        try:
            _record_archive(root, {"sha256": sha, "name": src.name, "archived_as": str(rel).replace("\\", "/"),
                                   "size": size, "time": datetime.now().isoformat(timespec="seconds")})
        except OSError as e:  # the move itself succeeded and was verified
            log.warning("Could not record the archiving of %s: %s", src.name, e)
        return True, str(rel)
    except Exception as e:
        return False, batch._friendly_os_error(src.name, e) if not isinstance(e, batch.SafetyError) else str(e)
