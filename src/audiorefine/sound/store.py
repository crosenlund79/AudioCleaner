"""Where sound analysis results live, and how they are kept consistent.

    <folder>/analysis/
        Sound_Events_Master.txt / .json       all analysed hours, chronological
        Transcript_Master_Annotated.txt        transcript + sound annotations (when transcripts exist)
        2026-11-25/
            Sound_Events_Day.txt
            08-00/  Sound_Events.json  Sound_Events.txt
        _AudioRefine_sources.json              source index: checksum, location, recording time
        _AudioRefine_reviews.json              the user's review marks (confirmed / incorrect)
        _sources/<sha256>.json                 analysis of one recording (all its events)

* Results are separate from the transcription output (``refined``) and from the originals; the
  analysis never writes anywhere else and opens recordings read-only.
* A recording is identified by the SHA-256 of its content, so moving it into ``processed`` (or
  renaming it) does not invalidate its results; playback finds it again by checksum.
* Every record carries the model, vocabulary and analysis version (``fingerprint``). Results of
  another version are never mixed in: they are recomputed, or left out of the combined reports.
* All files are written to a temporary name and then renamed (atomic); an interrupted run leaves
  either the old or the new version, never a half-written file.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

from ..batch import pid_alive
from ..hours import HourKey, read_json, write_json_atomic
from .model import MODEL_ID, MODEL_SHA256

ANALYSIS_DIR = "analysis"
ANALYSIS_VERSION = 1
HOUR_JSON = "Sound_Events.json"
HOUR_TXT = "Sound_Events.txt"
DAY_TXT = "Sound_Events_Day.txt"
MASTER_TXT = "Sound_Events_Master.txt"
MASTER_JSON = "Sound_Events_Master.json"
ANNOTATED_TXT = "Transcript_Master_Annotated.txt"
SOURCES_INDEX = "_AudioRefine_sources.json"
REVIEWS = "_AudioRefine_reviews.json"
SOURCE_CACHE_DIR = "_sources"
LOCK = ".audiorefine-analysis.lock"


def adir(root: Path) -> Path:
    return root / ANALYSIS_DIR


def fingerprint(vocab) -> dict:
    return {"model": MODEL_ID, "model_sha256": MODEL_SHA256[:16], "vocabulary": vocab.digest,
            "vocabulary_version": vocab.version, "analysis_version": ANALYSIS_VERSION}


def same_fingerprint(a: dict | None, b: dict) -> bool:
    return bool(a) and all(a.get(k) == b.get(k) for k in ("model", "model_sha256", "vocabulary", "analysis_version"))


def hour_dir(root: Path, key: HourKey) -> Path:
    return adir(root) / key.date / key.folder_name


def hour_records(root: Path) -> dict[HourKey, dict]:
    out = {}
    base = adir(root)
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if not d.is_dir() or len(d.name) != 10 or d.name[4] != "-":
            continue
        for h in sorted(d.iterdir()):
            if h.is_dir() and len(h.name) == 5 and h.name.endswith("-00"):
                r = read_json(h / HOUR_JSON)
                if r and r.get("app") == "AudioRefine" and r.get("state") == "complete":
                    out[HourKey(r["date"], int(r["hour"]))] = r
    return out


# ------------------------------------------------------------------------------ per-recording cache
def source_cache_path(root: Path, sha: str) -> Path:
    return adir(root) / SOURCE_CACHE_DIR / f"{sha}.json"


def load_source_result(root: Path, sha: str, fp: dict) -> dict | None:
    r = read_json(source_cache_path(root, sha))
    if r and same_fingerprint(r.get("fingerprint"), fp) and r.get("sha256") == sha:
        return r
    return None


def save_source_result(root: Path, sha: str, data: dict) -> None:
    p = source_cache_path(root, sha)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(p, data)


# ------------------------------------------------------------------------------ source index
def load_index(root: Path) -> dict:
    d = read_json(adir(root) / SOURCES_INDEX) or {}
    d.setdefault("files", {})      # relative path -> {size, mtime_ns, sha256}
    d.setdefault("sources", {})    # sha256 -> {name, path, size, rec_start, estimated, basis, duration}
    return d


def save_index(root: Path, index: dict) -> None:
    adir(root).mkdir(parents=True, exist_ok=True)
    write_json_atomic(adir(root) / SOURCES_INDEX, index)


def resolve_source(root: Path, sha: str, index: dict | None = None) -> Path | None:
    """Current location of a recording (it may have been moved into ``processed`` meanwhile)."""
    from ..batch import sha256_of
    from ..hourbatch import archive_journal

    index = index if index is not None else load_index(root)
    info = index["sources"].get(sha) or {}
    size = info.get("size")
    cands: list[Path] = []
    if info.get("path"):
        cands.append(root / info["path"])
    for r in archive_journal(root):
        if r.get("sha256") == sha and r.get("archived_as"):
            cands.append(root / r["archived_as"])
    for rel, f in index["files"].items():
        if f.get("sha256") == sha:
            cands.append(root / rel)
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        try:
            if c.is_file() and (size is None or c.stat().st_size == size):
                return c
        except OSError:
            continue
    # last resort: same name and size anywhere in the folder or processed/, verified by checksum
    name = info.get("name")
    if name:
        for base in (root, root / "processed"):
            if not base.is_dir():
                continue
            for p in ([base / name] if base == root else base.rglob(name)):
                try:
                    if p.is_file() and (size is None or p.stat().st_size == size) and sha256_of(p) == sha:
                        return p
                except OSError:
                    continue
    return None


# ------------------------------------------------------------------------------ reviews
def load_reviews(root: Path) -> dict:
    d = read_json(adir(root) / REVIEWS) or {}
    d.setdefault("version", 1)
    d.setdefault("reviews", {})
    return d


def set_review(root: Path, event: dict, state: str) -> None:
    """state: 'unreviewed' | 'confirmed' | 'incorrect'. The machine label is never changed."""
    d = load_reviews(root)
    if state == "unreviewed":
        d["reviews"].pop(event["id"], None)
    else:
        d["reviews"][event["id"]] = {"state": state, "category": event.get("category"),
                                     "source_sha256": event.get("source_sha256"), "src_start": event.get("src_start"),
                                     "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    adir(root).mkdir(parents=True, exist_ok=True)
    write_json_atomic(adir(root) / REVIEWS, d)


def review_state(reviews: dict, ev: dict) -> str:
    r = reviews.get("reviews", {}).get(ev["id"])
    if r:
        return r["state"]
    # an older analysis may have given the same occurrence another id: match by source, sound and time
    for r in reviews.get("reviews", {}).values():
        if (r.get("source_sha256") == ev.get("source_sha256") and r.get("category") == ev.get("category")
                and r.get("src_start") is not None and ev.get("src_start") is not None
                and ev["src_start"] - 2.0 <= float(r["src_start"]) <= ev.get("src_end", ev["src_start"]) + 2.0):
            return r["state"]
    return "unreviewed"


# ------------------------------------------------------------------------------ lock
class AnalysisLock:
    """One sound analysis per folder at a time; never while the folder is being transcribed."""

    def __init__(self, root: Path):
        self.path = adir(root) / LOCK
        self.held = False

    def acquire(self) -> None:
        from ..batch import SafetyError

        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    json.dump({"pid": os.getpid(), "host": socket.gethostname(), "time": time.time()}, f)
                self.held = True
                return
            except FileExistsError:
                info = read_json(self.path) or {}
                if info.get("host") == socket.gethostname() and not pid_alive(int(info.get("pid", 0))):
                    self.path.unlink(missing_ok=True)
                    continue
                raise SafetyError("Sounds in this folder are already being analysed by AudioRefine. "
                                  "Wait for it to finish and try again.")
        raise SafetyError("Could not lock the folder for sound analysis.")

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False


def analysis_busy(root: Path) -> bool:
    p = adir(root) / LOCK
    if not p.exists():
        return False
    info = read_json(p) or {}
    if info.get("host") == socket.gethostname() and not pid_alive(int(info.get("pid", 0))):
        return False
    return True


def cleanup_temp(root: Path) -> list[str]:
    """Remove half-written temporary files of an interrupted run."""
    removed = []
    base = adir(root)
    if not base.is_dir():
        return removed
    for p in base.rglob("*.tmp"):
        try:
            p.unlink()
            removed.append(str(p.relative_to(root)))
        except OSError:
            pass
    return removed
