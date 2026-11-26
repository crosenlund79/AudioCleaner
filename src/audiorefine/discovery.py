"""Finding the recordings to process (top level of the selected folder only)."""
from __future__ import annotations

import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .batch import PendingArchive, find_pending_archives, folder_busy
from .constants import AUDIO_EXTENSIONS, PROCESSED_DIR, REFINED_DIR
from .ordering import determine_order
from .probe import SourceInfo, probe


@dataclass
class ScanResult:
    root: Path
    sources: list[SourceInfo] = field(default_factory=list)       # readable, in processing order
    unreadable: list[SourceInfo] = field(default_factory=list)
    pending: list[PendingArchive] = field(default_factory=list)   # already processed, archiving unfinished
    order_basis: str = ""
    warnings: list[str] = field(default_factory=list)
    other_files: int = 0
    busy: bool = False

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.sources)

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "sources": [s.to_dict() for s in self.sources],
            "unreadable": [s.to_dict() for s in self.unreadable],
            "pending": [{"batch": p.batch, "names": p.names} for p in self.pending],
            "order_basis": self.order_basis,
            "warnings": self.warnings,
            "other_files": self.other_files,
            "busy": self.busy,
            "total_duration": self.total_duration,
        }


def _is_hidden_or_system(entry: os.DirEntry) -> bool:
    if entry.name.startswith((".", "~$")):
        return True
    if sys.platform == "win32":
        try:
            attrs = entry.stat(follow_symlinks=False).st_file_attributes
            return bool(attrs & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM))
        except OSError:
            return True
    return False


def candidate_files(root: Path) -> tuple[list[Path], int]:
    files, other = [], 0
    with os.scandir(root) as it:
        for e in it:
            try:
                if not e.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            if _is_hidden_or_system(e):
                continue
            if Path(e.name).suffix.lower() in AUDIO_EXTENSIONS:
                files.append(Path(e.path))
            else:
                other += 1
    return files, other


def scan_folder(root: Path, check_decode: bool = True) -> ScanResult:
    root = Path(root)
    res = ScanResult(root=root)
    if not root.is_dir():
        res.warnings.append("The folder does not exist.")
        return res
    res.busy = folder_busy(root)
    names = {root.name.lower(), root.parent.name.lower()}
    if names & {REFINED_DIR, PROCESSED_DIR}:
        res.warnings.append("This looks like one of AudioRefine's own output folders. Choose the folder that "
                            "contains your new recordings instead.")
    res.pending = find_pending_archives(root)
    pending_names = {n for p in res.pending for n in p.names}
    files, res.other_files = candidate_files(root)
    files = [f for f in files if f.name not in pending_names]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(files)))) as ex:
        infos = list(ex.map(lambda f: probe(f, check_decode=check_decode), files))
    ok = [i for i in infos if i.ok]
    res.unreadable = sorted([i for i in infos if not i.ok], key=lambda i: i.name.casefold())
    res.sources, res.order_basis, w = determine_order(ok)
    res.warnings += w
    if res.pending:
        n = sum(len(p.names) for p in res.pending)
        res.warnings.append(f"{n} recording(s) from an earlier batch were processed but not yet moved to "
                            f"'processed'. They are not included again and will be moved when you process.")
    return res
