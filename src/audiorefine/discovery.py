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
    plan: object | None = None                                    # planner.Plan (hourly layout)

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.sources)

    def to_dict(self) -> dict:
        planned = {ps.info.name: ps for ps in (self.plan.sources if self.plan else [])}
        srcs = []
        for s in self.sources:
            d = planned[s.name].to_dict() if s.name in planned else s.to_dict()
            srcs.append(d)
        return {
            "root": str(self.root),
            "sources": srcs,
            "plan": self.plan.summary() if self.plan else None,
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


def _quick_sha(root: Path):
    """For a fast scan, hash only files that may already have been processed (same name and size)."""
    from .batch import sha256_of
    from .hourbatch import archive_journal
    from .hours import load_complete_manifests

    known = {(p["source"], int(p["size"])) for m in load_complete_manifests(root).values() for p in m["pieces"]}
    known |= {(r["name"], int(r.get("size", -1))) for r in archive_journal(root)}

    def f(info):
        return sha256_of(info.path) if (info.name, info.size) in known else None

    return f


def scan_folder(root: Path, check_decode: bool = True, sha_of=None) -> ScanResult:
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
    from .planner import make_plan

    plan = make_plan(root, ok, sha_of or _quick_sha(root))
    res.plan = plan
    res.sources = [ps.info for ps in plan.sources]
    res.order_basis = "recording date and time"
    est = [ps.info.name for ps in plan.sources if ps.rt.estimated and ps.status in ("new", "partial")]
    if est:
        res.warnings.append(f"{len(est)} recording(s) have no recording time in the file name or file; their time "
                            "is estimated from the file dates (marked ≈).")
    dups = [ps for ps in plan.sources if ps.status == "duplicate"]
    if dups:
        res.warnings.append(f"{len(dups)} file(s) were already processed earlier (identical content is archived in "
                            "'processed'); they are left in the folder and not processed again: "
                            + ", ".join(d.info.name for d in dups[:3]) + ("…" if len(dups) > 3 else ""))
    if res.pending:
        n = sum(len(p.names) for p in res.pending)
        res.warnings.append(f"{n} recording(s) from an earlier batch were processed but not yet moved to "
                            f"'processed'. They are not included again and will be moved when you process.")
    return res
