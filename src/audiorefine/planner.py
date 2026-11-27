"""Planning which clock hours need (re)processing.

Every readable recording in the folder is placed on the clock by its recording start time
(``rectime``) and split into hour pieces. Against the completed hour manifests a recording is:

* ``new``        – nothing of it has been processed;
* ``partial``    – some of its hours are finished (an earlier run was interrupted);
* ``processed``  – all of it is in finished hours, but the original was not archived yet
                   (e.g. interrupted just before archiving) → it is archived, not reprocessed;
* ``duplicate``  – identical content was processed and archived before (a copy put back into
                   the folder) → left alone and reported, never transcribed twice.

Identity is the SHA-256 of the file content, so renamed copies are recognised and a different
recording with an old name is treated as new.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .constants import MASTER_SR
from .hourbatch import archived_shas
from .hours import Coverage, HourKey, hour_key_of, load_complete_manifests, split_into_hours
from .ordering import natural_key
from .rectime import RecTime, detect


@dataclass
class PlannedSource:
    info: object                     # probe.SourceInfo
    rt: RecTime
    sha: str | None
    status: str
    todo_hours: list[HourKey] = field(default_factory=list)
    all_hours: list[HourKey] = field(default_factory=list)
    archived_as: str = ""

    @property
    def start_key(self) -> HourKey:
        return hour_key_of(self.rt.start)

    def to_dict(self) -> dict:
        d = self.info.to_dict()
        d.update({"rec_time": self.rt.start.isoformat(), "rec_basis": self.rt.basis, "rec_estimated": self.rt.estimated,
                  "rec_note": self.rt.note, "status": self.status, "todo_hours": [str(h) for h in self.todo_hours],
                  "all_hours": [str(h) for h in self.all_hours], "archived_as": self.archived_as})
        return d


@dataclass
class Plan:
    sources: list[PlannedSource]
    hours: dict[HourKey, list[PlannedSource]]
    manifests: dict[HourKey, dict]
    coverage: Coverage

    @property
    def hour_keys(self) -> list[HourKey]:
        return sorted(self.hours)

    def summary(self) -> dict:
        keys = self.hour_keys
        return {
            "hours_to_process": len(keys), "days_to_process": len({k.date for k in keys}),
            "hours": [str(k) for k in keys],
            "completed_hours": len(self.manifests), "completed_days": len({k.date for k in self.manifests}),
            "pending_archive": [s.info.name for s in self.sources if s.status == "processed"],
            "duplicates": [s.info.name for s in self.sources if s.status == "duplicate"],
            "updates_existing_hours": [str(k) for k in keys if k in self.manifests],
        }


def make_plan(root: Path, infos: list, sha_of: Callable[[object], str | None],
              manifests: dict[HourKey, dict] | None = None) -> Plan:
    manifests = load_complete_manifests(root) if manifests is None else manifests
    cov = Coverage.build(manifests)
    archived = archived_shas(root)
    sources: list[PlannedSource] = []
    for info in infos:
        rt = detect(info)
        sha = sha_of(info)
        est = max(1, int(round(float(info.duration or 0) * MASTER_SR)))
        if sha and cov.fully_covered(sha):
            arch = archived.get(sha)
            ps = PlannedSource(info, rt, sha, "duplicate" if arch else "processed",
                               all_hours=cov.hours_of(sha), archived_as=arch["archived_as"] if arch else "")
        elif sha and sha in cov.ranges:
            total = cov.totals[sha]
            pieces = split_into_hours(rt.start, total)
            done = set(cov.hours_of(sha))
            ps = PlannedSource(info, rt, sha, "partial", [p.key for p in pieces if p.key not in done],
                               [p.key for p in pieces])
        else:
            keys = [p.key for p in split_into_hours(rt.start, est)]
            ps = PlannedSource(info, rt, sha, "new", list(keys), list(keys))
        sources.append(ps)
    sources.sort(key=lambda s: (s.rt.start.timestamp(), natural_key(s.info.name)))
    hours: dict[HourKey, list[PlannedSource]] = defaultdict(list)
    for s in sources:
        for k in s.todo_hours:
            hours[k].append(s)
    return Plan(sources, dict(hours), manifests, cov)
