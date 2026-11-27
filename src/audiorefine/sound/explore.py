"""Data side of Sound Explorer: load results, filter, count, and find what to play.

Kept free of Qt so it can be tested on its own; ``gui/sound_explorer.py`` only displays it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..hours import read_json
from ..rectime import local
from . import store

PRE_ROLL_S = 3.0          # playback starts a little before the event


@dataclass
class Results:
    root: Path
    events: list[dict] = field(default_factory=list)
    activities: list[dict] = field(default_factory=list)
    status: str = ""
    fingerprint: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)
    by_id: dict[str, dict] = field(default_factory=dict)

    @property
    def items(self) -> list[dict]:
        return self.events + self.activities

    @property
    def dates(self) -> list[str]:
        return sorted({item_date(x) for x in self.items})


def item_date(x: dict) -> str:
    return local(datetime.fromisoformat(x["start"])).strftime("%Y-%m-%d")


def item_hours(x: dict) -> tuple[float, float]:
    """Start and end as hours of the start date (end may exceed 24 for events past midnight)."""
    st, en = local(datetime.fromisoformat(x["start"])), local(datetime.fromisoformat(x["end"]))
    a = st.hour + st.minute / 60 + st.second / 3600
    b = a + max(0.0, (en - st).total_seconds()) / 3600
    return a, b


def load(root: Path) -> Results | None:
    d = read_json(store.adir(root) / store.MASTER_JSON)
    if not d or d.get("app") != "AudioRefine":
        return None
    r = Results(root, list(d.get("events", [])), list(d.get("activities", [])), d.get("status", ""),
                d.get("fingerprint", {}), store.load_index(root))
    reviews = store.load_reviews(root)
    for x in r.items:
        x["review"] = store.review_state(reviews, x)
        r.by_id[x["id"]] = x
    disp = {e["category"]: e["display"] for e in r.events}
    for a in r.activities:
        a.setdefault("supporting_labels", [disp.get(c, c) for c in a.get("supporting_categories", [])])
    return r


def label_of(x: dict) -> str:
    return x["display"]


def matches(x: dict, date: str | None = None, h_from: int = 0, h_to: int = 24, conf: str = "all",
            show_incorrect: bool = False, category: str | None = None) -> bool:
    if not show_incorrect and x.get("review") == "incorrect":
        return False
    if conf == "high" and x.get("confidence") != "high":
        return False
    if category is not None and label_of(x) != category:
        return False
    if date is not None and item_date(x) != date:
        return False
    if (h_from, h_to) != (0, 24):
        a, b = item_hours(x)
        if b < h_from or a >= h_to:          # overlap with the selected hours
            return False
    return True


def filter_items(res: Results, **kw) -> list[dict]:
    return sorted((x for x in res.items if matches(x, **kw)), key=lambda x: (x["start"], x.get("kind") == "activity"))


def counts(res: Results, **kw) -> list[tuple[str, int, bool]]:
    """(label, number of occurrences, is_activity) for categories with detections under the filter."""
    kw = dict(kw)
    kw.pop("category", None)
    c: Counter = Counter()
    act: dict[str, bool] = {}
    for x in res.items:
        if matches(x, **kw):
            c[label_of(x)] += 1
            act[label_of(x)] = x.get("kind") == "activity"
    return sorted(((k, n, act[k]) for k, n in c.items()), key=lambda t: (t[2], -t[1], t[0]))


def group_by_date_hour(items: list[dict]) -> list[tuple[str, list[tuple[int, list[dict]]]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    for x in items:
        st = local(datetime.fromisoformat(x["start"]))
        out.setdefault(item_date(x), {}).setdefault(st.hour, []).append(x)
    return [(d, sorted(h.items())) for d, h in sorted(out.items())]


def supporting(res: Results, activity: dict) -> list[dict]:
    return [res.by_id[i] for i in activity.get("supporting", []) if i in res.by_id]


@dataclass
class PlayTarget:
    path: Path | None
    start: float                 # seconds in that file where playback starts (pre-roll included)
    event_start: float           # seconds in that file where the event starts
    event_end: float
    kind: str                    # "original" | "enhanced"
    note: str = ""


def play_target(res: Results, ev: dict, enhanced: bool = False) -> PlayTarget:
    if ev.get("kind") == "activity":
        sup = supporting(res, ev)
        if not sup:
            return PlayTarget(None, 0, 0, 0, "original", "No supporting sound is available.")
        ev = sup[0]
    if enhanced:
        r = ev.get("refined")
        if r:
            p = res.root / r["file"]
            if p.is_file():
                off = float(r["offset"])
                return PlayTarget(p, max(0.0, off - PRE_ROLL_S), off, off + (ev["src_end"] - ev["src_start"]),
                                  "enhanced")
        return PlayTarget(None, 0, 0, 0, "enhanced", "No enhanced audio exists for this moment (process the folder "
                                                     "to create it).")
    p = store.resolve_source(res.root, ev["source_sha256"], res.index)
    if p is None:
        return PlayTarget(None, 0, 0, 0, "original", f"The recording '{ev.get('source_name')}' was not found in "
                                                     "the folder or in 'processed'.")
    return PlayTarget(p, max(0.0, float(ev["src_start"]) - PRE_ROLL_S), float(ev["src_start"]), float(ev["src_end"]),
                      "original")


def set_review(res: Results, ev: dict, state: str) -> None:
    store.set_review(res.root, ev, state)
    ev["review"] = state
