"""Curated sound vocabulary: which AudioSet classes make up each category, and how each category
is detected (thresholds, minimum evidence, merging). Loaded from ``assets/sound/categories.json``,
which is produced by the evaluation in ``tools/sound`` (see ACOUSTIC_ANALYSIS.md).

Kinds:

* ``impact``    – short sounds (knock, bang, door slam); one window is enough, repeated hits close
                  together are merged into one event with a count;
* ``vocal``     – non-speech sounds of people and animals (laughter, crying, barking);
* ``sustained`` – sounds that last (hairdryer, vacuum cleaner, running water); need several
                  consecutive windows of evidence;
* ``media``     – music, television, video game music: their presence also marks other sounds in
                  the same stretch as *possibly coming from a playback device*.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

from .model import asset_dir, labels

CATEGORIES_FILE = "categories.json"


@dataclass(frozen=True)
class Category:
    id: str
    display: str
    group: str
    kind: str
    labels: tuple[str, ...]
    moderate: float
    high: float
    min_windows: int
    merge_gap: float
    enabled: bool
    note: str = ""
    index: tuple[int, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class Vocabulary:
    version: str
    categories: tuple[Category, ...]
    media_context: float
    media_ids: tuple[str, ...]
    digest: str

    @property
    def enabled(self) -> list[Category]:
        return [c for c in self.categories if c.enabled]

    def by_id(self, cid: str) -> Category | None:
        for c in self.categories:
            if c.id == cid:
                return c
        return None


def vocabulary_file():
    override = os.environ.get("AUDIOREFINE_SOUND_VOCAB")      # tests only
    return override if override else asset_dir() / CATEGORIES_FILE


def load() -> Vocabulary:
    p = str(vocabulary_file())
    st = os.stat(p)
    return _load(p, st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=4)
def _load(path: str, _mtime: int = 0, _size: int = 0) -> Vocabulary:
    raw = open(path, "rb").read()
    d = json.loads(raw.decode("utf-8"))
    names = {n: i for i, n in enumerate(labels())}
    cats = []
    for c in d["categories"]:
        idx = tuple(names[n] for n in c["labels"])
        cats.append(Category(c["id"], c["display"], c["group"], c["kind"], tuple(c["labels"]), float(c["moderate"]),
                             float(c["high"]), int(c["min_windows"]), float(c["merge_gap"]), bool(c["enabled"]),
                             c.get("note", ""), idx))
    return Vocabulary(d["version"], tuple(cats), float(d["media_context"]), tuple(d["media_ids"]),
                      hashlib.sha256(raw).hexdigest()[:16])
