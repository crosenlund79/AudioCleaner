"""Possible activities suggested by combinations of detected sounds.

An activity is only ever a *suggestion* ("Possible cooking"), never a fact, and always lists the
detected sounds it is based on. A rule needs several *different* supporting sounds close together
in time (one sound alone is reported as that sound, not as an activity). Sounds marked *during
media playback* or reviewed as incorrect are not used. Nothing about intentions, people,
relationships or emotions is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Rule:
    id: str
    display: str
    support: frozenset[str]        # categories that count as evidence
    core: frozenset[str]           # at least one of these must be present
    min_distinct: int              # number of different supporting categories required
    gap_s: float                   # supporting sounds further apart than this start a new cluster
    description: str


RULES = (
    Rule("cooking", "Possible cooking", frozenset({"frying", "chopping", "dishes", "cutlery", "blender", "microwave",
                                                    "boiling", "tap_water"}),
         frozenset({"frying", "chopping", "blender", "microwave", "boiling"}), 2, 600.0,
         "frying/sizzling, chopping, a blender, a microwave or boiling together with other kitchen sounds"),
    Rule("washing_up", "Possible washing up", frozenset({"tap_water", "dishes", "cutlery"}),
         frozenset({"tap_water"}), 2, 180.0,
         "running water together with dishes or cutlery"),
)


def infer(events: list[dict]) -> list[dict]:
    """events: detected sounds with ``category``, ``start``/``end`` (ISO), ``confidence``, ``id``."""
    out = []
    usable = [e for e in events if not e.get("during_media") and e.get("review") != "incorrect"]
    for rule in RULES:
        evs = sorted((e for e in usable if e["category"] in rule.support), key=lambda e: e["start"])
        cluster: list[dict] = []
        last_end: datetime | None = None

        def flush():
            cats = {e["category"] for e in cluster}
            if len(cats) >= rule.min_distinct and cats & rule.core:
                st = min(datetime.fromisoformat(e["start"]) for e in cluster)
                en = max(datetime.fromisoformat(e["end"]) for e in cluster)
                n_high = sum(1 for e in cluster if e["confidence"] == "high")
                conf = "high" if len(cats) >= 3 and n_high >= 2 else "moderate"
                out.append({
                    "id": f"act-{rule.id}-{st.strftime('%Y%m%d%H%M%S')}",
                    "kind": "activity", "category": rule.id, "display": rule.display,
                    "start": st.isoformat(), "end": en.isoformat(), "confidence": conf,
                    "estimated": any(e.get("estimated") for e in cluster),
                    "supporting": [e["id"] for e in cluster],
                    "supporting_categories": sorted(cats),
                    "basis": rule.description,
                })

        for e in evs:
            st, en = datetime.fromisoformat(e["start"]), datetime.fromisoformat(e["end"])
            if cluster and last_end is not None and st - last_end > timedelta(seconds=rule.gap_s):
                flush()
                cluster = []
                last_end = None
            cluster.append(e)
            last_end = en if last_end is None else max(last_end, en)
        if cluster:
            flush()
    out.sort(key=lambda a: a["start"])
    return out
