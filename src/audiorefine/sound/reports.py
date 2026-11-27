"""Readable and machine-readable sound event reports, and the annotated transcript.

Everything here is regenerated from the hour records (``Sound_Events.json``), the review marks and
– for the annotated transcript – the transcription's hour manifests. Nothing is analysed again.
"""
from __future__ import annotations

import io
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from ..constants import MASTER_SR, REFINED_DIR, SEGMENT_SAMPLES
from ..hours import HourKey, clock_at, hour_key_of, load_complete_manifests
from ..hourtext import atomic_write_text, fmt_offset, hour_entries
from ..rectime import fmt_clock, local
from . import activities as act
from .store import (ANNOTATED_TXT, DAY_TXT, HOUR_TXT, MASTER_JSON, MASTER_TXT, adir, hour_dir, hour_records,
                    load_index, load_reviews, review_state, same_fingerprint)

RULE = "=" * 50
RULE2 = "-" * 30

ABOUT = ("About this report: the sounds below were recognised automatically by a sound classifier "
         "(not checked by a person). Times are approximate (about ±1 second) and a detection may be wrong. "
         "Only detections of high or moderate confidence are listed; confidence is the classifier's own "
         "rating, not a measured probability. A sound that is not listed may still have occurred. Sounds "
         "from a television, radio, phone or game can be recognised as if they happened in the room; "
         "detections during music or television audio are marked. 'Possible activity' entries are "
         "suggestions based on several detected sounds, never established facts. Times marked ≈ are "
         "estimated from the file dates because the recording time is not in the file name or the file.")


def conf_text(c: str) -> str:
    return "High" if c == "high" else "Moderate"


def span_text(ev: dict, with_date: bool = False) -> str:
    st, en = datetime.fromisoformat(ev["start"]), datetime.fromisoformat(ev["end"])
    a = fmt_clock(st, ev.get("estimated", False), with_date=with_date)
    b = local(en).strftime("%H:%M:%S")
    if local(en).date() != local(st).date():
        b = local(en).strftime("%Y-%m-%d %H:%M:%S")
    return f"{a} – {b}"


# ------------------------------------------------------------------------------ refined audio references
def refined_index(root: Path) -> dict[str, list[tuple[int, int, int, HourKey]]]:
    idx: dict[str, list[tuple[int, int, int, HourKey]]] = defaultdict(list)
    for key, m in load_complete_manifests(root).items():
        for p in m.get("pieces", []):
            idx[p["sha256"]].append((int(p["src_start"]), int(p["src_end"]), int(p["out_start"]), key))
    return idx


def refined_ref(root: Path, ridx: dict, sha: str, src_seconds: float) -> dict | None:
    s = int(round(src_seconds * MASTER_SR))
    for a, b, out_start, key in ridx.get(sha, []):
        if a <= s < b:
            out = out_start + (s - a)
            n = out // SEGMENT_SAMPLES
            f = f"Audio_{n + 1:03d}.flac"
            path = root / REFINED_DIR / key.date / key.folder_name / f
            if path.is_file():
                return {"hour": str(key), "file": f"{REFINED_DIR}/{key.date}/{key.folder_name}/{f}",
                        "offset": round((out - n * SEGMENT_SAMPLES) / MASTER_SR, 2)}
    return None


# ------------------------------------------------------------------------------ collection
def collect(root: Path, fp: dict) -> dict:
    """All current results: events (with review state and refined reference), activities, info."""
    recs = hour_records(root)
    current = {k: r for k, r in recs.items() if same_fingerprint(r.get("fingerprint"), fp)}
    reviews = load_reviews(root)
    ridx = refined_index(root)
    index = load_index(root)
    events = []
    for k in sorted(current):
        for e in current[k]["events"]:
            e = dict(e)
            e["review"] = review_state(reviews, e)
            e["refined"] = refined_ref(root, ridx, e["source_sha256"], e["src_start"])
            events.append(e)
    events.sort(key=lambda e: (e["start"], e["category"]))
    acts = act.infer(events)
    for a in acts:
        a["review"] = review_state(reviews, a)
    return {"hours": current, "stale_hours": sorted(set(recs) - set(current)), "events": events,
            "activities": acts, "sources": index.get("sources", {})}


def visible(items: list[dict]) -> list[dict]:
    return [e for e in items if e.get("review") != "incorrect"]


# ------------------------------------------------------------------------------ text blocks
def event_block(e: dict, with_date: bool = False) -> str:
    lines = [span_text(e, with_date)]
    if e.get("kind") == "activity":
        lines.append(f"Possible activity: {e['display'].replace('Possible ', '').capitalize()}")
        lines.append(f"Confidence: {conf_text(e['confidence'])}")
        lines.append("Supporting sounds: " + ", ".join(e.get("supporting_labels") or e["supporting_categories"]))
    else:
        lines.append(f"Detected sound: {e['display']}")
        lines.append(f"Confidence: {conf_text(e['confidence'])}")
        if e.get("hits", 1) > 1:
            lines.append(f"Repeated: {e['hits']} occurrences close together")
        if e.get("during_media"):
            lines.append("Note: during music/TV audio – may come from a playback device")
        if e.get("frequent"):
            lines.append("Note: detected very often in this recording – only high-confidence detections are listed")
        src = f"Source: {e['source_name']} at {fmt_offset(e['src_start'])}"
        if e.get("refined"):
            r = e["refined"]
            src += f" (enhanced audio: {r['file'].replace('/', chr(92))} at {fmt_offset(r['offset'])})"
        lines.append(src)
    if e.get("review") == "confirmed":
        lines.append("Reviewed: confirmed by the user")
    return "\n".join(lines)


def _label_activities(acts: list[dict], events: list[dict]) -> None:
    disp = {e["category"]: e["display"] for e in events}
    for a in acts:
        a["supporting_labels"] = [disp.get(c, c) for c in a["supporting_categories"]]


def render_hour(key: HourKey, record: dict, reviews: dict) -> str:
    buf = io.StringIO()
    buf.write("AudioRefine – sound events of one hour\n")
    buf.write(f"Recording date: {key.date}\nRecording hour: {key.label}\n")
    buf.write("Recordings in this hour:\n")
    for s in record["sources"]:
        buf.write(f"  {s['name']}  from {fmt_clock(datetime.fromisoformat(s['hour_start']), s['estimated'])}"
                  f"  [time from {s['time_basis']}]\n")
    buf.write("\n" + ABOUT + "\n\n")
    evs = []
    for e in record["events"]:
        e = dict(e, review=review_state(reviews, e))
        if e["review"] != "incorrect":
            evs.append(e)
    if not evs:
        buf.write("(No sound events of high or moderate confidence were detected in this hour.)\n")
    else:
        buf.write("\n\n".join(event_block(e) for e in evs) + "\n")
    return buf.getvalue()


def write_hour_txt(root: Path, key: HourKey, record: dict) -> None:
    atomic_write_text(hour_dir(root, key) / HOUR_TXT, render_hour(key, record, load_reviews(root)))


def _date_of(item: dict) -> str:
    return local(datetime.fromisoformat(item["start"])).strftime("%Y-%m-%d")


def _hour_of(item: dict) -> HourKey:
    return hour_key_of(datetime.fromisoformat(item["start"]))


def render_day(date: str, data: dict) -> str:
    evs = [e for e in visible(data["events"]) if _date_of(e) == date]
    acts = [a for a in visible(data["activities"]) if _date_of(a) == date]
    hours = sorted(k for k in data["hours"] if k.date == date)
    buf = io.StringIO()
    buf.write(f"{RULE}\nSOUND EVENTS – DATE: {date}\n{RULE}\n\n")
    buf.write(f"Hours analysed: {len(hours)} ({', '.join(k.label for k in hours)})\n\n{ABOUT}\n\n")
    items = sorted(evs + acts, key=lambda x: (x["start"], x.get("kind") == "activity"))
    for k in hours:
        sect = [x for x in items if _hour_of(x) == k]
        buf.write(f"{RULE2}\n{k.label}\n{RULE2}\n\n")
        buf.write(("\n\n".join(event_block(x) for x in sect) if sect else "(No sound events detected.)") + "\n\n")
    return buf.getvalue()


def render_master(data: dict, status: str) -> str:
    evs, acts = visible(data["events"]), visible(data["activities"])
    hours = sorted(data["hours"])
    dates = sorted({k.date for k in hours})
    buf = io.StringIO()
    buf.write(f"{RULE}\nAUDIOREFINE — ACOUSTIC EVENT REPORT\n{RULE}\n\n")
    buf.write(f"Status: {status}\nUpdated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    buf.write(f"Dates analysed: {len(dates)}\nHours analysed: {len(hours)}\n")
    buf.write(f"Detected sound events: {len(evs)}\nPossible activities: {len(acts)}\n")
    if data["stale_hours"]:
        buf.write(f"Not included: {len(data['stale_hours'])} hour(s) analysed by an older version whose "
                  f"recordings are no longer in the folder.\n")
    counts = Counter(e["display"] for e in evs)
    if counts:
        buf.write("\nDetections per sound:\n")
        for name, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            buf.write(f"  {name:<34} {n}\n")
    buf.write(f"\n{ABOUT}\n\n")
    items = sorted(evs + acts, key=lambda x: (x["start"], x.get("kind") == "activity"))
    for d in dates:
        buf.write(f"{RULE}\nDATE: {d}\n{RULE}\n\n")
        for k in [h for h in hours if h.date == d]:
            sect = [x for x in items if _hour_of(x) == k]
            buf.write(f"{RULE2}\nHOUR: {k.label}\n{RULE2}\n\n")
            buf.write(("\n\n".join(event_block(x) for x in sect) if sect else "(No sound events detected.)") + "\n\n")
    buf.write(f"{RULE}\nEND OF REPORT\n{RULE}\n")
    return buf.getvalue()


def master_json(data: dict, status: str, fp: dict, vocab) -> dict:
    return {
        "app": "AudioRefine", "kind": "sound-events-master", "status": status,
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"), "fingerprint": fp,
        "categories": [{"id": c.id, "display": c.display, "group": c.group, "kind": c.kind} for c in vocab.enabled],
        "hours": [str(k) for k in sorted(data["hours"])], "stale_hours": [str(k) for k in data["stale_hours"]],
        "sources": data["sources"], "events": data["events"], "activities": data["activities"],
        "note": "score_peak / score_mean are internal classifier scores (0-1), not probabilities.",
    }


def write_combined(root: Path, fp: dict, vocab, status: str, dates: set[str] | None = None) -> dict:
    data = collect(root, fp)
    _label_activities(data["activities"], data["events"])
    for d in sorted(dates if dates is not None else {k.date for k in data["hours"]}):
        p = adir(root) / d
        if p.is_dir():
            atomic_write_text(p / DAY_TXT, render_day(d, data))
    atomic_write_text(adir(root) / MASTER_TXT, render_master(data, status))
    from ..hours import write_json_atomic
    write_json_atomic(adir(root) / MASTER_JSON, master_json(data, status, fp, vocab))
    return data


# ------------------------------------------------------------------------------ annotated transcript
ANNOT_REPEAT_S = 120.0


def _annotation(x: dict) -> str:
    if x.get("kind") == "activity":
        return (f"[Possible activity: {x['display'].replace('Possible ', '').capitalize()} — "
                f"{x['confidence']} confidence, {span_text(x)}; based on: "
                f"{', '.join(x.get('supporting_labels') or x['supporting_categories']).lower()}]")
    extra = []
    if x.get("_repeat", 1) > 1:
        extra.append(f"{x['_repeat']} times")
    if x.get("during_media"):
        extra.append("during music/TV audio")
    tail = f", {span_text(x)}" + (f", {', '.join(extra)}" if extra else "")
    return f"[Detected sound: {x['display']} — {x['confidence']} confidence{tail}]"


def write_annotated(root: Path, data: dict) -> Path | None:
    """Transcript + sound annotations, for hours that have both. Never changes the clean transcript."""
    manifests = load_complete_manifests(root)
    if not manifests:
        return None
    analysed = set(data["hours"])
    items: list[tuple[datetime, int, str]] = []
    keys = sorted(k for k in manifests if k in analysed)
    if not keys:
        return None
    for k in keys:
        m = manifests[k]
        for t, order, block in hour_entries(m):
            dt, _, _ = clock_at(m["pieces"], t)
            if dt is not None:
                items.append((dt, order, block))
    # annotations: one line per event; the same sound repeating within 2 minutes is folded into one line
    sounds = [e for e in visible(data["events"]) if _hour_of(e) in analysed and _hour_of(e) in manifests]
    folded: list[dict] = []
    last: dict[str, dict] = {}
    for e in sounds:
        p = last.get(e["category"])
        if p is not None and datetime.fromisoformat(e["start"]) - datetime.fromisoformat(p["end"]) <= timedelta(seconds=ANNOT_REPEAT_S):
            p["end"] = max(p["end"], e["end"])
            p["_repeat"] = p.get("_repeat", 1) + 1
            if e["confidence"] == "high":
                p["confidence"] = "high"
            continue
        q = dict(e)
        folded.append(q)
        last[e["category"]] = q
    for x in folded + [a for a in visible(data["activities"]) if _hour_of(a) in manifests]:
        st = datetime.fromisoformat(x["start"])
        items.append((st, 0 if x.get("kind") == "activity" else 1,
                      f"[{fmt_clock(st, x.get('estimated', False))}]\n{_annotation(x)}"))
    items.sort(key=lambda i: (i[0], i[1]))
    buf = io.StringIO()
    buf.write(f"{RULE}\nAUDIOREFINE — MASTER TRANSCRIPT WITH SOUND ANNOTATIONS\n{RULE}\n\n")
    buf.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    buf.write(f"Hours included: {len(keys)} (hours that have both a transcript and a sound analysis)\n\n")
    buf.write("The spoken text is exactly the text of refined\\Transcript_Master_Complete.txt (unchanged). "
              "Lines in square brackets starting with 'Detected sound' or 'Possible activity' are added "
              "automatically by the sound analysis: approximate, possibly wrong, and never verified facts. "
              "A sound repeated within two minutes is shown once with the number of times.\n\n")
    cur_date = cur_hour = None
    for dt, _, block in items:
        k = hour_key_of(dt)
        if k.date != cur_date:
            buf.write(f"{RULE}\nDATE: {k.date}\n{RULE}\n\n")
            cur_date, cur_hour = k.date, None
        if k != cur_hour:
            buf.write(f"{RULE2}\nHOUR: {k.label}\n{RULE2}\n\n")
            cur_hour = k
        buf.write(block + "\n\n")
    buf.write(f"{RULE}\nEND OF TRANSCRIPT\n{RULE}\n")
    p = adir(root) / ANNOTATED_TXT
    atomic_write_text(p, buf.getvalue())
    return p
