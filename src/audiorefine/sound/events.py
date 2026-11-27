"""From window scores to events: one event per continuous occurrence of a sound, not one per window.

For every enabled category (``vocab.Category``):

1. **Hysteresis** – an occurrence starts where the window score reaches the category's *moderate*
   threshold and extends over neighbouring windows while the score stays above half of it, so a
   hairdryer that runs for six minutes with a few weaker windows stays one event.
2. **Merging** – occurrences closer than the category's ``merge_gap`` become one event (repeated
   knocks → one "Knocking" event with a hit count; a vacuum cleaner switched off for a moment
   → one event).
3. **Evidence** – an event needs at least ``min_windows`` windows at or above the moderate
   threshold (sustained sounds need several consecutive seconds; a knock needs one window).
4. **Confidence** – *High* when the peak reaches the category's high threshold (sustained sounds:
   in at least two windows), otherwise *Moderate*. Below moderate nothing is reported.
5. **Playback context** – stretches where music, television or video game music is detected are
   recorded; other events inside them are marked *during media playback* (they may come from the
   loudspeaker rather than from the room) and are never rated High.
6. **Repetition** – when one sound gives more than ``FREQUENT_PER_HOUR`` moderate detections per
   hour in a recording (at least 10 minutes long), its moderate detections are dropped for that
   recording and only the high-confidence ones are kept (a constantly re-triggering sound is more
   likely a property of the place or the microphone than a series of events).

Times are seconds from the start of the recording. An event covers the middle half of its
first to last window (windows are 1 s long, every 0.5 s), so start and end are accurate to about
±0.5 s.
"""
from __future__ import annotations

import numpy as np

from .vocab import Vocabulary

OFF_RATIO = 0.5          # hysteresis: continue while score >= OFF_RATIO * moderate
MEDIA_GAP = 5.0          # media context spans closer than this are joined
FREQUENT_PER_HOUR = 20   # moderate detections per hour above which only high ones are kept


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of consecutive True values."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    a = np.flatnonzero(d == 1)
    b = np.flatnonzero(d == -1) - 1
    return list(zip(a.tolist(), b.tolist()))


def hysteresis(score: np.ndarray, on: float, off: float) -> list[tuple[int, int]]:
    seeds = score >= on
    if not seeds.any():
        return []
    out = []
    for a, b in _runs(score >= off):
        if seeds[a:b + 1].any():
            out.append((a, b))
    return out


def window_span(starts: np.ndarray, i0: int, i1: int, win_s: float, hop_s: float, duration: float) -> tuple[float, float]:
    pad = max(0.0, (win_s - hop_s) / 2)
    a = float(starts[i0]) + pad
    b = float(starts[i1]) + win_s - pad
    if i0 == 0:
        a = 0.0 if starts[0] <= 0 else a
    return max(0.0, min(a, duration)), max(0.0, min(max(b, a), duration))


def media_spans(S: np.ndarray, starts: np.ndarray, vocab: Vocabulary, win_s: float, hop_s: float,
                duration: float) -> list[tuple[float, float]]:
    cols = [i for i, c in enumerate(vocab.categories) if c.id in vocab.media_ids]
    if not cols or len(starts) == 0:
        return []
    m = S[:, cols].max(axis=1) >= vocab.media_context
    spans = [window_span(starts, a, b, win_s, hop_s, duration) for a, b in _runs(m)]
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= MEDIA_GAP:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _overlap(a: float, b: float, spans: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans)


def build_events(S: np.ndarray, starts: np.ndarray, vocab: Vocabulary, win_s: float, hop_s: float,
                 duration: float) -> tuple[list[dict], list[tuple[float, float]]]:
    """S: [n_windows, n_categories] scores in vocabulary order; starts: window start times (s)."""
    events: list[dict] = []
    if len(starts) == 0:
        return events, []
    media = media_spans(S, starts, vocab, win_s, hop_s, duration)
    for j, c in enumerate(vocab.categories):
        if not c.enabled:
            continue
        s = S[:, j]
        occ = hysteresis(s, c.moderate, OFF_RATIO * c.moderate)
        groups: list[list[tuple[int, int]]] = []
        for a, b in occ:
            if groups:
                pa, pb = groups[-1][-1]
                if float(starts[a]) - (float(starts[pb]) + win_s) <= c.merge_gap:
                    groups[-1].append((a, b))
                    continue
            groups.append([(a, b)])
        for g in groups:
            i0, i1 = g[0][0], g[-1][1]
            seg = s[i0:i1 + 1]
            n_mod = int((seg >= c.moderate).sum())
            if n_mod < c.min_windows:
                continue
            n_high = int((seg >= c.high).sum())
            peak = float(seg.max())
            a, b = window_span(starts, i0, i1, win_s, hop_s, duration)
            if b <= a:
                continue
            high = peak >= c.high and (c.kind != "sustained" or n_high >= 2)
            during_media = c.id not in vocab.media_ids and _overlap(a, b, media) >= 0.5 * (b - a)
            if during_media:
                high = False
            events.append({
                "category": c.id, "start": round(a, 2), "end": round(b, 2),
                "confidence": "high" if high else "moderate",
                "score_peak": round(peak, 4), "score_mean": round(float(seg[seg >= OFF_RATIO * c.moderate].mean()), 4),
                "windows": n_mod, "hits": len(g), "during_media": bool(during_media),
            })
    if duration >= 600:
        hours = duration / 3600
        for cid in {e["category"] for e in events}:
            mods = [e for e in events if e["category"] == cid and e["confidence"] == "moderate"]
            if len(mods) / hours > FREQUENT_PER_HOUR:
                events = [e for e in events if not (e["category"] == cid and e["confidence"] == "moderate")]
                for e in events:
                    if e["category"] == cid:
                        e["frequent"] = True
    events.sort(key=lambda e: (e["start"], e["category"]))
    return events, media
