"""Detection of typical Whisper hallucinations and confidence flags.

Whisper was trained on subtitled video and is known to emit subtitle credits and outros
("Tak fordi du så med", "Undertekster af ...", "Thanks for watching") on silence or noise, and
to loop on a phrase. These checks never invent or rewrite words: a segment is either kept
(possibly flagged as uncertain), shortened by removing an obvious repetition loop (flagged), or
dropped when it matches a known artefact *and* the acoustic evidence for speech is weak.
"""
from __future__ import annotations

import re
import unicodedata

# Normalised (lower-case, no punctuation) phrases that Whisper is known to hallucinate.
KNOWN_ARTEFACTS = (
    "tak fordi du så med", "tak for at se med", "tak fordi i så med", "tak for i dag",
    "undertekster af", "danske tekster", "dansk tekst", "tekstet af", "tekster af", "oversat af",
    "undertekster", "teksten er lavet af", "tekstning", "abonner på kanalen", "husk at abonnere",
    "thank you for watching", "thanks for watching", "thank you so much for watching",
    "please subscribe", "like and subscribe", "subtitles by", "subtitled by", "amaraorg",
    "transcribed by", "transcription by", "captions by", "www", "com", "♪", "copyright",
    "nextvideo", "see you in the next video", "bye bye", "music", "musik",
)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = _PUNCT.sub(" ", t)
    return " ".join(t.split())


def is_known_artefact(text: str) -> bool:
    n = normalise(text)
    if not n:
        return True
    for a in KNOWN_ARTEFACTS:
        if n == a or (len(a) > 8 and a in n and len(n) <= len(a) + 25):
            return True
    return False


def collapse_repetition(words: list[str], max_repeats: int = 3) -> tuple[int, int] | None:
    """Find a phrase of 1-8 words repeated consecutively more than ``max_repeats`` times.

    Returns (start_index, end_index) of the redundant tail to remove, keeping ``max_repeats``
    occurrences, or None. Single-word repeats need 8+ occurrences ("ja ja ja" is real speech).
    """
    norm = [normalise(w) for w in words]
    n = len(norm)
    for size in range(1, 9):
        need = 8 if size == 1 else max_repeats + 1
        i = 0
        while i + size * need <= n:
            unit = norm[i:i + size]
            if not any(unit):
                i += 1
                continue
            k = 1
            while i + size * (k + 1) <= n and norm[i + size * k:i + size * (k + 1)] == unit:
                k += 1
            if k >= need:
                keep = max_repeats if size > 1 else 4
                return i + size * keep, i + size * k
            i += 1
    return None


def segment_is_uncertain(avg_logprob: float, word_probs: list[float], temperature: float,
                         compression_ratio: float) -> bool:
    if avg_logprob < -0.9:
        return True
    if temperature >= 0.6:
        return True
    if compression_ratio > 2.4:
        return True
    if word_probs:
        low = sum(1 for p in word_probs if p < 0.35) / len(word_probs)
        mean = sum(word_probs) / len(word_probs)
        if mean < 0.55 or low > 0.35:
            return True
    return False
