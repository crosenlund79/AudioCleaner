"""Language selection policy applied to every 30-second Whisper window.

Whisper's language identification is run per window (faster-whisper ``multilingual=True``) on
the same encoder output that is used for decoding, so it costs nothing extra. The raw decision
is post-processed because:

* Danish is frequently mis-identified as Norwegian (no/nn) or Swedish by Whisper. Transcribing
  Danish speech with a Norwegian language token produces Norwegian spellings, so the
  probability mass of these closely related languages is attributed to Danish.
* Very short or noisy windows give unstable decisions, so switching language requires clear
  evidence (hysteresis). This prevents a Danish conversation being transcribed in English
  because of one ambiguous window, and vice versa.
* Danish and English are the supported languages. Another language is only accepted when
  Whisper is highly confident and neither Danish nor English is plausible; the spoken language
  is always preserved (the task is always "transcribe", never "translate").
"""
from __future__ import annotations

import threading
from collections import Counter

SCANDINAVIAN = ("da", "no", "nn", "sv")
SUPPORTED = ("da", "en")


class LanguagePolicy:
    SWITCH_MIN = 0.60       # score the new language needs to replace the current one
    OTHER_MIN = 0.90        # confidence required to accept a language other than da/en
    OTHER_MAX_SUPPORTED = 0.05

    def __init__(self, default: str = "da"):
        self.default = default
        self.current: str | None = None
        self.counts: Counter[str] = Counter()
        self.switches = 0
        self.overridden_scandinavian = 0
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.current = None

    def choose(self, probs: dict[str, float]) -> str:
        da = sum(probs.get(c, 0.0) for c in SCANDINAVIAN)
        en = probs.get("en", 0.0)
        top_lang = max(probs, key=probs.get) if probs else self.default
        top_p = probs.get(top_lang, 0.0)
        with self._lock:
            if top_lang in ("no", "nn", "sv"):
                self.overridden_scandinavian += 1
            if top_lang not in SCANDINAVIAN and top_lang != "en" and top_p >= self.OTHER_MIN \
                    and max(da, en) <= self.OTHER_MAX_SUPPORTED:
                choice = top_lang
            else:
                scores = {"da": da, "en": en}
                best = max(scores, key=scores.get)
                cur = self.current if self.current in scores else None
                if cur is None:
                    choice = best if scores[best] > 0 else self.default
                elif best != cur and scores[best] >= self.SWITCH_MIN and scores[best] > scores[cur]:
                    choice = best
                else:
                    choice = cur
            if self.current is not None and choice != self.current:
                self.switches += 1
            self.current = choice
            self.counts[choice] += 1
            return choice

    def summary(self) -> dict:
        return {
            "windows_per_language": dict(self.counts),
            "language_switches": self.switches,
            "norwegian_or_swedish_detections_attributed_to_danish": self.overridden_scandinavian,
        }


class LanguagePolicyModel:
    """Wraps a ctranslate2 Whisper model; only ``detect_language`` is altered."""

    def __init__(self, inner, policy: LanguagePolicy):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_policy", policy)

    def detect_language(self, *args, **kwargs):
        results = self._inner.detect_language(*args, **kwargs)
        out = []
        for res in results:
            probs = {tok[2:-2]: float(p) for tok, p in res}
            lang = self._policy.choose(probs)
            token = f"<|{lang}|>"
            chosen = [(token, probs.get(lang, 0.0))]
            out.append(chosen + [(t, p) for t, p in res if t != token])
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)
