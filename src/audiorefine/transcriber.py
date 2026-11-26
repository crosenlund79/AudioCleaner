"""Speech recognition over the complete stitched recording.

Strategy (see AUDIO_PROCESSING.md, "Transcription"):

1. Silero VAD speech regions (whole recording) are merged into *clips* of at most 30 s of
   timeline (gaps of at most 4 s are kept inside a clip so short utterances get context).
2. Clips are grouped into *blocks* of up to ~20 minutes whose boundaries always lie in a pause.
   A block is decoded by faster-whisper with ``clip_timestamps`` so only speech is decoded,
   while timestamps stay on the original timeline (no audio is removed or shifted).
3. Context flows across clips and blocks: Whisper conditions on the previously decoded text,
   and each block starts with the tail of the previous block's text as prompt.
   The exported 10-minute audio segmentation plays no role here, so sentences that cross a
   10-minute boundary are decoded in one piece.
4. Each clip is level-normalised before recognition (quiet speech is presented to Whisper at a
   normal level). The listening path is not used for recognition: Whisper is more accurate on
   the unprocessed signal than on noise-reduced audio (TEST_REPORT.md).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .constants import ASR_SR
from .hallucination import (MAX_PLAUSIBLE_CPS, chars_per_second, collapse_repetition, find_immediate_repeat,
                            is_known_artefact, normalise, segment_is_uncertain)
from .language import LanguagePolicy, LanguagePolicyModel
from .rawaudio import raw_length, read_raw
from .transcript import TSegment, TWord

log = logging.getLogger("audiorefine.transcriber")

CLIP_MAX_S = 30.0
CLIP_MAX_GAP_S = 4.0
REGION_MAX_S = 600.0        # longer continuous "speech" (e.g. a TV) is cut at its quietest moment
BLOCK_MAX_S = 1200.0
TARGET_P95_DBFS = -14.0     # 95th percentile of 30 ms frame levels inside the clip
MAX_GAIN_DB = 40.0
MIN_GAIN_DB = -12.0
PROMPT_CHARS = 200


@dataclass
class Block:
    start: int                      # 16 kHz samples, absolute
    end: int
    clips: list[tuple[int, int]] = field(default_factory=list)


def _split_long_region(asr_raw: Path | None, a: int, b: int, audio: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Cut a region longer than REGION_MAX_S at the quietest 200 ms within +-15 s of each target."""
    out = []
    max_len = int(REGION_MAX_S * ASR_SR)
    cur = a
    while b - cur > max_len:
        target = cur + max_len - int(15 * ASR_SR)
        lo, hi = target - int(15 * ASR_SR), min(b, target + int(15 * ASR_SR))
        seg = audio[lo:hi] if audio is not None else read_raw(asr_raw, lo, hi - lo)
        fr = int(0.2 * ASR_SR)
        n = len(seg) // fr
        if n >= 1:
            e = np.square(seg[: n * fr].reshape(n, fr), dtype=np.float64).mean(axis=1)
            cut = lo + int(np.argmin(e)) * fr + fr // 2
        else:
            cut = target
        out.append((cur, cut))
        cur = cut
    out.append((cur, b))
    return out


def _split_at(regions: list[tuple[int, int]], breaks: list[int]) -> list[tuple[int, int]]:
    """Split regions at the given positions (junctions between recordings)."""
    if not breaks:
        return regions
    out = []
    for a, b in regions:
        cur = a
        for k in breaks:
            if cur < k < b:
                if k - cur > ASR_SR // 10:
                    out.append((cur, k))
                cur = k
        if b - cur > ASR_SR // 10 or cur == a:
            out.append((cur, b))
    return out


def plan_clips(regions: list[tuple[int, int]], asr_raw: Path | None = None, audio: np.ndarray | None = None,
               breaks: list[int] | None = None) -> list[tuple[int, int]]:
    """Merge VAD regions into recognition clips (<= 30 s, gaps <= 4 s), never across ``breaks``.

    Clips never span the junction between two recordings: a new recording often means a new
    situation, speaker or language, and a 30-second Whisper window decodes in one language.
    """
    breaks = sorted(breaks or [])
    pieces: list[tuple[int, int]] = []
    for a, b in _split_at(regions, breaks):
        if b - a > REGION_MAX_S * ASR_SR:
            pieces.extend(_split_long_region(asr_raw, a, b, audio))
        else:
            pieces.append((a, b))
    clips: list[list[int]] = []
    for a, b in pieces:
        if clips:
            ca, cb = clips[-1]
            crosses = any(cb <= k <= a or ca < k < b for k in breaks)
            if (not crosses and b - ca <= CLIP_MAX_S * ASR_SR and a - cb <= CLIP_MAX_GAP_S * ASR_SR
                    and cb - ca < CLIP_MAX_S * ASR_SR):
                clips[-1][1] = b
                continue
        clips.append([a, b])
    return [(a, b) for a, b in clips]


def plan_blocks(clips: list[tuple[int, int]]) -> list[Block]:
    blocks: list[Block] = []
    for a, b in clips:
        if blocks and b - blocks[-1].start <= BLOCK_MAX_S * ASR_SR:
            blocks[-1].clips.append((a, b))
            blocks[-1].end = b
        else:
            blocks.append(Block(a, b, [(a, b)]))
    return blocks


def clip_gain(x: np.ndarray) -> float:
    fr = int(0.03 * ASR_SR)
    n = len(x) // fr
    if n < 1:
        return 1.0
    lv = 10 * np.log10(np.square(x[: n * fr].reshape(n, fr), dtype=np.float64).mean(axis=1) + 1e-12)
    p95 = float(np.percentile(lv, 95))
    g_db = float(np.clip(TARGET_P95_DBFS - p95, MIN_GAIN_DB, MAX_GAIN_DB))
    peak = float(np.percentile(np.abs(x), 99.9)) + 1e-9
    g_db = min(g_db, 20 * math.log10(0.99 / peak))
    return 10 ** (g_db / 20)


def _smooth_probs(p: np.ndarray, k: int = 9) -> np.ndarray:
    """Moving average over ~0.3 s: Silero probabilities fluctuate strongly inside real speech."""
    if p.size < k:
        return p
    return np.convolve(np.pad(p, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


@dataclass
class TranscribeOptions:
    beam_size: int = 5
    normalise_level: bool = True
    condition_on_previous_text: bool = True
    hallucination_silence_threshold: float | None = 2.0


class Transcriber:
    def __init__(self, whisper_model, options: TranscribeOptions | None = None, default_language: str = "da"):
        self.model = whisper_model
        self.opts = options or TranscribeOptions()
        self.policy = LanguagePolicy(default_language)
        if not isinstance(self.model.model, LanguagePolicyModel):
            self.model.model = LanguagePolicyModel(self.model.model, self.policy)
        else:
            object.__setattr__(self.model.model, "_policy", self.policy)
        self.stats = {"segments": 0, "dropped_artefacts": 0, "repetitions_collapsed": 0,
                      "duplicate_segments_dropped": 0, "uncertain_segments": 0, "unclear_markers": 0,
                      "repair_attempts": 0, "repaired_segments": 0, "repair_seconds": 0.0}

    # ------------------------------------------------------------------ main entry
    def transcribe(self, asr_raw: Path | None, regions: list[tuple[int, int]], probs: np.ndarray | None = None,
                   audio: np.ndarray | None = None, progress: Callable[[float], None] | None = None,
                   cancel: Callable[[], None] | None = None, breaks: list[int] | None = None) -> list[TSegment]:
        """Transcribe speech regions of a 16 kHz raw file (or in-memory ``audio``).

        ``breaks`` are the 16 kHz positions where one recording ends and the next begins.
        """
        clips = plan_clips(regions, asr_raw, audio, breaks)
        blocks = plan_blocks(clips)
        total_speech = sum(b - a for a, b in clips) or 1
        done = 0
        prompt: str | None = None
        out: list[TSegment] = []
        log.info("Transcription plan: %d speech regions -> %d clips in %d blocks (%.1f min of speech timeline)",
                 len(regions), len(clips), len(blocks), total_speech / ASR_SR / 60)
        for bi, blk in enumerate(blocks):
            if cancel:
                cancel()
            seg_audio = audio[blk.start:blk.end].copy() if audio is not None else read_raw(asr_raw, blk.start, blk.end - blk.start)
            seg_audio = seg_audio.astype(np.float32, copy=True)
            if self.opts.normalise_level:
                for a, b in blk.clips:
                    sl = slice(a - blk.start, b - blk.start)
                    seg_audio[sl] *= clip_gain(seg_audio[sl])
                np.clip(seg_audio, -1.0, 1.0, out=seg_audio)
            ts: list[float] = []
            for a, b in blk.clips:
                ts += [(a - blk.start) / ASR_SR, (b - blk.start) / ASR_SR]
            offset = blk.start / ASR_SR
            segments, _info = self.model.transcribe(
                seg_audio,
                language=self.policy.current or self.policy.default,
                task="transcribe",
                beam_size=self.opts.beam_size,
                best_of=5,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=self.opts.condition_on_previous_text,
                prompt_reset_on_temperature=0.5,
                initial_prompt=prompt,
                word_timestamps=True,
                hallucination_silence_threshold=self.opts.hallucination_silence_threshold,
                multilingual=True,
                vad_filter=False,
                clip_timestamps=ts,
            )
            block_segs: list[TSegment] = []
            for s in segments:
                if cancel:
                    cancel()
                seg = self._convert(s, offset)
                if seg is not None:
                    block_segs.append(seg)
                if progress:
                    pos = blk.start + int(s.end * ASR_SR)
                    covered = done + sum(min(b, pos) - a for a, b in blk.clips if a < pos)
                    progress(min(1.0, covered / total_speech))
            done += sum(b - a for a, b in blk.clips)
            if progress:
                progress(min(1.0, done / total_speech))
            block_segs = self._clean(block_segs, probs)
            if probs is not None:
                block_segs = self._repair_coverage(block_segs, blk, seg_audio, probs, cancel)
            out.extend(block_segs)
            tail = " ".join(s.text for s in block_segs if not s.uncertain)[-PROMPT_CHARS:]
            prompt = tail if tail.strip() else None
        out = self._dedupe_across(out)
        if probs is not None:
            out = self._mark_unclear(out, regions, probs)
        out.sort(key=lambda s: (s.start, s.end))
        self.stats["segments"] = len(out)
        self.stats["uncertain_segments"] = sum(1 for s in out if s.uncertain)
        return out

    # ------------------------------------------------------------------ helpers
    def _convert(self, s, offset: float) -> TSegment | None:
        words = []
        for w in (s.words or []):
            if not w.word.strip():
                continue
            words.append(TWord(start=round(offset + w.start, 3), end=round(offset + max(w.end, w.start), 3),
                               text=w.word, prob=float(w.probability)))
        text = s.text.strip()
        if not text:
            return None
        start = words[0].start if words else round(offset + s.start, 3)
        end = words[-1].end if words else round(offset + s.end, 3)
        if end < start:
            end = start
        uncertain = segment_is_uncertain(s.avg_logprob, [w.prob for w in words], s.temperature, s.compression_ratio)
        return TSegment(start=start, end=end, text=text, words=words, avg_logprob=float(s.avg_logprob),
                        no_speech_prob=float(s.no_speech_prob), compression_ratio=float(s.compression_ratio),
                        temperature=float(s.temperature), language=self.policy.current or "", uncertain=uncertain)

    def _uncovered_runs(self, segs: list[TSegment], a: int, b: int, probs: np.ndarray) -> list[tuple[int, int]]:
        """Runs (16 kHz samples) of confident speech inside [a, b) that no recognised word covers."""
        wa, wb = a // 512, max(a // 512 + 1, b // 512)
        p = probs[wa:wb]
        if p.size == 0:
            return []
        covered = np.zeros(p.size, dtype=bool)
        tol = 0.3
        for s in segs:
            spans = [(w.start, w.end) for w in s.words] or [(s.start, s.end)]
            for ws, we in spans:
                i0 = max(0, int((ws - tol) * ASR_SR / 512) - wa)
                i1 = min(p.size, int(math.ceil((we + tol) * ASR_SR / 512)) - wa)
                if i1 > i0:
                    covered[i0:i1] = True
        speech = (_smooth_probs(p) >= 0.35) & ~covered
        runs, start = [], None
        for i in range(p.size + 1):
            on = i < p.size and speech[i]
            if on and start is None:
                start = i
            elif not on and start is not None:
                # bridge tiny gaps of < 0.3 s
                if runs and (start - runs[-1][1]) * 512 < 0.3 * ASR_SR:
                    runs[-1] = (runs[-1][0], i)
                else:
                    runs.append((start, i))
                start = None
        out = []
        for r0, r1 in runs:
            if (r1 - r0) * 512 >= 1.0 * ASR_SR and float(p[r0:r1].mean()) >= 0.4:
                out.append(((wa + r0) * 512, (wa + r1) * 512))
        return out

    def _repair_coverage(self, segs: list[TSegment], blk: Block, blk_audio: np.ndarray, probs: np.ndarray,
                         cancel: Callable[[], None] | None) -> list[TSegment]:
        """Re-transcribe confident speech that produced no words (e.g. a language change inside a
        30-second window, or a decoder that stopped early). Each gap is decoded on its own, with
        fresh language detection and no prompt. Words are only accepted inside the gap."""
        budget = max(30.0, 0.25 * sum(b - a for a, b in blk.clips) / ASR_SR)
        added: list[TSegment] = []
        for ca, cb in blk.clips:
            for ra, rb in self._uncovered_runs(segs, ca, cb, probs):
                if cancel:
                    cancel()
                if self.stats["repair_seconds"] >= budget:
                    return segs + added
                pad = int(0.3 * ASR_SR)
                sa, sb = max(ca, ra - pad), min(cb, rb + pad)
                sub = blk_audio[sa - blk.start: sb - blk.start].astype(np.float32, copy=True)
                if sub.size < ASR_SR // 2:
                    continue
                self.stats["repair_attempts"] += 1
                self.stats["repair_seconds"] += (sb - sa) / ASR_SR
                existing = [(w.start, w.end) for sg in segs + added for w in sg.words]
                saved = self.policy.current
                self.policy.current = None          # fresh language decision for this piece
                try:
                    res, _ = self.model.transcribe(
                        sub, language=saved or self.policy.default, task="transcribe", beam_size=self.opts.beam_size,
                        best_of=5, temperature=[0.0, 0.2, 0.4, 0.6], compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0, no_speech_threshold=0.6, condition_on_previous_text=False,
                        initial_prompt=None, word_timestamps=True, multilingual=True, vad_filter=False)
                    got = []
                    for s in res:
                        seg = self._convert(s, sa / ASR_SR)
                        if seg is None:
                            continue
                        seg.words = [w for w in seg.words if ra / ASR_SR - 0.6 <= w.start <= rb / ASR_SR + 0.3
                                     and not any(w.start < e - 0.05 and w.end > b + 0.05 for b, e in existing)]
                        if not seg.words or seg.no_speech_prob > 0.6 or seg.avg_logprob < -1.2:
                            continue
                        seg.text = "".join(w.text for w in seg.words).strip()
                        seg.start, seg.end = seg.words[0].start, seg.words[-1].end
                        got.append(seg)
                    got = self._clean(got, probs)
                finally:
                    lang_used = self.policy.current
                    self.policy.current = saved
                for g in got:
                    g.language = lang_used or g.language
                self.stats["repaired_segments"] += len(got)
                added.extend(got)
        if added:
            log.info("Coverage repair recovered %d segment(s) in block %.1f-%.1f s", len(added),
                     blk.start / ASR_SR, blk.end / ASR_SR)
        return sorted(segs + added, key=lambda s: s.start)

    def _speech_support(self, seg: TSegment, probs: np.ndarray | None) -> float:
        if probs is None or probs.size == 0:
            return 1.0
        a = int(seg.start * ASR_SR / 512)
        b = max(a + 1, int(math.ceil(seg.end * ASR_SR / 512)))
        p = probs[a:b]
        return float(p.mean()) if p.size else 0.0

    def _clean(self, segs: list[TSegment], probs: np.ndarray | None) -> list[TSegment]:
        cleaned: list[TSegment] = []
        for seg in segs:
            if is_known_artefact(seg.text):
                weak = seg.no_speech_prob > 0.2 or seg.avg_logprob < -0.6 or self._speech_support(seg, probs) < 0.5
                if weak:
                    self.stats["dropped_artefacts"] += 1
                    continue
                seg.uncertain = True
            if seg.words:
                rep = collapse_repetition([w.text for w in seg.words])
                if rep:
                    a, b = rep
                    seg.words = seg.words[:a] + seg.words[b:]
                    seg.text = "".join(w.text for w in seg.words).strip()
                    seg.end = seg.words[-1].end if seg.words else seg.end
                    seg.uncertain = True
                    seg.note = "repetition removed"
                    self.stats["repetitions_collapsed"] += 1
            if seg.words and len(seg.words) >= 10:
                rep2 = find_immediate_repeat([w.text for w in seg.words])
                if rep2:
                    a, b = rep2
                    seg.uncertain = True
                    if chars_per_second(seg.text, seg.start, seg.end) > MAX_PLAUSIBLE_CPS:
                        # too much text for the time it took: the second copy was not spoken
                        seg.words = seg.words[:a] + seg.words[b:]
                        seg.text = "".join(w.text for w in seg.words).strip()
                        seg.end = seg.words[-1].end if seg.words else seg.end
                        seg.note = "duplicated phrase removed"
                        self.stats["repetitions_collapsed"] += 1
            if seg.text:
                cleaned.append(seg)
        return cleaned

    def _dedupe_across(self, segs: list[TSegment]) -> list[TSegment]:
        """Drop decoding loops across windows.

        * a segment of >= 4 words that exactly repeats the previous two;
        * a run of >= 3 consecutive *low-confidence* segments with the same short text
          (e.g. "Thank you." over and over in noise): only the first is kept, flagged.
        Confident repetitions of short words ("ja", "okay") are real speech and are kept.
        """
        out: list[TSegment] = []
        for seg in segs:
            n = normalise(seg.text)
            if len(n.split()) >= 4 and len(out) >= 2 and normalise(out[-1].text) == n and normalise(out[-2].text) == n:
                self.stats["duplicate_segments_dropped"] += 1
                out[-1].uncertain = True
                continue
            out.append(seg)
        res: list[TSegment] = []
        i = 0
        while i < len(out):
            j = i
            n = normalise(out[i].text)
            while j + 1 < len(out) and normalise(out[j + 1].text) == n:
                j += 1
            run = out[i:j + 1]
            if len(run) >= 3 and not run[0].marker and all(r.uncertain for r in run):
                run[0].note = "repeated low-confidence text removed"
                res.append(run[0])
                self.stats["duplicate_segments_dropped"] += len(run) - 1
            else:
                res.extend(run)
            i = j + 1
        return res

    def _mark_unclear(self, segs: list[TSegment], regions: list[tuple[int, int]], probs: np.ndarray) -> list[TSegment]:
        """Insert '[unclear speech]' where VAD is confident about speech but nothing was transcribed."""
        if not regions:
            return segs
        covered = np.zeros(probs.size, dtype=bool)
        for s in segs:
            covered[int(s.start * ASR_SR / 512): int(math.ceil(s.end * ASR_SR / 512)) + 1] = True
        markers = []
        for a, b in regions:
            wa, wb = a // 512, max(a // 512 + 1, b // 512)
            p = _smooth_probs(probs[wa:wb])
            free = ~covered[wa:wb]
            # find runs of uncovered, confidently-speech windows of >= 1.5 s
            run_start = None
            for i in range(len(p) + 1):
                ok = i < len(p) and free[i] and p[i] >= 0.5
                if ok and run_start is None:
                    run_start = i
                elif not ok and run_start is not None:
                    if (i - run_start) * 512 / ASR_SR >= 1.5:
                        s_t = (wa + run_start) * 512 / ASR_SR
                        e_t = (wa + i) * 512 / ASR_SR
                        markers.append(TSegment(start=round(s_t, 3), end=round(e_t, 3), text="[unclear speech]",
                                                words=[], uncertain=True, marker=True))
                    run_start = None
        self.stats["unclear_markers"] = len(markers)
        return segs + markers
