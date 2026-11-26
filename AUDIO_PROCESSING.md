# AudioRefine – Audio and speech processing

This document describes exactly what AudioRefine 1.0 does to the audio, with every filter
frequency and parameter, and why. It is written so that the approach can be reviewed
independently. Measurements referred to here are in [TEST_REPORT.md](TEST_REPORT.md); the code
is in `src/audiorefine/` and `native/arfilter/`.

**Guiding principle:** the output must remain recognisably the same conversation, only easier
to understand. Nothing is synthesised, nothing is gated to silence, no pause is removed, the
timeline is preserved sample-for-sample. When in doubt, the gentler option was chosen.

---

## 1. Two paths from one timeline

```
originals ──decode──► 48 kHz mono "master timeline" (float32, back-to-back, no gaps)
                         │
                         ├─► LISTENING PATH (48 kHz)
                         │     per recording: pre-gain → high-pass → [hum notch] → DeepFilterNet3
                         │     (attenuation-limited) → presence EQ → leveller → peak limiter
                         │     → junction fades → stitched → exact 10-min FLAC segments
                         │
                         └─► RECOGNITION PATH (16 kHz, exact 3:1 decimation of the same samples)
                               Silero VAD → clips → per-clip level normalisation → Whisper large-v3
                               → hallucination checks → coverage repair → transcripts
```

The enhanced audio that sounds best is not the audio that transcribes best. We measured this
explicitly (TEST_REPORT.md §4.2, Whisper large-v3, 12 conditions, Danish and English): pooled
word error rate was **10.3 %** for the original signal with level normalisation (the path
AudioRefine uses), 11.2 % for the original signal, 11.1 % for conservatively filtered audio,
**14.7 %** for the listening-path audio with bounded DeepFilterNet and 17.6 % with unbounded
DeepFilterNet. In babble noise, neural noise reduction more than doubled the errors (Danish
21.7 % → 40.1 %, English 32.2 % → 66.7 %); in the other conditions the differences were within
±2.4 points (at most 4 words per excerpt) in both directions. Meanwhile DeepFilterNet improved
the intelligibility estimate (ESTOI) for listeners in noisy conditions. AudioRefine therefore
uses two internal paths. The user sees neither.

Both paths share one timeline: 16 kHz sample *i* is 48 kHz sample *3i*. All timestamps are
derived from sample counts, so audio files and transcripts cannot drift apart.

---

## 2. Decoding and stitching (`decode.py`, `pipeline.py`)

* Decoder: FFmpeg libraries via PyAV. The original file is opened read-only.
* Resampling: FFmpeg swresample to 48 kHz float (default filter, 32-tap polyphase), per file.
* Channels:
  * mono → as is;
  * stereo → average of L and R, **except** when a whole-file analysis shows that the channels
    are phase-inverted (energy of the average < 25 % of the mean channel energy and correlation
    < −0.3; averaging would cancel the speech → `(L−R)/2`), or that one channel is > 30 dB weaker
    (dead/unconnected input → the live channel only);
  * more channels → average.
* Damaged packets are skipped and counted; a truncated file contributes its readable part;
  both are reported. A file that yields no audio at all is reported as unreadable and left
  untouched.
* Container timestamp gaps are ignored: decoded samples are appended back-to-back. **No silence
  is inserted between recordings** and nothing is removed.
* Each recording's start/end sample on the master timeline is recorded in
  `_AudioRefine_details.json` and printed in `Transcript_Complete.txt`.

The 16 kHz recognition signal is produced in the same pass by a 241-tap Kaiser-windowed
(β = 8) linear-phase low-pass (cut-off 7.6 kHz, > 57 dB alias rejection measured at 9 kHz) with
its 120-sample delay compensated, keeping exactly `ceil(N/3)` samples.

---

## 3. Voice activity detection (`vad.py`)

* **Silero VAD v6** (MIT; the ONNX model shipped with faster-whisper), 512-sample windows
  (32 ms) at 16 kHz, run over the whole recording in a streaming fashion (LSTM state carried
  across blocks, so arbitrarily long recordings need constant memory).
* Hysteresis: speech starts at p ≥ **0.35**, ends after p < **0.20** for ≥ **1.0 s**; regions
  shorter than **90 ms** are discarded; regions are padded by **0.4 s** on both sides (gaps
  shorter than 0.8 s are split in the middle).
* The threshold is lower than Silero's default 0.5 on purpose: missing quiet speech is worse
  than decoding a little non-speech, which the hallucination checks handle.
* **Level-normalised detection input.** Silero is not fully level-invariant: in the Windows
  end-to-end test an utterance at ≈ −65 dBFS (a speaker far from the microphone) was only half
  detected, and Whisper therefore never saw its first sentence. The VAD now analyses a copy of
  the signal with a slow gain: the 1-second peak envelope of 30 ms frame levels is brought
  towards −20 dBFS, gain 0…+40 dB, smoothed over ~0.3 s — **but only where that envelope lies at
  least 12–18 dB above the local noise floor** (moving minimum over ±5 s). The first version
  without this contrast condition boosted a 10-minute stretch of plain room noise in the CI test
  by ~34 dB; Silero then called it speech and Whisper hallucinated "Thank you." 25 times. With
  the condition, the quiet utterance is detected (0.92 mean probability, was 0.0–0.7), the
  noise stretch yields no speech regions, and recall/false-alarm rates on the corpus are
  unchanged. Nothing in the audio outputs is affected by this gain.
* VAD is used for: (1) deciding what Whisper decodes, (2) the leveller's speech level
  estimate, (3) per-recording speech/noise statistics, (4) detecting speech that produced no
  words (coverage repair and `[unclear speech]` markers).
* VAD **never removes audio** from the listening path.

---

## 4. Per-recording analysis and the adaptive plan (`analysis.py`)

Every recording is analysed on its own, because recordings may come from different devices:

| Measurement | How |
|---|---|
| Speech level | mean power of 10 ms frames where VAD p ≥ 0.5 |
| Noise level | 20th percentile of frame levels outside speech (floored at −100 dBFS) |
| SNR | speech level − noise level |
| Low-frequency noise | Welch PSD (65 536-point Hann, 50 % overlap, up to 120 blocks of pauses): mean power 20–80 Hz relative to 200–2000 Hz, and its absolute level |
| Mains hum | 50 and 60 Hz families, harmonics 1–10: peak (±2 bins) vs median of neighbours 4–12 Hz away; a harmonic is "present" at ≥ 12 dB prominence; hum = ≥ 2 present harmonics, or the fundamental at ≥ 20 dB |
| Peak, clipping, bandwidth | logged for diagnostics |

The analysis only selects among a few **bounded** settings (no open-ended adaptation):

| Decision | Rule |
|---|---|
| Pre-gain (DeepFilterNet input only) | bring speech to −26 dBFS, limited to −20…+45 dB |
| High-pass | default **70 Hz, 2nd-order Butterworth (12 dB/oct)**. If pauses contain strong rumble/wind/handling noise (20–80 Hz band > 10 dB above the 200–2000 Hz band *and* within 35 dB of the speech level): **80 Hz, 4th-order (24 dB/oct)** |
| Hum notches | only if hum detected: notches at the fundamental and 2nd harmonic (50/100 Hz or 60/120 Hz), Q = 30 (≈ 1.7–4 Hz wide) |
| DeepFilterNet attenuation limit | SNR ≥ 40 dB → **6 dB**; 25–40 → **10 dB**; 12–25 → **14 dB**; < 12 → **18 dB** |
| Presence EQ | always: peaking +2 dB at 3 kHz, Q = 0.9 |

All decisions and the measurements behind them are logged per recording.

---

## 5. Listening path in detail

### 5.1 Frequency choices

Speech intelligibility is carried mostly by 500 Hz – 4 kHz (ANSI S3.5 Speech Intelligibility
Index band-importance peaks around 1.5–3.5 kHz), but consonant cues (/s/, /f/, /t/, /k/) extend
to 8 kHz and beyond, and voice fundamentals of low male voices go down to ~75–85 Hz.
Therefore:

* **No low-pass filter at all.** Telephone-band restrictions (300–3400 Hz) were explicitly
  rejected; the full source bandwidth up to 24 kHz is kept. Broadband hiss is handled by the
  noise suppressor, not by cutting high frequencies that carry consonants.
* **High-pass at 70 Hz, 12 dB/oct**: removes rumble, traffic and handling thumps below the
  speech range while keeping fundamentals (−3 dB at 70 Hz, −0.3 dB at 140 Hz). The steeper 80 Hz
  variant is used only for recordings with measured strong low-frequency noise.
* **Hum notches only at 50/100 (or 60/120) Hz.** We first notched up to 10 harmonics; measurement
  showed that notching harmonics inside the speech band damages speech (ESTOI on *clean*
  speech dropped from 0.996 to 0.908 with 9 notches, Q 30), whereas DeepFilterNet removes the
  remaining harmonics very well (ESTOI 0.973 for the hum+rumble condition). See TEST_REPORT.md.
* **Presence +2 dB at 3 kHz (Q 0.9, ≈ 1.9–4.6 kHz)**: a mild lift of the region with the
  highest intelligibility importance, compensating typical muffling of distant or pocketed
  microphones. Deliberately small so voices do not become harsh.

All IIR filters are RBJ-cookbook biquads in 64-bit precision (transposed direct form II).

### 5.2 Noise reduction: DeepFilterNet3 with an attenuation limit

* Model: **DeepFilterNet3** (Schröter et al., 2023), 48 kHz, 20 ms window / 10 ms hop, ERB-gain
  stage plus deep-filtering stage for the lower spectrum; run by its Rust/tract implementation
  (libDF 0.5.6) inside `arfilter`. Default thresholds (−15/35/35 dB local SNR), no post-filter.
* **Attenuation limit** (§4): the network output is mixed in the spectral domain with the
  unprocessed spectrum, `Y = (1−λ)·Ŝ + λ·X` with `λ = 10^(−lim/20)`. Consequences:
  * noise is reduced by at most 6–18 dB, so the background stays natural and
    "breathing"/pumping artefacts are avoided;
  * anything the network mistakes for noise — quiet, distant or whispered speech — is
    still present at **at least −lim dB**, never removed. This directly implements
    "do not remove quiet speech because it resembles background noise";
  * no frame is ever set to digital silence.
* Why limited instead of full suppression: on our babble condition the limited setting gave a
  *higher* intelligibility estimate than unlimited suppression (ESTOI 0.690 vs 0.671; raw 0.628),
  and it introduces fewer artefacts on clean recordings (TEST_REPORT.md).
* Pre-gain: libDF zeroes frames whose mean square is below 1e-7 (≈ −70 dBFS) and skips its
  state update. Quiet recordings are therefore brought to a normal level (−26 dBFS speech)
  before the network. Frames that are still that quiet (digital silence) receive a tiny
  deterministic noise floor (≈ −68 dBFS) so the network keeps running normally; it is then
  attenuated like any other noise.
* Delay: the STFT + look-ahead latency (1440 samples = 30 ms) is compensated exactly; output
  length equals input length (verified by tests for lengths 0…, and by cross-correlation).
* Parallelism: recordings are processed in 300 s chunks on all CPU cores; each chunk is preceded
  by 4 s of discarded warm-up audio and joined with a 20 ms raised-cosine crossfade. Measured
  difference to single-pass processing: < −26 dB.
* Reverberation: DeepFilterNet3 reduces late reverberation somewhat; no separate
  dereverberation is applied (see §9).

### 5.3 Leveller (automatic gain control) and limiter (`enhance.py`)

Goal: quiet speakers and people far from the microphone become clearly audible, loud speakers
are tamed, sudden loud noises do not hurt — without pumping, without amplifying pauses beyond
the level of the surrounding speech, and without touching the timeline.

* Analysis on 10 ms frames of the DeepFilterNet output; VAD speech probability ≥ 0.5 marks
  speech frames.
* **Transient-robust level estimate**: for level estimation only, frame energies are clipped to
  12 dB above the local median speech level (0.5 s blocks), so a door slam does not make the
  leveller turn the following speech down.
* Local speech level = Gaussian-weighted (σ = **0.75 s**) power average over speech frames;
  where there is little speech nearby, a wider window (σ = 4 s) and finally the recording's
  global speech level are used. Gains in pauses are thus *held* from the adjacent speech
  (no noise pumping, no gating).
* Target: **−20 dBFS** speech level. Gain = target − local level, limited to
  **[reference − 10 dB, reference + 18 dB]**, where reference is the gain that brings the
  recording's overall speech level to the target, and to an absolute **−25 … +45 dB**.
  A speaker 20 dB quieter than the others is therefore raised to within a few dB of them
  (tested), but background chatter in long pauses cannot be pushed up without bound.
* Gain curve smoothed (σ = 0.15 s) – slow enough to keep syllable dynamics (over-compression
  harms intelligibility), fast enough to follow turn-taking.
* **Look-ahead peak limiter**: per frame, the gain is limited so that the peak in a ±20 ms
  neighbourhood stays below **−1 dBFS**; this limit is spread with a ±30 ms minimum filter and
  light smoothing (fast attack, ~50 ms release), then re-applied. Because the gain is
  interpolated between frame centres whose limits both include the peak frame, no sample can
  exceed the ceiling; a final safety clip at ±1.0 is counted (expected 0) and logged.
* Effect on a sudden bang: its peak is brought to −1 dBFS while speech is normalised around
  −20 dBFS, and speech before/after is unaffected beyond ~50 ms.

### 5.4 Joining and segmentation (`pipeline.py`, `segments.py`)

* Recordings are joined back-to-back. At each junction a **5 ms raised-cosine fade-out/fade-in**
  prevents a click from the waveform discontinuity. Samples are attenuated over 5 ms, none are
  removed or inserted.
* Segments: exactly **28 800 000 samples (10:00.000)** each; the last one may be shorter. Sample
  counts are verified after writing by decoding every FLAC file again and comparing a SHA-256 of
  its PCM data with what was written.
* Format: FLAC, 48 kHz, 16-bit, mono, TPDF dither (±1 LSB triangular, fixed seed).

---

## 6. Recognition path in detail (`transcriber.py`)

### 6.1 Model and decoding parameters

Whisper **large-v3** via faster-whisper/CTranslate2; CPU int8, GPU float16 (int8_float16 below
6 GB VRAM). Decoding: beam size 5, best-of 5, temperature fallback 0 → 0.2 → … → 1.0,
compression-ratio threshold 2.4, log-prob threshold −1.0, no-speech threshold 0.6,
conditioning on previous text with prompt reset when the fallback temperature reaches 0.5,
word timestamps on, `hallucination_silence_threshold` 2 s, task always *transcribe* (never
*translate*).

### 6.2 What is decoded: clips and blocks

1. VAD regions are merged into **clips** of at most **30 s** span, merging only across pauses of
   at most **4 s**, and **never across the junction between two recordings** (a new recording
   often means a new situation, speaker or language). Very long continuous regions (> 10 min,
   e.g. a TV) are cut at the quietest 200 ms within ±15 s of the 10-minute mark.
2. Clips are grouped into **blocks** of up to 20 minutes whose boundaries always lie in a pause.
   Each block is decoded by faster-whisper with `clip_timestamps`, so only speech is decoded but
   timestamps stay on the original timeline (no audio is cut out or shifted).
3. **Context across boundaries:** within a block Whisper conditions on the previously decoded
   text; each new block starts with the last ~200 characters of the previous block's confident
   text as prompt. The 10-minute audio segmentation plays no role here: a sentence spoken from
   09:58 to 10:03 is decoded in one piece and only split for the per-file transcripts
   afterwards (§7).

### 6.3 Level normalisation for recognition

Each clip is scaled so that the 95th percentile of its 30 ms frame levels is at **−14 dBFS**
(gain limited to −12…+40 dB, and so that at most 0.1 % of samples would exceed full scale; the
rest is clipped – irrelevant for recognition). Whisper's log-Mel normalisation is not fully
level-invariant, so very quiet speech otherwise reaches the model at an unusual level.

### 6.4 Language

* Language identification runs for **every 30-second window** on the same encoder output that
  is decoded (no extra cost), and the result passes through a small policy
  (`language.py`):
  * Norwegian (no/nn) and Swedish probability mass is attributed to **Danish** – Whisper often
    mistakes Danish for them, and decoding Danish with a Norwegian token produces Norwegian
    spelling;
  * switching between Danish and English requires a score ≥ **0.6** and higher than the current
    language (hysteresis against unstable short windows);
  * another language is accepted only at ≥ **0.9** confidence while Danish and English are both
    ≤ 0.05 – the spoken language is always preserved, never translated.
* Mixed Danish/English recordings: the language follows the speech window by window; English
  words inside Danish sentences are usually transcribed as spoken by Whisper.

### 6.5 Hallucination and integrity safeguards

Whisper can invent text, especially on silence, noise and music. Measures, in order:

1. **VAD gating** – silence and pure noise are not decoded at all.
2. **Whisper's own checks** – temperature fallback on compression ratio > 2.4 or average log-prob
   < −1.0; a window is treated as silence only when no-speech probability > 0.6 *and* log-prob
   < −1.0 (so quiet but real speech with high no-speech probability is kept);
   `hallucination_silence_threshold` skips long silent stretches inside a window when the
   output looks like a hallucination.
3. **Known artefacts** – subtitle credits and outros that Whisper learned from video
   ("Tak fordi du så med", "Tak fordi du lyttede med", "Danske tekster af … Scandinavian Text
   Service", "Undertekster af …", "Thanks for watching", "Thank you.", "Subtitles by …",
   "[Musik]" …; credit lines match as a prefix whatever names follow) are **dropped only if the
   acoustic evidence for speech is weak** (no-speech probability > 0.2, log-prob < −0.6 or VAD
   support < 0.5); otherwise kept and flagged. A genuine, clearly spoken "thank you" is kept.
4. **Repetition loops** – a phrase of 2–8 words repeated more than 3 times in a row (single
   words: 8+ times) is cut back to 3 occurrences (4 for single words) and flagged; a segment
   identical to the two previous ones (≥ 4 words) is dropped, and a run of ≥ 3 consecutive
   *low-confidence* segments with identical text keeps only the first (confident repetitions
   such as "ja … ja … ja" are kept). A phrase of ≥ 5 words immediately
   followed by an identical copy is flagged `[uncertain]`; the copy is removed only if the
   segment also contains implausibly much text for its duration (> 21 characters per second;
   normal speech is 12–18). This catches Whisper emitting a sentence twice (observed once on the
   Windows CI machine, 28 characters/s) without deleting genuine repetitions.
5. **Coverage repair** – confident speech (smoothed VAD ≥ 0.35 for ≥ 1 s, mean ≥ 0.4) that
   received no words is decoded again on its own, with fresh language detection and no prompt;
   words are only accepted inside the gap and not overlapping existing words. This recovers
   speech dropped by a language change or an early-stopping decoder (it recovered a complete
   English sentence in our tests). Budget: at most 25 % extra decoding per block.
6. **Markers instead of guesses** – where speech was detected (smoothed VAD ≥ 0.5 for ≥ 1.5 s)
   but nothing could be recognised, `[unclear speech]` is written. Segments with average
   log-prob < −0.9, heavy temperature fallback (≥ 0.6), compression ratio > 2.4, mean word
   probability < 0.55 or > 35 % of words below 0.35 are marked **`[uncertain]`** (SRT: `(?)`).
   Low confidence alone never deletes text, and confidence is not presented as proof of
   correctness.

No language model rewrites, summarises or completes the text; only Whisper's own punctuation
and capitalisation are kept.

---

## 7. Timestamps

* Master timeline: 48 kHz sample index of the joined recording. Seconds = samples / 48 000.
* Whisper segment and word times are relative to the decoded block and offset by the block's
  start sample → absolute timeline. Segment start/end are taken from the first/last word.
  Word times come from cross-attention alignment (DTW) and are typically accurate to
  ~0.1–0.3 s.
* `.txt` and `Transcript_Complete.txt`: absolute timeline, `HH:MM:SS.mmm`.
* `.srt`: relative to the start of that audio file (`HH:MM:SS,mmm`), cues ≤ 2 lines of ~42
  characters and ≤ 7 s, split at word boundaries into evenly sized cues, no overlaps, minimum
  display time 0.3 s.
* Segments crossing a 10-minute boundary are split at the word level: a word belongs to the file
  in which it *starts*. Pieces are marked `…`; no word is dropped or duplicated (tested at
  09:58–10:03, at exactly 10:00.000, and before/after).

---

## 8. Known weaknesses

* **Overlapping speakers** are not separated; Whisper transcribes the dominant voice and may
  merge or drop the other. Speaker labels are out of scope.
* **Babble / TV / music in the background** remain partly audible (by design of the attenuation
  limit) and can be transcribed if they contain intelligible speech. Babble at 5 dB SNR raised
  WER to 19–29 % in our tests.
* **Strong reverberation**: only partially reduced; no dereverberation stage.
* **Clipped (distorted) recordings** are detected and logged but not repaired.
* **Danish dialects, slang and names**: Whisper's Danish is good on clear speech (≈ 7 % WER on
  read speech) but degrades on dialect and fast conversational speech; no Danish fine-tune is
  used yet.
* **Language switching inside one sentence** may be transcribed in the window's main language.
* The **CPU is slow for long recordings** with large-v3 (≈ 0.8× the speech duration on a 4-core
  VM); GPU acceleration is implemented but not yet verified on real NVIDIA hardware.
* Word timestamps can be off by a few hundred milliseconds, which affects which file a word at
  a 10-minute boundary is assigned to (never whether it appears).
* No human listening test has been performed; listening-quality claims rest on objective
  measures (ESTOI) and on the conservative design.

## 9. Opportunities for future improvement

1. **Danish-specialised recognition**: evaluate Danish fine-tunes of Whisper large-v3 (e.g.
   "hviske" models trained on the CoRal corpus) on conversational Danish, verifying timestamp
   quality and English performance; possibly route per window by detected language.
2. **Conversational Danish evaluation set** with real recordings (the current corpus is read
   speech with synthetic degradations).
3. **Dereverberation** (e.g. WPE) for far-field recordings, if listening tests show benefit.
4. **Stereo/binaural output** for true stereo sources (spatial cues help listeners separate
   voices).
5. **Declipping** for overloaded recordings.
6. **AMD/Intel GPU acceleration** via whisper.cpp + Vulkan.
7. **Speaker diarisation** (explicitly out of scope for 1.0).
8. **Per-recording language hints** (e.g. a whole recording detected as English) to bias the
   window-level decisions.
9. Human listening tests (MUSHRA-style) to tune the attenuation limits and presence EQ.
