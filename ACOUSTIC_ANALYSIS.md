# AudioRefine – Acoustic event detection (Analyze sounds, Sound Explorer)

Technical hand-over: what was investigated, what was chosen and why, how it works, how well it
works (measured), and what it cannot do. Written so that the decisions can be reviewed
independently. All figures were measured in this project unless a source is given.

## 1. Feasibility conclusion

**Feasible for a useful subset of sounds, with clear limits.** A compact AudioSet-trained
classifier running locally on the CPU recognises many everyday sounds well enough to be useful
for browsing large collections (e.g. knocking, crying, laughter, snoring, typing, frying, a
dog, a cat, music, alarms, sirens), at roughly 30–60 × real time on an ordinary CPU. Many other
requested sounds (shouting, whimpering, sighs, door slams, footsteps, dishes, cutlery, vacuum
cleaner, microwave, washing machine, dishwasher, kettle …) are either not separable from
everyday speech/noise at an acceptable false-alarm rate, or have too few reliable examples, and
are therefore **disabled** rather than shown with misleading results (§6). Activity inference is
limited to two conservative rules (possible cooking, possible washing up). Timing is accurate to
about ±1 s for short sounds and ±5 s for music/TV.

The concept is therefore delivered as *direct sound event detection with honest confidence*,
plus playback-based review – not as an activity or behaviour recogniser.


## 2. Technologies investigated

| Candidate | What it is | Licence | Size | Investigated how |
|---|---|---|---|---|
| **EfficientAT** `mn10_as`, `mn40_as_ext`, `dymn10_as`, `dymn20_as` (Schmid, Koutini, Widmer 2023) | MobileNetV3 / dynamic MobileNets distilled from transformer ensembles, AudioSet 527 classes; published single-model mAP 0.471 / 0.487 / 0.478 / 0.493 | MIT | 4.9 M – 68 M parameters (19.5 MB – 274 MB) | measured on AudioSet eval (§7.1), mn10 exported to ONNX and used |
| **PANNs** CNN14 and CNN14-DecisionLevelMax (Kong et al. 2020) | CNNs on AudioSet; the DecisionLevelMax variant gives frame-wise outputs for event detection; mAP 0.431 / 0.385 | MIT (code), weights on Zenodo | 80 M parameters, 327 MB | measured (§7.1) |
| **Audio Spectrogram Transformer** AST (Gong et al. 2021) | ViT on 10.24 s spectrograms; mAP 0.459 | BSD-3 | 87 M parameters, 345 MB | measured (§7.1) |
| **BEATs** (Chen et al. 2022) | self-supervised transformer, AudioSet mAP ≈ 0.48–0.50 | MIT | 90 M parameters | **not obtainable**: the official checkpoints are hosted on an Azure blob that refuses access (HTTP 409) and no official mirror exists |
| **CLAP** (LAION `clap-htsat-unfused`, `larger_clap_general`) | audio–text embedding; zero-shot classification by text prompts | CC0 / Apache-2.0 (weights) | 590–740 MB | downloaded; considered as a second-stage validator (§3) |
| **YAMNet** (Google) | MobileNetV1 on AudioSet, 521 classes, mAP 0.306 | Apache-2.0 | 3.7 M parameters | the original TF-Hub URL is gone (moved to Kaggle); the only ONNX export found (Qualcomm AI Hub) gave implausible outputs with the documented front-end (e.g. "Silence" for a barking dog) – rejected on published accuracy rather than spending more time |
| DCASE domestic SED baselines (DESED, 10 classes) | CRNN / BEATs-based event detectors for 10 domestic classes | various | – | not evaluated: only 10 classes, all also covered by the AudioSet models |
| ONNX Runtime | inference engine | MIT | already bundled (Silero VAD) | used for the selected model |

All candidates run offline after download. GPU: all are PyTorch/ONNX models that could use a GPU,
but CPU speed of the selected model is sufficient (§8), so GPU support was not added.

## 3. Selected technology and rationale

**EfficientAT `mn10_as`, exported to ONNX (19.5 MB, bundled), run with ONNX Runtime on the CPU,
in a two-scale scheme (§5).**

* Accuracy per CPU cost: on the AudioSet evaluation clips mn10 is within a few points of the
  larger EfficientAT models, CNN14 and AST (§7.1), at 1/10 – 1/20 of their compute.
* Deployment: a single 19.5 MB ONNX file; the application already ships ONNX Runtime, so the
  package grows by only ~20 MB and needs no PyTorch (≈ 200 MB+), no download and no internet.
* Licence: MIT (code and weights); the AudioSet label list is CC BY 4.0.
* The export was verified against PyTorch (max. difference 7·10⁻⁷) and the numpy front-end
  against the PyTorch front-end (max. difference 4·10⁻⁵).

A two-stage design "cheap detector → large validator" (e.g. mn10 → AST or CLAP) was considered.
It was not adopted: (1) the larger supervised models are only a few AP points better (§7.1) but
10–40 × slower; (2) CLAP zero-shot accuracy for AudioSet-style categories is clearly below the
supervised models in the literature; (3) a second model would add 350–750 MB and a second
failure mode. The *two-scale* use of one model (§5) gave a larger gain than a bigger model would.

## 4. Audio input and preprocessing

* **Original audio only.** The analysis decodes the original recordings (in the folder and in
  `processed/**`) with FFmpeg (PyAV), read-only. The speech-enhanced audio is not used: it
  deliberately removes non-speech sounds.
* **Sample rate:** resampled to 32 kHz (the model's rate) by FFmpeg's resampler.
* **Channels:** mixed to mono by averaging; phase-inverted stereo (which would cancel) uses the
  channel difference instead. Analysing channels separately was not adopted: typical phone and
  dictaphone recordings are mono or near-identical stereo, and averaging only lowers a one-sided
  sound by 6 dB – within the range covered by the quiet/distant tests (§7.3) – while doubling
  the cost.
* **Level normalisation / filtering:** not applied. The model's log-mel front-end is largely
  level-invariant in the range of real recordings (AudioSet contains every recording level),
  and the quiet and distant conditions in §7.3 did not benefit from extra gain in spot checks;
  additional filtering would change the spectral shape the model was trained on.
* **Front-end:** identical to the model's training front-end: pre-emphasis 0.97, 25 ms Hann
  window every 10 ms, 1024-point FFT, 128 Kaldi mel bands 0–15 kHz, `(log(mel+1e-5)+4.5)/5`;
  computed in a streaming fashion (exactly equal to the whole-signal computation, tested), so
  memory does not grow with recording length.

## 5. Event localisation: two time scales

**Finding:** EfficientAT models saturate on short inputs. For a 10-second clip of a barking dog
the model outputs Animal 0.68 / Dog 0.51, but for 1–2 s excerpts every class goes towards 1.0
(the classifier only behaves when its features are averaged over ≥ ~4 s). Applying the classifier
to individual time steps of the feature map saturates in the same way. Placing a short excerpt
in the middle of 10 s of silence ("padded window") restores sensible outputs.

Padded short windows alone, however, lose much accuracy compared with natural 10-second context
(average precision on AudioSet eval clips: shouting 0.39 vs 0.71, baby crying 0.48 vs 0.72,
blender 0.51 vs 0.76, television 0.11 vs 0.44). The adopted scheme combines both:

* **context** score: natural 10 s windows every 5 s (the model's training condition);
* **localisation** score: 2 s windows every 1 s, each centred in 10 s of silence;
* per 2 s window: **geometric mean** of its own score and the best context window covering it
  (music, television, video game music, radio: context only, they are only recognisable with
  context).

On the AudioSet evaluation clips the combination keeps the context accuracy (mean AP: impacts
0.595 vs 0.603 context-only, sustained 0.749 vs 0.748, vocal 0.707 vs 0.723) while localising to
about ±1 s. Media categories are localised to about ±5 s.

Events are then formed per category (``sound/events.py``): hysteresis (start at the moderate
threshold, continue while ≥ half of it), merging of occurrences closer than a per-kind gap
(impacts 5 s, vocal sounds 8 s, sustained sounds 10 s, media 20 s; repeated knocks become one
event with a hit count, a snoring sleeper one event per snoring period), a minimum amount of evidence (sustained sounds ≥ 3 windows, media ≥ 6),
and High/Moderate confidence (§9). An event covers the middle part of its first to last window.

Hour boundaries: each recording is analysed completely once; each event is listed in the hour in
which it *starts*, with its full span (an event 08:59:30–09:01:10 is listed once, under 08:00).
Recording times, DST handling and the estimated-time rules are those of the transcription
(`rectime`; for recordings already transcribed, the exact start time stored in the hour
manifest is reused, so both outputs agree).

<!-- TABLES -->

## 9. Confidence handling

* Per category two thresholds on the combined score (§6): *moderate* and *high*. Below moderate
  nothing is reported. Thresholds are per category – one universal threshold does not work
  (a knock scores 0.6–0.8 when present, a doorbell 0.25–0.5, music up to 0.95).
* *High* additionally requires, for sustained sounds, at least two windows at the high level,
  and is never given to a detection during music/TV audio.
* The raw scores are kept in the JSON files (`score_peak`, `score_mean`) for diagnostics only;
  they are sigmoid outputs of a classifier, **not** calibrated probabilities, and are not shown
  as such. Users see only High / Moderate.
* Every report and the Explorer state that detections are approximate and may be wrong, and
  that a missing detection does not mean the sound did not occur.

## 10. False-positive mitigation (what is done, in order of effect)

1. **Two-scale agreement** – a sound must be supported by both the 10-second context and the
   2-second window (geometric mean). This alone removed almost all false detections on
   conversational speech compared with short windows (e.g. read speech: from dozens of
   "laughter/whisper/zipper" detections per hour to a handful).
2. **Per-category thresholds** tuned for precision, then raised until event-free speech and
   quiet-room recordings produce at most 0.5 moderate / 0.1 high detections per hour (§6).
3. **Disabled categories** where no threshold reaches the targets (§6.2).
4. **Minimum evidence** and **hysteresis** (§5): sustained sounds need several windows; one
   occurrence is one event.
5. **Merging** of repeated hits (repeated knocks = one event with a count).
6. **Repetition rule**: a sound with > 20 moderate detections per hour in one recording keeps only
   its high-confidence detections there (marked *detected very often*).
7. **Playback context**: inside music/TV stretches, other detections are flagged *during
   music/TV audio* and capped at Moderate. The model cannot tell a laugh on TV from a laugh in
   the room; this flag is the honest alternative to pretending it can.
8. **Overlapping categories** were avoided by design (e.g. baby crying is part of
   *Crying / sobbing*, not a second category), so one sound is not reported twice.
9. **User review**: *Incorrect detection* hides an event from lists and reports; the machine
   label and the mark are both kept. No automatic retraining.

## 11. Activity inference

Implemented as two conservative rules (``sound/activities.py``); everything else was deferred.

| Suggestion | Rule | Why only this |
|---|---|---|
| *Possible cooking* | ≥ 2 different kitchen sounds within 10 minutes, at least one of frying/sizzling, chopping, blender, microwave or boiling | these sounds are rarely produced by anything else in a home |
| *Possible washing up* | running water + dishes or cutlery within 3 minutes | only if those categories are enabled |

Rules use only detections that are not during music/TV and not marked incorrect. An activity
shows its time span, *Moderate* confidence (High only with ≥ 3 different supporting sounds, two of
them High), and the supporting sounds, which can each be played. Showering, cleaning, hair
drying, gaming and TV watching were **not** implemented as activities: their key sounds are
either not reliable (vacuum cleaner, shower-like water) or the "activity" would only rename a
single detected sound (hairdryer → hair drying, TV audio → watching TV) and suggest more than
is known (TV or game audio does not show that anyone was watching or playing).

## 12. Storage, recovery, versions

```
analysis/
  Sound_Events_Master.txt / .json      combined, chronological, regenerated after every hour
  Transcript_Master_Annotated.txt       transcript + annotations (only hours with both)
  <date>/Sound_Events_Day.txt
  <date>/<HH-00>/Sound_Events.json/.txt   per hour; the JSON is the completion record
  _sources/<sha256>.json                analysis of one recording (all its events)
  _AudioRefine_sources.json             checksum cache (path, size, mtime → SHA-256), recording times
  _AudioRefine_reviews.json             confirmed / incorrect marks
```

* Recordings are identified by SHA-256 of their content: moving an original into `processed`
  (as the transcription does) or renaming it does not invalidate anything; playback finds the
  file again by checksum (`store.resolve_source`).
* Every record carries a fingerprint (model file hash, vocabulary hash, analysis version).
  Records of another fingerprint are re-analysed; until then they are left out of the combined
  reports – results of different versions are never mixed.
* Atomic writes (temporary file + rename) everywhere; an interrupted run leaves complete hours
  and an *INTERRUPTED* status; the next run continues with the missing hours. A new recording
  in an existing hour re-does only that hour (other recordings come from the per-recording
  cache, event identifiers stay the same, so review marks stay attached).
* Event IDs: `<sha256[:12]>-<category>-<start in 0.1 s>`; review marks are also matched by
  source, category and time, so they survive a re-analysis that shifts an ID.
* Lock: analysis and transcription never run on the same folder at the same time.
* Size: < 0.1 MB of results per analysed hour in typical recordings.

## 13. Playback and review (Sound Explorer)

Selecting an event shows its date and time, confidence, whether the time is estimated, the
source file and offset, and (if the folder was transcribed) the matching position in the
enhanced audio. *Play from event* starts 3 s before the event in the **original** recording;
playback continues past the event; pause/seek over the whole file are possible; *Enhanced*
switches to the enhanced 10-minute file at the same moment (when it exists). Decoding uses the
same FFmpeg decoder as the analysis, so the position is exactly the analysed position (measured
against a full decode: WAV, FLAC, M4A/AAC, MP3, Ogg Vorbis and Opus at 32/44.1/48 kHz all within
0.02 ms, after compensating the MP3 encoder delay that FFmpeg reports as the stream start time). For activities,
the supporting sounds are listed under it and each can be played; nothing is joined across
recordings. Review marks: Confirmed / Incorrect detection / Unreviewed.

## 14. What is deliberately not done

No speaker identification or voice biometrics, no emotion, intention, behaviour or relationship
inference, no live/microphone monitoring, no notifications, no user-adjustable thresholds or
model choice, no audio editing. Sounds are reported as observable sounds ("Crying / sobbing",
"Shouting"), never as states of a person.

## 15. Known weaknesses

* **Real-world accuracy is not measured on the owner's recordings.** The numbers in §6–§7 come
  from AudioSet evaluation clips (YouTube), ESC-50 (Freesound) mixtures and synthetic scenes.
  Real domestic recordings (phone in a pocket, distant microphone, Danish homes) will differ;
  the owner's acceptance test is the real measure.
* **AudioSet labels are weak and incomplete** (clip-level, some missing labels), so measured
  precision is conservative for common sounds and recall is approximate.
* **Loudspeaker vs. room**: sounds from TV, radio, phones and games are recognised like real
  ones; only stretches the model itself recognises as music/TV are flagged.
* **Overlapping speech** reduces recall of vocal sounds (laughter, crying) and quiet sounds;
  quiet or distant sounds are often missed (§7.3).
* **Similar sounds are confused**: alarm clock vs. beeps, hairdryer vs. vacuum cleaner vs.
  fan, running water vs. frying vs. rain, knocking vs. tapping vs. footsteps.
* **Timing**: ±1 s for short sounds, ±5 s for music/TV; events closer than the merge gap of the
  same kind are reported as one event.
* **Overlapping recordings** (two devices at once) produce two detections of the same sound,
  one per recording.
* **Estimated recording times** (file dates) can place events in the wrong hour; they are marked ≈.
* A recording that is open for playback in Sound Explorer cannot be moved by the transcription
  on Windows (file in use); it is moved on the next run.

## 16. Future improvements

* Fine-tune the classifier (or a small head on its embeddings) on labelled domestic recordings
  – the owner's confirmed/incorrect marks are the natural training data – instead of using
  AudioSet thresholds only.
* A frame-level detector trained for strong labels (e.g. ATST-Frame, BEATs-based DCASE
  systems) for sharper boundaries of short sounds.
* An ensemble (mn10 + dymn20) for categories near the enable threshold.
* GPU / DirectML execution for very large collections.
* A "playback device" detector trained to separate loudspeaker audio from room sound.
* More activity rules once the supporting categories are validated on real homes.
