# AudioRefine – Acoustic event detection (Analyze sounds, Sound Explorer)

Technical hand-over: what was investigated, what was chosen and why, how it works, how well it
works (measured), and what it cannot do. Written so that the decisions can be reviewed
independently. All figures were measured in this project unless a source is given.

## 1. Feasibility conclusion

**Feasible for a useful subset of sounds, with clear limits.** A compact AudioSet-trained
classifier running locally on the CPU recognises many everyday sounds well enough to be useful
for browsing large collections – 54 categories are enabled, among them crying, laughter,
snoring, sneezing, shouting, knocking, breaking glass, doorbell, hairdryer, vacuum cleaner,
toilet flush, frying, blender, keyboard typing, phone ringing, alarms, music, video game music,
dogs, cats, birds, rain, thunder and traffic sounds – at about 40 × (one CPU core) to 100 ×
(four cores) real time. On held-out AudioSet clips the enabled categories reach a mean precision
of 0.78 (0.92 at *High*) and a mean recall of 0.54; event-free speech and quiet rooms gave no
detections. Recall drops strongly when sounds are quiet, distant, or masked by conversation or
TV (§7.3). Many other requested sounds (coughing, sighing, whimpering, screaming, footsteps,
doors and drawers, banging, dishes, microwave, kettle, running water, television, mouse clicks,
washing machine, dishwasher …) are **disabled** because no threshold reached the precision and
recall targets, or because the model has no such class – rather than being shown with
misleading results (§6). Activity inference is implemented but effectively deferred: with the
enabled categories only *Possible cooking* (frying + blender) can fire (§11). Timing is accurate
to about ±1 s for short sounds and ±5 s for music/TV.

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

* Accuracy per CPU cost: on the AudioSet evaluation clips mn10 (mean AP 0.694) is as accurate as
  AST (0.694) and better than CNN14 (0.669), at 1/15 – 1/60 of their CPU time; only dymn20 is
  clearly better (0.736) at 6 × the cost (§7.1).
* Deployment: a single 19.5 MB ONNX file; the application already ships ONNX Runtime, so the
  package grows by only ~20 MB and needs no PyTorch (≈ 200 MB+), no download and no internet.
* Licence: MIT (code and weights); the AudioSet label list is CC BY 4.0.
* The export was verified against PyTorch (max. difference 7·10⁻⁷) and the numpy front-end
  against the PyTorch front-end (max. difference 4·10⁻⁵).

A two-stage design "cheap detector → large validator" (e.g. mn10 → AST or CLAP) was considered.
It was not adopted: (1) the larger supervised models are at most a few AP points better (§7.1)
but 6–60 × slower; (2) CLAP zero-shot accuracy for AudioSet-style categories is clearly below the
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


## 6. Vocabulary, thresholds and which categories are enabled

### 6.1 Method (``tools/sound/tune.py``)

* **Candidates**: 107 categories built from the requested list (``tools/sound/categories_candidates.json``),
  each a set of AudioSet classes (e.g. *Crying / sobbing* = "Crying, sobbing" + "Baby cry, infant
  cry"), with a kind (impact, vocal, sustained, media) that sets merging and evidence rules.
  Categories that would report the same sound twice were merged (baby crying into crying,
  pouring into running water, cutlery into dishes) or left out (generic "Water", mechanical hum).
* **Data**: the AudioSet *evaluation* split (17,141 clips downloaded; the models were trained only
  on the training splits) – 10,003 clips: every clip with any candidate label plus 2,000 random
  others. Each clip scored exactly as the application does (2 s windows + 10 s context, combined).
* **Split**: 50/50 by video id. Thresholds are chosen on half A, all reported precision/recall is
  measured on the untouched half B.
* **Thresholds**: *moderate* = lowest threshold with precision ≥ 0.65 on A (vocal sounds ≥ 0.70,
  because they occur inside conversation), *high* = precision ≥ 0.85 (vocal ≥ 0.88); floors 0.12 /
  0.30; then raised until event-free material yields ≤ 0.5 moderate and ≤ 0.1 high detections per
  hour (1.4 h of Danish/English read speech + 0.5 h of quiet room).
* **Enabled** only with ≥ 12 test examples, precision ≥ 0.62 and recall ≥ 0.25 at the moderate
  threshold on half B, and not excluded for a design reason.
* Labels in AudioSet are clip-level and incomplete, so precision is underestimated for common
  sounds (unlabelled true occurrences count as false alarms) and recall is approximate.

Result: **54 of 107 candidate categories enabled**. On the test half the enabled categories have a
mean precision of 0.78 and mean recall of 0.54 at *Moderate* (median 0.75 / 0.53), and mean
precision 0.92 / recall 0.33 at *High*. Event-free read speech and quiet room: 0 detections per
hour at the chosen thresholds.

### Enabled categories

| Group | Category | AudioSet classes | Moderate / High threshold | Precision / recall at moderate (test half) | n | False detections per hour of speech + quiet room |
|---|---|---|---|---|---|---|
| Human sounds | Crying / sobbing | Crying, sobbing, Baby cry, infant cry | 0.29 / 0.58 | 0.78 / 0.56 | 45 | 0.0 |
| Human sounds | Laughter | Laughter, Baby laughter, Giggle, Snicker, Belly laugh, Chuckle, chortle | 0.16 / 0.30 | 0.75 / 0.49 | 128 | 0.0 |
| Human sounds | Sneezing | Sneeze | 0.38 / 0.98 | 0.69 / 0.39 | 23 | 0.0 |
| Human sounds | Snoring | Snoring | 0.17 / 0.30 | 0.90 / 0.69 | 26 | 0.0 |
| Human sounds | Shouting | Shout, Yell, Children shouting | 0.12 / 0.30 | 0.73 / 0.38 | 66 | 0.0 |
| Human sounds | Singing | Singing, Male singing, Female singing, Child singing, Choir | 0.13 / 0.35 | 0.70 / 0.57 | 232 | 0.0 |
| Human sounds | Whistling | Whistling | 0.18 / 0.36 | 0.92 / 0.92 | 25 | 0.0 |
| Human sounds | Clapping / applause | Clapping, Applause | 0.13 / 0.71 | 0.62 / 0.56 | 54 | 0.0 |
| Impacts and movement | Knocking | Knock | 0.21 / 0.31 | 0.81 / 0.56 | 23 | 0.0 |
| Impacts and movement | Tapping | Tap | 0.12 / 0.30 | 0.75 / 0.26 | 23 | 0.0 |
| Impacts and movement | Breaking glass | Shatter, Breaking | 0.23 / 0.70 | 0.64 / 0.39 | 41 | 0.0 |
| Impacts and movement | Coins dropping | Coin (dropping) | 0.23 / 0.98 | 0.67 / 0.35 | 34 | 0.0 |
| Household | Doorbell | Doorbell, Ding-dong | 0.12 / 0.30 | 0.71 / 0.51 | 43 | 0.0 |
| Household | Hammering | Hammer | 0.12 / 0.30 | 1.00 / 0.27 | 37 | 0.0 |
| Household | Zipper | Zipper (clothing) | 0.38 / 0.48 | 1.00 / 0.30 | 30 | 0.0 |
| Household | Vacuum cleaner | Vacuum cleaner | 0.12 / 0.98 | 0.83 / 0.46 | 22 | 0.0 |
| Household | Printer | Printer | 0.12 / 0.30 | 0.73 / 0.37 | 30 | 0.0 |
| Household | Clock ticking | Clock, Tick, Tick-tock | 0.12 / 0.30 | 0.81 / 0.33 | 75 | 0.0 |
| Kitchen | Frying / sizzling | Frying (food), Sizzle | 0.12 / 0.30 | 0.79 / 0.54 | 43 | 0.0 |
| Kitchen | Blender | Blender | 0.12 / 0.98 | 0.71 / 0.40 | 25 | 0.0 |
| Kitchen | Packaging / crinkling | Crumpling, crinkling, Tearing | 0.12 / 0.30 | 0.85 / 0.33 | 52 | 0.0 |
| Personal care and bathroom | Hairdryer | Hair dryer | 0.12 / 0.30 | 1.00 / 0.62 | 21 | 0.0 |
| Personal care and bathroom | Toilet flushing | Toilet flush | 0.12 / 0.30 | 0.80 / 0.50 | 16 | 0.0 |
| Entertainment and electronics | Music | Music | 0.29 / 0.74 | 0.66 / 0.93 | 1024 | 0.0 |
| Entertainment and electronics | Radio | Radio | 0.31 / 0.45 | 0.82 / 0.33 | 27 | 0.0 |
| Entertainment and electronics | Video game music | Video game music | 0.12 / 0.30 | 0.73 / 0.46 | 24 | 0.0 |
| Entertainment and electronics | Keyboard typing | Typing, Computer keyboard, Typewriter | 0.12 / 0.30 | 0.81 / 0.58 | 60 | 0.0 |
| Entertainment and electronics | Phone ringing | Telephone, Telephone bell ringing, Ringtone | 0.12 / 0.33 | 0.71 / 0.67 | 60 | 0.0 |
| Entertainment and electronics | Beeps | Beep, bleep | 0.55 / 0.98 | 0.88 / 0.29 | 24 | 0.0 |
| Entertainment and electronics | Alarm | Alarm, Alarm clock, Smoke detector, smoke alarm, Fire alarm, Buzzer | 0.12 / 0.30 | 0.75 / 0.50 | 106 | 0.0 |
| Outdoor and transport | Siren | Siren, Police car (siren), Ambulance (siren), Fire engine, fire truck (siren), Civil defense siren | 0.12 / 0.30 | 0.94 / 0.76 | 84 | 0.0 |
| Outdoor and transport | Car | Car, Car passing by, Motor vehicle (road) | 0.20 / 0.34 | 0.63 / 0.56 | 164 | 0.0 |
| Outdoor and transport | Car horn | Vehicle horn, car horn, honking | 0.55 / 0.98 | 0.67 / 0.38 | 21 | 0.0 |
| Outdoor and transport | Engine | Engine, Idling, Accelerating, revving, vroom | 0.21 / 0.33 | 0.65 / 0.32 | 159 | 0.0 |
| Outdoor and transport | Train | Train, Train whistle, Train horn, Railroad car, train wagon | 0.12 / 0.30 | 0.93 / 0.57 | 98 | 0.0 |
| Outdoor and transport | Aircraft | Aircraft, Aircraft engine, Fixed-wing aircraft, airplane, Jet engine | 0.12 / 0.30 | 0.85 / 0.73 | 85 | 0.0 |
| Outdoor and transport | Helicopter | Helicopter | 0.12 / 0.30 | 0.73 / 0.42 | 26 | 0.0 |
| Outdoor and transport | Rain | Rain, Raindrop, Rain on surface | 0.12 / 0.30 | 0.73 / 0.58 | 38 | 0.0 |
| Outdoor and transport | Wind | Wind, Wind noise (microphone) | 0.12 / 0.30 | 0.67 / 0.35 | 88 | 0.0 |
| Outdoor and transport | Thunder | Thunder, Thunderstorm | 0.12 / 0.30 | 0.75 / 0.65 | 23 | 0.0 |
| Outdoor and transport | Lawn mower | Lawn mower | 0.12 / 0.30 | 0.67 / 0.37 | 27 | 0.0 |
| Outdoor and transport | Chainsaw | Chainsaw | 0.41 / 0.98 | 0.88 / 0.48 | 31 | 0.0 |
| Outdoor and transport | Church bells | Church bell | 0.13 / 0.37 | 0.63 / 0.70 | 27 | 0.0 |
| Outdoor and transport | Fireworks | Fireworks, Firecracker | 0.12 / 0.30 | 0.88 / 0.69 | 32 | 0.0 |
| Animals | Dog barking | Bark, Yip, Bow-wow, Dog | 0.23 / 0.70 | 0.71 / 0.74 | 122 | 0.0 |
| Animals | Howling | Howl | 0.14 / 0.74 | 0.65 / 0.77 | 22 | 0.0 |
| Animals | Cat meowing | Meow, Cat, Caterwaul | 0.28 / 0.43 | 0.78 / 0.74 | 58 | 0.0 |
| Animals | Cat purring | Purr | 0.12 / 0.30 | 0.83 / 0.80 | 25 | 0.0 |
| Animals | Birds | Bird, Bird vocalization, bird call, bird song, Chirp, tweet, Squawk, Crow, Caw, Pigeon, dove | 0.20 / 0.63 | 0.69 / 0.53 | 133 | 0.0 |
| Animals | Rooster / chickens | Chicken, rooster, Crowing, cock-a-doodle-doo | 0.18 / 0.30 | 0.66 / 0.94 | 31 | 0.0 |
| Animals | Insects | Insect, Cricket, Mosquito, Fly, housefly | 0.12 / 0.30 | 0.92 / 0.49 | 71 | 0.0 |
| Animals | Frogs | Frog | 0.12 / 0.30 | 0.81 / 0.92 | 24 | 0.0 |
| Animals | Horse | Horse | 0.20 / 0.49 | 0.77 / 0.63 | 43 | 0.0 |
| Animals | Cattle | Cattle, bovinae | 0.12 / 0.30 | 0.91 / 0.75 | 28 | 0.0 |

### Investigated but disabled

| Group | Category | Reason (test half) |
|---|---|---|
| Human sounds | Baby crying | included in 'Crying / sobbing' (one sound must not be reported twice); recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 27) |
| Human sounds | Whimpering | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 33) |
| Human sounds | Sighing | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 21) |
| Human sounds | Coughing | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 23) |
| Human sounds | Screaming | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 22) |
| Human sounds | Whispering | precision 0.60 < 0.62 (P 0.60, R 0.90, n 29) |
| Human sounds | Humming (voice) | precision 0.43 < 0.62; recall 0.19 < 0.25 (P 0.43, R 0.19, n 16) |
| Human sounds | Groaning | recall 0.11 < 0.25 (P 1.00, R 0.11, n 46) |
| Human sounds | Heavy breathing | recall 0.25 < 0.25 (P 0.87, R 0.25, n 109) |
| Human sounds | Throat clearing | precision 0.55 < 0.62 (P 0.55, R 0.27, n 22) |
| Human sounds | Sniffing | recall 0.00 < 0.25 (P 1.00, R 0.00, n 29) |
| Human sounds | Baby babbling | precision 0.57 < 0.62 (P 0.57, R 0.32, n 25) |
| Human sounds | Children playing | precision 0.62 < 0.62 (P 0.61, R 0.42, n 19) |
| Human sounds | Chewing / eating | precision 0.62 < 0.62 (P 0.62, R 0.59, n 22) |
| Impacts and movement | Footsteps | precision 0.58 < 0.62 (P 0.58, R 0.33, n 21) |
| Impacts and movement | Running (footsteps) | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 20) |
| Impacts and movement | Banging / impact | recall 0.19 < 0.25 (P 0.72, R 0.19, n 93) |
| Impacts and movement | Door slamming | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 23) |
| Impacts and movement | Crash / smash | precision 0.61 < 0.62 (P 0.61, R 0.38, n 29) |
| Impacts and movement | Scraping | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 26) |
| Impacts and movement | Clattering | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 30) |
| Impacts and movement | Squeaking / creaking | recall 0.23 < 0.25 (P 0.85, R 0.23, n 47) |
| Impacts and movement | Keys jangling | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 32) |
| Household | Door opening / closing | precision 0.50 < 0.62; recall 0.05 < 0.25 (P 0.50, R 0.05, n 38) |
| Household | Cupboard / drawer | recall 0.18 < 0.25 (P 1.00, R 0.17, n 57) |
| Household | Scissors | recall 0.19 < 0.25 (P 0.80, R 0.19, n 21) |
| Household | Fan / air conditioning | recall 0.04 < 0.25 (P 1.00, R 0.04, n 53) |
| Household | Mechanical / electrical hum | constant background hum of fridges, mains and fans – would mark almost every hour; precision 0.50 < 0.62; recall 0.20 < 0.25 (P 0.50, R 0.20, n 45) |
| Household | Sewing machine | recall 0.16 < 0.25 (P 1.00, R 0.16, n 32) |
| Household | Power tool / drill | precision 0.57 < 0.62 (P 0.57, R 0.29, n 41) |
| Household | Sawing | recall 0.22 < 0.25 (P 0.86, R 0.22, n 27) |
| Kitchen | Glass clinking | precision 0.57 < 0.62; recall 0.08 < 0.25 (P 0.57, R 0.08, n 50) |
| Kitchen | Pouring / filling | included in 'Running or pouring water'; recall 0.12 < 0.25 (P 0.64, R 0.12, n 57) |
| Kitchen | Chopping (food) | recall 0.19 < 0.25 (P 1.00, R 0.19, n 21) |
| Kitchen | Dishes / cutlery | precision 0.57 < 0.62; recall 0.13 < 0.25 (P 0.57, R 0.13, n 30) |
| Kitchen | Cutlery | included in 'Dishes / cutlery' (P 0.75, R 0.25, n 24) |
| Kitchen | Microwave oven | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 31) |
| Kitchen | Boiling | recall 0.19 < 0.25 (P 0.71, R 0.18, n 27) |
| Personal care and bathroom | Electric shaver | precision 0.50 < 0.62 (P 0.50, R 0.63, n 19) |
| Personal care and bathroom | Toothbrush | recall 0.15 < 0.25 (P 0.64, R 0.15, n 48) |
| Personal care and bathroom | Running or pouring water | precision 0.61 < 0.62 (P 0.60, R 0.46, n 101) |
| Personal care and bathroom | Water splashing | precision 0.60 < 0.62 (P 0.60, R 0.27, n 22) |
| Personal care and bathroom | Dripping | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 36) |
| Personal care and bathroom | Water (general) | too generic (any water sound) to be useful; precision 0.51 < 0.62 (P 0.51, R 0.68, n 110) |
| Entertainment and electronics | Television | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 32) |
| Entertainment and electronics | Electronic sound effects | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 16) |
| Entertainment and electronics | Clicking | recall 0.05 < 0.25 (P 1.00, R 0.05, n 19) |
| Entertainment and electronics | Chime / ding | precision 0.51 < 0.62 (P 0.51, R 0.46, n 44) |
| Outdoor and transport | Motorcycle | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 26) |
| Outdoor and transport | Traffic | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 24) |
| Animals | Dog growling | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 31) |
| Animals | Dog whining / whimpering | recall 0.00 < 0.25; no usable threshold (P 1.00, R 0.00, n 22) |
| Speech | Speech | speech is covered by the transcription; recall 0.10 < 0.25 (P 0.90, R 0.10, n 1486) |


Notable gaps among the requested sounds: coughing, sighing, whimpering, screaming, heavy
breathing, footsteps, door slams/doors/drawers, banging, dishes, cutlery, microwave, kettle,
electric toothbrush, running water (precision 0.61 – just below the bar), splashing, fans and
mechanical hum, television audio, mouse clicks and electronic sound effects are **disabled**.
Washing machine, tumble dryer, dishwasher, kettle, yawning, shower and furniture movement have
no AudioSet class and could not be considered. Video game audio is available only as *Video
game music*.

## 7. Test results

### 7.1 Model comparison (AudioSet evaluation clips, 10-second clips)

992 AudioSet evaluation clips (25 positives per category where available + 400 negatives),
each model on the whole 10-second clip (the condition all of them were trained for), average
precision per category (max over the category's AudioSet classes). Time per clip measured
separately on 20 clips, 4 CPU threads (Intel Xeon 2.1 GHz), after warm-up.

| Category | n | **mn10** (selected) | dymn10 | dymn20 | mn40_ext | CNN14 | CNN14-DecisionLevelMax | AST |
|---|---|---|---|---|---|---|---|---|
| Crying / sobbing | 42 | 0.89 | 0.89 | 0.90 | 0.89 | 0.85 | 0.77 | 0.87 |
| Laughter | 32 | 0.61 | 0.57 | 0.61 | 0.58 | 0.52 | 0.51 | 0.59 |
| Coughing | 26 | 0.79 | 0.85 | 0.85 | 0.82 | 0.83 | 0.78 | 0.77 |
| Sneezing | 25 | 0.67 | 0.68 | 0.71 | 0.70 | 0.72 | 0.59 | 0.62 |
| Shouting | 25 | 0.76 | 0.83 | 0.83 | 0.79 | 0.78 | 0.71 | 0.81 |
| Screaming | 25 | 0.71 | 0.63 | 0.72 | 0.62 | 0.61 | 0.54 | 0.57 |
| Footsteps | 25 | 0.70 | 0.61 | 0.66 | 0.68 | 0.53 | 0.61 | 0.62 |
| Knocking | 26 | 0.72 | 0.76 | 0.83 | 0.79 | 0.71 | 0.82 | 0.82 |
| Banging / impact | 25 | 0.43 | 0.49 | 0.48 | 0.49 | 0.39 | 0.33 | 0.45 |
| Door opening / closing | 20 | 0.35 | 0.34 | 0.43 | 0.37 | 0.38 | 0.31 | 0.34 |
| Hairdryer | 25 | 0.93 | 0.93 | 0.95 | 0.91 | 0.92 | 0.92 | 0.98 |
| Toothbrush | 25 | 0.68 | 0.73 | 0.75 | 0.76 | 0.72 | 0.61 | 0.76 |
| Running or pouring water | 32 | 0.81 | 0.82 | 0.86 | 0.79 | 0.80 | 0.66 | 0.81 |
| Frying / sizzling | 27 | 0.88 | 0.87 | 0.90 | 0.87 | 0.91 | 0.84 | 0.89 |
| Dishes / cutlery | 25 | 0.52 | 0.59 | 0.64 | 0.61 | 0.62 | 0.51 | 0.59 |
| Microwave oven | 26 | 0.52 | 0.58 | 0.55 | 0.53 | 0.45 | 0.41 | 0.55 |
| Blender | 25 | 0.75 | 0.75 | 0.79 | 0.77 | 0.67 | 0.51 | 0.72 |
| Vacuum cleaner | 25 | 0.77 | 0.76 | 0.73 | 0.75 | 0.69 | 0.65 | 0.78 |
| Fan / air conditioning | 26 | 0.67 | 0.67 | 0.71 | 0.64 | 0.55 | 0.42 | 0.63 |
| Music | 121 | 0.74 | 0.77 | 0.79 | 0.74 | 0.79 | 0.61 | 0.68 |
| Television | 26 | 0.42 | 0.45 | 0.57 | 0.38 | 0.33 | 0.22 | 0.43 |
| Video game music | 28 | 0.89 | 0.93 | 0.91 | 0.91 | 0.87 | 0.80 | 0.91 |
| Dog barking | 28 | 0.81 | 0.83 | 0.86 | 0.86 | 0.83 | 0.75 | 0.82 |
| Engine | 30 | 0.63 | 0.64 | 0.64 | 0.61 | 0.57 | 0.57 | 0.64 |
| **Mean AP** | | **0.694** | 0.707 | **0.736** | 0.702 | 0.669 | 0.604 | 0.694 |
| **s per 10 s clip, 4 threads** | | **0.012** (ONNX) / 0.030 (PyTorch) | 0.046 | 0.071 | 0.101 | 0.186 | 0.186 | 0.697 |
| Parameters / file size | | 4.9 M / 19.5 MB | 10.6 M / 42 MB | 40 M / 160 MB | 68 M / 274 MB | 80 M / 327 MB | 80 M / 327 MB | 87 M / 345 MB |

mn10 in ONNX gives exactly the same scores as in PyTorch (identical column, not repeated).
With ~25 positives per category the per-category differences of a few points are within noise;
the means are more reliable. Reading: **mn10 is as accurate as AST and better than both PANNs
models at 1/15 – 1/60 of their CPU time.** **dymn20 is the only clearly better candidate
(+4 AP points on average)** but is 6 × slower, 8 × larger, and its dynamic convolutions did not
export to ONNX with the standard exporter (PyTorch would add ~200 MB to the package). It is the
first candidate for a future upgrade (§16). The frame-wise CNN14 variant, designed for event
detection, was clearly worse as a detector. AST needs fixed 10-second inputs, so ±1 s localisation
would cost 10 inferences per second of audio (≈ 7 × real time on 4 threads).

### 7.2 Held-out AudioSet evaluation clips (the application's two-scale scoring)

See the tables in §6 (precision/recall on test half B per category).

### 7.3 Difficult conditions (ESC-50 – Freesound recordings, independent of AudioSet)

ESC-50 clips (test folds 4–5, 32 categories mapped to enabled categories) were mixed into
10-second backgrounds at levels relative to the background (``tools/sound/make_mixtures.py``):
read speech (conversation-like), a TV-like bed (music + speech from AudioSet clips labelled
television/music, through a 180 Hz – 7 kHz "small loudspeaker" filter) and a quiet room;
"distant" = synthetic reverberation (RT60 0.7 s, weak direct sound) and −10 dB. Detection =
the event's category reported at moderate confidence anywhere in the clip.

| Condition | Detected (enabled categories) |
|---|---|
| none clean | 423/544 = 78% |
| quiet -20dB | 255/544 = 47% |
| quiet distant | 173/544 = 32% |
| speech 0dB | 301/544 = 55% |
| speech -10dB | 219/544 = 40% |
| speech -20dB | 146/544 = 27% |
| speech distant | 122/544 = 22% |
| tv 0dB | 158/544 = 29% |
| tv -10dB | 66/544 = 12% |

| Category | none clean | quiet -20dB | quiet distant | speech 0dB | speech -10dB | speech -20dB | speech distant | tv 0dB | tv -10dB |
|---|---|---|---|---|---|---|---|---|---|
| aircraft | 6/16 | 0/16 | 0/16 | 3/16 | 2/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| alarm | 15/16 | 6/16 | 7/16 | 11/16 | 9/16 | 6/16 | 6/16 | 7/16 | 3/16 |
| bird | 29/32 | 29/32 | 25/32 | 28/32 | 26/32 | 22/32 | 22/32 | 22/32 | 13/32 |
| cat | 16/16 | 16/16 | 16/16 | 15/16 | 14/16 | 14/16 | 14/16 | 7/16 | 6/16 |
| cattle | 16/16 | 13/16 | 11/16 | 14/16 | 14/16 | 14/16 | 13/16 | 7/16 | 3/16 |
| chainsaw | 5/16 | 1/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 1/16 | 0/16 |
| church_bell | 15/16 | 8/16 | 10/16 | 10/16 | 6/16 | 1/16 | 8/16 | 3/16 | 1/16 |
| clapping | 16/16 | 8/16 | 0/16 | 14/16 | 10/16 | 4/16 | 0/16 | 9/16 | 4/16 |
| clock | 13/16 | 16/16 | 1/16 | 3/16 | 3/16 | 3/16 | 0/16 | 2/16 | 2/16 |
| crying | 16/16 | 7/16 | 0/16 | 9/16 | 4/16 | 2/16 | 1/16 | 6/16 | 0/16 |
| dog | 16/16 | 12/16 | 14/16 | 16/16 | 11/16 | 7/16 | 10/16 | 12/16 | 1/16 |
| engine | 10/16 | 0/16 | 0/16 | 2/16 | 2/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| fireworks | 15/16 | 0/16 | 0/16 | 12/16 | 9/16 | 0/16 | 0/16 | 8/16 | 2/16 |
| frog | 16/16 | 12/16 | 8/16 | 12/16 | 9/16 | 6/16 | 4/16 | 8/16 | 4/16 |
| glass_break | 14/16 | 1/16 | 0/16 | 7/16 | 6/16 | 1/16 | 0/16 | 6/16 | 3/16 |
| helicopter | 6/16 | 0/16 | 0/16 | 4/16 | 0/16 | 0/16 | 0/16 | 1/16 | 0/16 |
| horn | 15/16 | 5/16 | 7/16 | 4/16 | 1/16 | 1/16 | 1/16 | 1/16 | 0/16 |
| insect | 21/32 | 21/32 | 8/32 | 20/32 | 15/32 | 13/32 | 5/32 | 3/32 | 2/32 |
| knock | 15/16 | 16/16 | 11/16 | 11/16 | 5/16 | 3/16 | 1/16 | 2/16 | 1/16 |
| laughter | 13/16 | 1/16 | 1/16 | 8/16 | 4/16 | 1/16 | 0/16 | 5/16 | 1/16 |
| rain | 12/16 | 2/16 | 0/16 | 8/16 | 4/16 | 0/16 | 0/16 | 2/16 | 0/16 |
| rooster | 31/32 | 31/32 | 23/32 | 30/32 | 28/32 | 25/32 | 25/32 | 24/32 | 11/32 |
| siren | 14/16 | 12/16 | 13/16 | 13/16 | 10/16 | 6/16 | 8/16 | 5/16 | 4/16 |
| sneeze | 9/16 | 3/16 | 3/16 | 5/16 | 2/16 | 0/16 | 1/16 | 1/16 | 0/16 |
| snoring | 16/16 | 12/16 | 10/16 | 12/16 | 7/16 | 3/16 | 3/16 | 5/16 | 1/16 |
| thunder | 10/16 | 0/16 | 0/16 | 5/16 | 1/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| toilet | 15/16 | 9/16 | 4/16 | 9/16 | 2/16 | 0/16 | 0/16 | 3/16 | 3/16 |
| train | 12/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| typing | 16/16 | 14/16 | 0/16 | 14/16 | 15/16 | 14/16 | 0/16 | 8/16 | 1/16 |
| vacuum | 0/16 | 0/16 | 1/16 | 2/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| wind | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |

Background only (10 s clips):
  speech: 0 false detections in 60 clips (0.17 h) – {}
  tv: 3 false detections in 60 clips (0.17 h) – {'laughter': 1, 'doorbell': 1, 'horse': 1}


Reading: clean, isolated recordings from another collection are recognised 78 % of the time;
masking by conversation, distance and especially TV audio reduces recall a lot (to 12–55 %).
False detections on the backgrounds themselves stayed rare (0 on speech; 3 on TV, all
plausible TV content). Some enabled categories transfer poorly to ESC-50 recordings (vacuum
cleaner 0/16, wind 0/16, train/aircraft/engine low) – their recall outside AudioSet is low, while
their detections, when they occur, are mostly right.

### 7.4 Background material (false detections per hour, final thresholds)

| Material | Duration | Detections per hour |
|---|---|---|
| Read speech, Danish + English (FLEURS dev, with pauses and room noise) | 1.4 h | 0 |
| Quiet room (noise + faint mains hum) | 0.5 h | 0 |
| Meeting recordings (AMI, single distant microphone, real meetings) | 2.4 h | 48: laughter 47, clapping 0.9, cat purring 0.4 – laughter is very frequent in these meetings, so most are probably real; the purring is false |
| Speech with music / hum / distance degradations | 0.4 h | 10: music 7.5 (correct – the material contains music), insects 2.5 (false) |

### 7.5 Real recordings from Wikimedia Commons (CC0 / public domain, isolated sounds)

| Recording (Wikimedia Commons, CC0/PD) | Detected (confidence) |
|---|---|
| alarm | Alarm (high) |
| applause | Clapping / applause (moderate) |
| baby_cry | Crying / sobbing (moderate) |
| birds | Birds (high), Birds (moderate) |
| cat | Cat meowing (high) |
| clock | Clock ticking (moderate) |
| doorbell | Doorbell (moderate) |
| frying | Frying / sizzling (high) |
| knock | Knocking (high) |
| knock2 | Knocking (high) |
| laughter | Laughter (high) |
| microwave | Alarm (moderate) |
| pouring | – |
| rain_thunder | Rain (moderate), Thunder (moderate) |
| scream | Shouting (moderate) |
| siren | Whistling (moderate) |
| sneeze | Sneezing (moderate) |
| snoring | Snoring (high) |
| typing | Keyboard typing (high) |
| videogame | Music (moderate) |


16 of 20 recordings are labelled correctly (scream → *Shouting* is a reasonable broader label);
the microwave's beeps are reported as *Alarm* (microwave is disabled), a toy siren as
*Whistling*, chiptune game music only as *Music*; pouring water is not detected (category
disabled).

### 7.6 End-to-end scene with known times (CI, ``tools/ci/make_sound_folder.py``)

Six recordings over three days (incl. one crossing 09:00, one already archived in `processed`,
one without a recording time) with real CC0/PD sounds at −24 dBFS in room noise and speech:
10 of the 11 expected sounds of enabled categories were found, each within ±1 s of the true
start (the missed one is a 1.3-second laugh); the knock at 09:00:20 in the recording that
started at 08:59:45 appears once, in the 09:00 hour; the recording without a time is marked ≈;
originals unchanged; a second run finds nothing new. The same check runs on every push with the
packaged Windows build (TEST_REPORT.md §3, run #13: identical result – 10 of 11 found, no other
detections; the transcribed folder's archived originals analysed, 0 events in speech-only audio,
annotated transcript with all 39 entries unchanged).

## 8. Performance

| Measure | Value |
|---|---|
| Analysis speed, 1 hour of audio (M4A), 4 threads, Intel Xeon 2.1 GHz (cloud VM) | 35 s (≈ 100 × real time) |
| Same, 1 thread | 89 s (≈ 40 × real time) |
| Packaged Windows build on the CI runner (4 vCPU), 16–17 min of audio | 10–11 s incl. model loading (≈ 95 × real time) |
| Peak memory (whole process) | ≈ 460 MB |
| Model inferences per hour of audio | 3,600 localisation + 720 context windows |
| Added to the portable ZIP | ≈ 20 MB (model 19.5 MB, labels, vocabulary) + PortAudio ≈ 0.3 MB |
| Results on disk | < 0.1 MB per analysed hour |
| GPU | not used (not needed at this speed) |

A collection of 500 hours of recordings takes roughly 5 hours on a 4-core desktop CPU –
far less than its transcription. Decoding of compressed formats is included in these times.


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

Implemented as two conservative rules (``sound/activities.py``). **In practice deferred**: most
supporting categories (chopping, dishes, microwave, boiling, running water) did not pass the
evaluation and are disabled, so with the shipped vocabulary only *Possible cooking* from frying
+ blender can fire, and *Possible washing up* cannot. The rules switch on automatically if those
categories are enabled after real-world validation.

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
* Switch to (or add) EfficientAT `dymn20_as` (+4 AP points in §7.1): needs an ONNX export of its
  dynamic convolutions (e.g. the dynamo-based exporter) and re-running the tuning (§6).
* GPU / DirectML execution for very large collections.
* A "playback device" detector trained to separate loudspeaker audio from room sound.
* More activity rules once the supporting categories are validated on real homes.
