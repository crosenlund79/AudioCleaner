# AudioRefine 1.0 – Test report

Status legend: **Verified** = executed and passed, with evidence below. **Partly verified** =
covered by automated or indirect checks only. **Unverified** = not executed (reason given).

Test environments:

* **Linux development VM** – 4 vCPU Intel Xeon @ 2.1 GHz, 15.7 GB RAM, no GPU, Python 3.13.
* **GitHub Actions `windows-latest`** – Windows Server 2025 (10.0.26100), 4 vCPU AMD EPYC,
  16 GB RAM, no GPU. Every push builds, tests and packages the application here.

## 1. Summary

| Area | Status | Evidence |
|---|---|---|
| Automated test suite (112 tests) on Linux | Verified | §2 |
| Same suite on Windows (GitHub Actions) | Verified – 108/108 passed in run #4, re-run on every push | §2 |
| Portable Windows build (PyInstaller) builds; packaged self-test | Verified | §3 |
| Packaged desktop window starts on Windows and lists a folder | Verified (automated screenshot, native Windows platform) | §3 |
| Packaged first-run download of the 3.1 GB model, SHA-256 verified | Verified on Windows | §3 |
| Resuming an interrupted download | Verified (Linux, HTTP Range through redirects) | §2 |
| Packaged end-to-end run on Windows, Whisper large-v3, real Danish + English speech, 12 min, crossing a 10-minute boundary | Verified | §3 |
| Multi-hour processing (2 h, two files, packaged engine) | Verified (Linux) | §5 |
| Enhancement vs. transcription accuracy (original / conservative / neural) | Verified with WER, large-v3 | §4.2 |
| Intelligibility effect of the enhancement | Partly verified (objective ESTOI only) | §4.3 |
| Listening tests by people | **Unverified** – not possible in this environment | – |
| Interactive use of the window by a person on a Windows desktop | **Unverified** – no interactive Windows session; GUI flows were driven programmatically (Linux) and the packaged window was started on Windows | §3 |
| Conversational / dialect Danish | **Unverified** – no openly licensed conversational Danish corpus without registration; tests use read speech with realistic degradations | §4 |
| NVIDIA GPU acceleration | **Unverified** – no GPU in development or CI; GPU detection returns "none" and the CPU path is verified | – |
| AMR input | Partly verified – AMR-NB/WB decoders present in the bundled FFmpeg; no encoder available to create a test file | §2 |
| Behaviour when the PC loses power | Partly verified – engine process killed mid-run (same effect on files) | §2 |

## 2. Automated tests

`python -m pytest` (`tests/`), 112 tests. Pipeline tests use the real pipeline with the small
`faster-whisper-tiny` model on real FLEURS speech; everything else runs without a model.

| File | Tests | What is covered |
|---|---|---|
| `test_ordering.py` | 10 | natural sort (`Recording2` < `Recording10`, case, leading zeros, multiple numbers); date/time in file names (8 formats incl. Danish day-first), rejection of non-dates; metadata parsing; chronological vs. natural strategy; metadata contradiction warning; file-system time never used |
| `test_rawaudio.py` | 16 | exact 48→16 kHz decimation length for 12 lengths incl. 0–4 samples; block-size independence; zero-delay alignment; pass-band/alias rejection; raw file round trip |
| `test_decode_discovery.py` | 22 | 8 sample rates (8–96 kHz) to exact 48 kHz duration; WAV, FLAC, OGG, MP3, M4A/AAC, Opus, AIFF, WMA, ADTS-AAC decode; mono/stereo/phase-inverted/dead-channel downmix; damaged, empty and truncated files; top-level-only discovery, subfolders/hidden/temp/unrelated files ignored |
| `test_segments_transcript.py` | 17 | exact 10-minute segmentation for N−1, N, N+1 samples and multi-hour (5.5 h → 34 files); FLAC write/read-back with samples just before/at/after a boundary, no loss/duplication; segment verification; timestamp formats; sentence crossing 10:00 split at word level with `…`; word exactly at 10:00.000; SRT relative (10:04 → 00:00:04) and TXT absolute; SRT cue limits/no overlap/no lost words; markers; complete transcript; empty segment |
| `test_batch_safety.py` | 12 | batch name collisions (`_2`, `_3`); staging + atomic commit; never overwrite; stale-staging cleanup only with marker; archive with SHA-256 verification; existing file in `processed` not overwritten; **partial archival failure (file locked) and recovery on the next run**; original changed after processing is not moved; crash between move and journal; folder lock; stale lock; stale work folders; scan cleans up after a killed run |
| `test_enhance_asr_logic.py` | 24 | arfilter keeps exact length (0, 1, 479–481, …) and adds no delay; parallel chunks ≈ single pass (< −26 dB); leveller: −1 dBFS ceiling, door slam limited, quiet speech raised, nothing gated; 20 dB speaker difference equalised to < 6 dB; hum detection 50/60 Hz; bounded plan; VAD regions/padding; **VAD detects a −65 dBFS utterance** and **does not turn 2 minutes of room noise into speech**; subtitle-credit artefacts; repeated low-confidence segments; clip/block planning; per-clip gain; hallucination phrases; repetition loops; duplicated-sentence detection; uncertainty flags; language policy (Norwegian/Swedish → Danish, hysteresis, other languages) |
| `test_pipeline.py` | 11 | full run (mixed m4a/mp3/wav, outputs, order, back-to-back stitching, archive checksums, unrelated file untouched, transcript text never in the log); **repeat processing** only new files, old batch untouched; **cancellation** leaves everything unchanged; failure injection (missing enhancement helper) → nothing moved; corrupt model → clean failure; insufficient disk space; leftovers of a crash cleaned and retry; locked original completed on a later run (also with no new recordings); Danish characters `æøå ÆØÅ –` in folder and file names; **engine process killed mid-run** then retry; (slow) 21-minute recording → 10:00 + 10:00 + 1:00 |

Results:

* Linux: 112 passed (the slow 21-minute test is run separately).
* Windows (GitHub Actions run #4): 108 passed, 0 failed, 0 skipped (the VAD regression test was
  added afterwards and runs from run #5 on).

## 3. Windows build and packaged end-to-end verification (GitHub Actions)

Each push runs `.github/workflows/build-windows.yml` on `windows-latest`:

1. Build `arfilter.exe` (Rust, DeepFilterNet 0.5.6 pinned) – cached.
2. Run the full test suite on Windows.
3. Build the portable folder with PyInstaller; packaged `AudioRefineEngine.exe selftest`
   (all native components load: arfilter, NumPy, soundfile, PyAV, CTranslate2, faster-whisper,
   ONNX Runtime, Silero VAD).
4. Start the packaged `AudioRefine.exe` on the native Windows platform, let it scan the test
   folder and save a screenshot (the window shows 3 recordings in natural order with durations
   and formats, the damaged `broken.wma` in red, and the "cannot be read" warning).
5. Packaged first-run download of Whisper large-v3 (3.09 GB, verified; 11–16 s on GitHub's
   network) – cached afterwards.
6. Packaged end-to-end run of the engine on a folder with: Danish FLEURS speech (AAC `.m4a`),
   English FLEURS speech (stereo `.mp3`), a 10.5-minute FLAC in which Danish speech crosses the
   10:00 boundary of the joined recording, a damaged `.wma` and an unrelated `notes.txt`.
7. Automatic verification (`tools/ci/verify_e2e.py`) and upload of the ZIP and of the
   verification material (screenshot, logs, outputs).

Results of run #4 (all checks passed):

| Check | Result |
|---|---|
| One batch folder, no leftover staging folder, batch state `complete` | pass |
| Order `Recording 1.m4a`, `Recording 2.mp3`, `Recording 10.flac` (natural) | pass |
| Originals archived with identical SHA-256; damaged and unrelated files untouched | pass |
| `Audio_001.flac` exactly 28 800 000 samples (10:00.000); total samples = decoded samples (35 449 152) | pass |
| `Audio_002.txt` on the continuous timeline; `Audio_002.srt` relative to its file | pass |
| WER Danish (AAC) | **7.4 %** |
| WER English (MP3) | **16.0 %** (of which most is the very quiet utterance, see below) |
| WER Danish across the 10:00 boundary | **9.8 %** – the sentence crossing 10:00 is complete and split as "… metafor for eksotiske fjerne egne." in `Audio_002.txt` |
| Processing time (12:19 of audio, 2:13 of speech, 4 vCPU) | 4 min 30 s: transcription 227 s (1.7× speech duration), enhancement 29 s |

Defects found and fixed through this run (see git history):

* Run #1: an English sentence at the start of the second recording was silently dropped
  because the 30-second window also contained the end of the Danish recording and was decoded
  as Danish. **Fix:** clips never span a junction between recordings; coverage-repair pass.
* Run #2: Whisper emitted one Danish sentence twice in one segment (28 characters/s).
  **Fix:** duplicated-phrase guard with text-density check (run #4: removed, flagged
  `[uncertain]`).
* Run #4: the English utterance "There are many beaches, due to Auckland's straddling of two
  harbours." is recorded at ≈ −65 dBFS (≈ 40 dB quieter than the rest) and its first half was
  missed by the voice detector. **Fix:** level-normalised VAD input (verified locally: detection
  0.9–1.0; regression test added; run #5 re-verifies on Windows).
* Run #5: the first version of that fix boosted a 10-minute noise-only stretch into "speech";
  Whisper hallucinated "Thank you." ×25 and a fake subtitle credit, which the artefact filter
  missed because of a length rule. **Fix:** VAD gain only where the level stands ≥ 12–18 dB
  above the local noise floor; credit lines matched as prefixes; runs of identical
  low-confidence segments collapsed; three regression tests. This is exactly the kind of
  failure the packaged end-to-end test exists to catch.
* The offscreen Qt platform renders no text on Windows (no font database); the CI screenshot
  now uses the native platform. Real users always use the native platform.

## 4. Real-speech evaluation

### 4.1 Test material

`tools/experiments/make_corpus.py` builds ~4-minute "conversations" from distinct utterances of
the **Google FLEURS** dev split (CC BY 4.0): Danish 19 utterances / 364 words, English 23
utterances / 474 words, separated by 0.4–1.6 s pauses. Each is degraded in reproducible ways,
keeping the identical timeline so that the clean version is a sample-exact reference:

| Condition | Construction |
|---|---|
| clean | speech at −23 dBFS |
| quiet | −35 dB → speech ≈ −58 dBFS, microphone self-noise at −80 dBFS |
| babble 5 dB SNR | 8-talker Danish/English babble (other FLEURS speakers) |
| traffic 0 dB SNR | low-pass brown noise with passing-vehicle swells |
| hum + rumble | 50 Hz mains hum with harmonics (15 dB SNR) + 1/f² rumble (10 dB SNR) |
| far + reverb | synthetic room impulse response RT60 0.8 s, DRR −6 dB, + noise 20 dB SNR |
| speakers 20 dB apart | every other utterance 20 dB quieter, noise floor −60 dBFS |
| sudden bangs | 0.25 s noise bursts at 0.95 peak every ~20 s |
| music 10 dB SNR | synthetic rhythmic harmonic music bed |

Limitations of the material: read speech (not spontaneous conversation), one speaker per
utterance, synthetic noise/reverb, 16 kHz source bandwidth. Results therefore indicate
relative merits of the approaches rather than the absolute accuracy to expect on real
conversations.

Metrics: **WER** after lower-casing and removing punctuation (jiwer), pooled over the
condition's utterances; **ESTOI** (extended short-time objective intelligibility, 0–1) of the
processed signal against the clean reference. All processing uses the product code
(`tools/experiments/run_eval.py` imports `audiorefine.*`). Raw results:
`tools/experiments/results/*.jsonl`.

{{TABLES}}

## 5. Multi-hour test

Packaged engine (Linux build), folder with `Part 1.m4a` (70 min AAC) and `Part 2.mp3`
(50 min MP3), Danish and English speech scattered through room noise; whisper-tiny model
(the large model would have taken several hours on this VM), while other experiments loaded
the same 4-CPU machine.

| Measure | Result |
|---|---|
| Result | success, validation passed, both originals archived |
| Output | 13 FLAC files: 12 × exactly 28 800 000 samples + 1 × 3 072 samples (the AAC file decodes to 70:00.064); total = decoded samples |
| Wall time | 49 min 40 s (enhancement of 2 h audio ≈ 9 min on 4 shared cores) |
| Peak memory of engine + helpers | 846 MB (plus ~3.3 GB for large-v3 in normal use) |
| Peak temporary disk use | 2.5 GB (led to the corrected pre-flight disk-space estimate) |
| Output size | 195 MB for 2 h (≈ 100 MB per hour) |

## 6. Performance and resource figures

| | Figure |
|---|---|
| Whisper large-v3 int8 CPU, 4 vCPU | 0.8–2.1 × the speech duration (VM vs. GitHub runner) |
| Whisper large-v3-turbo int8 CPU, 4 vCPU | ≈ 0.4–0.6 × the speech duration |
| Model RAM (large-v3 int8) | ≈ 3.3 GB resident |
| DeepFilterNet3 (arfilter), per core | real-time factor 0.16 (1 h of audio ≈ 10 CPU-minutes, parallel over cores) |
| Portable ZIP | 156 MB download, 353 MB extracted |
| Model download | 3.09 GB (+ 1.26 GB GPU runtime only on suitable NVIDIA PCs) |

## 7. Known limitations found during testing

* Very short final segment possible (e.g. 64 ms) when the joined recording is a few
  milliseconds longer than a multiple of 10 minutes; required by the "exactly 10 minutes"
  rule.
* Babble/TV-like backgrounds raise WER substantially (≈ 19–29 % at 5 dB SNR).
* Results on run-to-run and machine-to-machine basis are not bit-identical (int8 CPU kernels
  differ between Intel and AMD); word choices can differ slightly.
* GPU acceleration, human listening quality and conversational Danish remain to be verified.
