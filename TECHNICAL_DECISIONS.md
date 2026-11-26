# AudioRefine – Technical decisions

This document records *what* was chosen for AudioRefine 1.0 and *why*, including the
alternatives that were evaluated and rejected. Audio/speech processing details (filters,
parameters, transcription strategy) are in [AUDIO_PROCESSING.md](AUDIO_PROCESSING.md);
measurements are in [TEST_REPORT.md](TEST_REPORT.md).

Priorities, in the order given by the project owner: speech intelligibility → transcription
accuracy → preservation of originals → simplicity → performance.

---

## 1. Architecture overview

```
AudioRefine.exe  (PySide6 window, no console)
   │  starts as a child process, reads JSON lines from stdout, writes "cancel" to stdin
   ▼
AudioRefineEngine.exe  (same Python runtime, console subsystem; also usable from the command line)
   ├─ discovery / probing / ordering ...... PyAV (FFmpeg libraries)
   ├─ decoding + resampling to 48 kHz ..... PyAV / swresample
   ├─ 48 kHz → 16 kHz decimation .......... own linear-phase FIR (NumPy)
   ├─ voice activity detection ............ Silero VAD v6 (ONNX Runtime)
   ├─ analysis, leveller, limiter ......... NumPy
   ├─ high-pass / hum notch / DeepFilterNet3 / EQ ... arfilter.exe (own Rust program, streamed)
   ├─ speech recognition .................. faster-whisper → CTranslate2, Whisper large-v3
   ├─ FLAC writing ......................... soundfile (libsndfile)
   └─ staging / commit / archive / journal  plain file-system operations (batch.py)
```

The window and the engine are separate processes on purpose:

* **Responsiveness** – the UI thread never does audio work.
* **Crash isolation** – a crash in a native library (GPU driver, decoder) ends the engine, not
  the window; the window reports it and, if it happened while loading the model on the GPU,
  retries once on the CPU.
* **Safe, immediate cancellation** – the engine stops cooperatively; if it does not within
  30 s it is killed. Because all output is staged and originals are only moved at the very end,
  killing the engine at any moment is safe.
* **Testability / scripting** – the identical engine runs headless in CI and from a command
  prompt (`AudioRefineEngine.exe process <folder>`).

Both executables live in one PyInstaller *one-folder* bundle and share the same runtime (no
duplication).

## 2. Language and UI toolkit: Python 3.13 + PySide6 (Qt 6)

**Chosen:** Python for the engine and the UI, Qt via PySide6 (LGPL) for a native-looking window,
one Rust helper for the neural noise suppressor.

Reasons:
* The best locally executable speech stack with *integrated, well-tested long-form handling*
  (VAD, temperature fallback, compression-ratio and log-prob checks, word timestamps,
  hallucination-silence handling) is faster-whisper/CTranslate2, which is a Python library.
* The exact code that ships was the code that was measured: the evaluation harness
  (`tools/experiments`) imports the product modules, so the reported numbers describe the product.
* One language for engine and UI keeps the code base small and maintainable by one person
  plus AI collaborators. PySide6 provides proper native dialogs (folder picker), high-DPI
  support, and a clean look with very little code.

**Considered and rejected:**
* *C# / WPF + whisper.cpp (Whisper.net)* – attractive (small, native, Vulkan GPU support for
  NVIDIA/AMD/Intel). Rejected for 1.0 because the speech-enhancement stack and the
  measurement harness would have had to be split across two languages, whisper.cpp's long-form
  VAD/hallucination handling is less mature than faster-whisper's, and none of the WPF part
  could be executed or tested in the Linux build environment used for development. It remains a
  good candidate if AMD/Intel GPU acceleration becomes important (see §6).
* *WPF front-end + Python engine* – two toolchains for little gain; the PySide6 window is
  sufficient for a one-button utility.
* *Tkinter* – smaller, but noticeably dated look, weak high-DPI handling.
* *Electron/Tauri* – large runtime or a second toolchain for a 1-screen utility.

## 3. Speech recognition: faster-whisper + Whisper large-v3

**Chosen:** `faster-whisper 1.2.1` on `CTranslate2 4.9.1`, model **Whisper large-v3**
(`Systran/faster-whisper-large-v3`, MIT licence, pinned to revision `edaa852e…`), int8 on CPU,
float16 (or int8_float16 below 6 GB VRAM) on NVIDIA GPUs.

Selection criteria and how they were weighed:

| Criterion | Assessment |
|---|---|
| Accuracy (priority) | Whisper large-v3 is the most accurate openly available multilingual model that runs locally with mature tooling. Measured on our corpus: Danish read speech 6.6 % WER, English 5.3 % (TEST_REPORT.md). |
| Danish | Supported natively (Whisper's Danish training data is substantial). large-v3 beats large-v3-turbo on Danish in our measurements (TEST_REPORT.md). Danish-specific fine-tunes exist (see below). |
| Timestamps | Segment timestamps + DTW word timestamps; word-level timestamps are needed for splitting at 10-minute boundaries and for SRT cues. |
| Noisy / quiet speech | Whisper was trained on 5M hours of weakly supervised, often noisy audio; it is robust without front-end enhancement (confirmed by our measurements). Quiet speech is handled by level normalisation before recognition. |
| CPU | int8 CTranslate2: ~0.8× real-time of the *speech* duration on the 4-core test VM, faster on typical 8-core desktops; ~3.3 GB RAM. |
| GPU | CUDA via CTranslate2 when an NVIDIA RTX 30/40/50 GPU is present. |
| Deployment | Pure wheels; the model is a 3.1 GB download on first use, verified by SHA-256. |

**large-v3-turbo** (4 decoder layers, ~2.3× faster on CPU here) was evaluated as the default and
as a CPU-only fallback. It was *not* chosen because accuracy is the priority and it was
measurably worse on Danish (TEST_REPORT.md). Keeping one model also keeps behaviour identical
on every PC.

**Considered and rejected or deferred:**
* *openai-whisper (PyTorch)* – same model, but slower and 2–3 GB of PyTorch/CUDA in the bundle.
* *whisper.cpp* – see §2.
* *Danish fine-tunes of Whisper (e.g. "hviske", CoRal project models)* – reported to reduce
  Danish error rates substantially, but: (a) fine-tunes are often trained without timestamp
  tokens and lose reliable timestamps, (b) English and code-switching degrade, (c) licences and
  training data terms vary (the CoRal dataset uses an OpenRAIL-style licence), and (d) it would
  mean a second multi-GB model plus routing logic. Top candidate for a future, measured upgrade.
* *NVIDIA Canary / Parakeet* – strong ASR models, Canary-1B-v2 covers Danish; require NeMo /
  PyTorch, much heavier deployment, less proven long-form timestamping. Deferred.
* *wav2vec2-style Danish models* – no punctuation/casing, weaker on conversational audio,
  no robust long-form pipeline.
* *Cloud services* – excluded by requirement (privacy).

## 4. Speech enhancement: DeepFilterNet3 (bounded) + conventional DSP

**Chosen:** a conservative chain of high-pass, (conditional) mains-hum notches,
**DeepFilterNet3** with an SNR-dependent attenuation limit, a gentle presence EQ, a speech-aware
leveller and a look-ahead peak limiter. Full rationale and parameters: AUDIO_PROCESSING.md.

DeepFilterNet3 (MIT/Apache-2.0) runs through its own Rust/tract implementation (libDF 0.5.6),
compiled into a small helper program `arfilter.exe` that streams samples through stdin/stdout.
This avoids PyTorch entirely (the Python DeepFilterNet package needs it), gives deterministic
sample-exact delay compensation, and lets several chunks be processed in parallel. The helper
is built from pinned source (git tag `v0.5.6`) in CI.

Rejected for the listening path: RNNoise / FFmpeg `arnndn` (older, more artefacts, cruder
gain model), FFmpeg `afftdn`/`anlmdn` (stationary-noise oriented, musical noise), spectral
gating (`noisereduce`), Demucs/denoiser and MetricGAN+ (PyTorch, heavier, artefacts), and all
*generative* restoration (VoiceFixer, Resemble Enhance, etc.), because they synthesise speech,
which the requirements forbid.

## 5. Decoding: PyAV (FFmpeg libraries) instead of an `ffmpeg.exe`

PyAV is already a dependency of faster-whisper and ships LGPL FFmpeg shared libraries with all
relevant decoders (AAC, MP3, Opus, Vorbis, FLAC, WMA 1/2/Pro/Voice, AMR-NB/WB, ALAC, PCM
variants, ADPCM, GSM, AC-3, Speex, WavPack…). Decoding in-process with packet-level error
handling lets AudioRefine skip damaged packets, report them, and continue, which an external
`ffmpeg.exe` would make harder to control. Bundling a separate `ffmpeg.exe` would add ~100 MB
for nothing.

## 6. GPU acceleration

* Detection: `nvidia-smi` (installed with every NVIDIA driver) → name, VRAM, driver,
  compute capability. Candidate if VRAM ≥ 4 GB and compute capability ≥ 8.6 (the CUDA
  architectures compiled into CTranslate2 4.9.1: sm_86, sm_100, sm_120 → RTX 30/40/50).
* CUDA runtime: CTranslate2's Windows wheel does not contain cuBLAS 12 or the cuDNN 9
  sub-libraries. AudioRefine downloads NVIDIA's official redistributable wheels from PyPI on
  first use (cuBLAS 12.8.4.1, cuDNN 9.10.2.21; 1.26 GB, SHA-256 pinned) and extracts only the DLLs.
  Not bundling them keeps the ZIP small for the majority without a suitable GPU, and avoids
  redistributing NVIDIA binaries.
* Use: the model is loaded on the GPU and a short test transcription is run; any exception →
  CPU. A hard crash of the engine while loading → the window retries once with `--cpu`.
* AMD and Intel GPUs are not accelerated (CTranslate2 has no ROCm/oneAPI build on Windows).
  whisper.cpp with Vulkan would cover them; noted as future work.
* **Status: GPU acceleration is implemented but unverified** – no NVIDIA GPU was available to
  the development or CI environment.

## 7. Output format: FLAC, 48 kHz, 16-bit, mono

* **Lossless** (no generational loss), roughly 50–60 % of WAV size: a 2-hour batch is about
  0.35–0.45 GB instead of 0.69 GB WAV.
* Plays natively on Windows 10/11 (Media Player, Groove/Media Player app), VLC, Android, iOS
  Files app, all major audio editors.
* **48 kHz** keeps everything the sources contain (DeepFilterNet's native rate; no further
  resampling in the listening path). Narrow-band sources (AMR 8 kHz) are not artificially
  band-limited further – nothing is band-limited to telephone bandwidth.
* **16-bit with TPDF dither**: after levelling, speech sits around −20 dBFS, leaving > 70 dB
  above the quantisation floor; 24-bit would add ~30 % size for no audible benefit.
* **Mono**: sources are mixed from different devices into one timeline; mono avoids layout
  changes between files. Loss: spatial cues of genuine stereo recordings (future option).
* WAV was evaluated (maximum compatibility) but rejected because of twice the size for
  multi-hour batches; nothing in the requirements needs WAV specifically.

## 8. File safety design

See README "How your files are handled" and `batch.py`. Key decisions:

* **Batch folders** (`refined/<YYYY-MM-DD_HH-MM-SS>`, `processed/<same>`) instead of numbered
  file names: never collides, keeps runs separate, and the pairing of outputs and originals is
  obvious. Collisions (two runs in the same second) get `_2`, `_3`.
* **Staging + single atomic rename** for outputs; **move (rename) with SHA-256 verification
  before and after** for originals; cross-volume moves fall back to copy → verify → delete.
* **Journal** (`_AudioRefine_details.json`) updated after every archived file; incomplete
  archiving is detected on the next scan, the affected originals are excluded from new
  batches and moved first when the user processes again.
* **Folder lock** with stale-lock detection (PID check) prevents concurrent runs.
* **Stale staging folders** (from crashes) are removed only if they carry AudioRefine's marker
  file.
* Originals are hashed before processing and re-hashed before archiving; a recording that
  changed during processing (e.g. still being copied) aborts the run.
* Pre-flight disk-space check, clear messages for permission/locked/disk-full errors.

## 9. Recording order

Natural (Explorer-like) file-name order by default; chronological by file-name timestamps when
*all* names carry one. Embedded metadata only produces a warning; file-system times are never
used. Rationale in README "How the order is decided" and `ordering.py`.

## 10. Packaging and distribution

* PyInstaller 6.22 one-folder build (`packaging/audiorefine.spec`), no UPX (fewer antivirus
  false positives), Windows version resources and icon. ~250–300 MB unpacked (dominated by
  Qt, FFmpeg, CTranslate2, ONNX Runtime).
* Data in `%LOCALAPPDATA%\AudioRefine` (models survive upgrades); portable mode when a `data`
  folder exists next to the executable.
* GitHub Actions (`windows-latest`): build Rust helper → run full test suite on Windows
  (incl. real-speech pipeline tests with whisper-tiny) → PyInstaller → packaged self-test →
  packaged GUI smoke test (offscreen screenshot) → packaged first-run model download →
  packaged end-to-end processing of real Danish/English speech with large-v3 and automatic
  verification (order, exact segment lengths, checksums of archived originals, untouched
  unrelated/damaged files, timeline formats, word error rate) → upload artifact.
* Not code-signed (no certificate available); SmartScreen will warn on first start.

## 11. Licensing review

All bundled components are under permissive licences (MIT, BSD, Apache-2.0, PSF) or LGPL used
as unmodified shared libraries (Qt/PySide6, FFmpeg via PyAV, libsndfile), which permits
distribution in a portable ZIP alongside the notices in `THIRD_PARTY_NOTICES.txt`. The Whisper
large-v3 weights are MIT. The NVIDIA CUDA libraries are not redistributed by AudioRefine; they
are fetched from NVIDIA's PyPI packages on the user's machine. DeepFilterNet (MIT/Apache-2.0)
model weights are compiled into `arfilter.exe`. Silero VAD is MIT. No component restricts
commercial use. FLEURS test audio in the repository is CC BY 4.0 with attribution in
`tests/data/README.md`.
