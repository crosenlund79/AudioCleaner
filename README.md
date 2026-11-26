# AudioRefine

**One folder. One button. Clean audio. Timestamped transcripts.**

AudioRefine is a small Windows utility that turns a folder of raw recordings (phone recordings,
dictaphone files, voice memos …) into:

* speech-enhanced audio, joined into one continuous recording and cut into exact
  **10-minute FLAC files** (`Audio_001.flac`, `Audio_002.flac`, …),
* a **timestamped transcript** and an **SRT subtitle file** for every 10-minute file,
* one **`Transcript_Complete.txt`** with the whole conversation in chronological order, ready to
  read or to paste into ChatGPT or Claude,

and then moves the original recordings, unchanged, into a `processed` folder.

Everything runs locally on your computer. Recordings are never uploaded anywhere.
Danish and English are recognised automatically (also mixed in the same batch).

---

## Getting started

1. Download the portable ZIP (`AudioRefine-1.0.0-win-x64`) and extract it anywhere,
   e.g. to `C:\Tools\AudioRefine`.
2. Start **`AudioRefine.exe`**.
3. Click **Browse…** and choose the folder with your recordings
   (or drag the folder onto the window).
4. Check the list: it shows the recordings in the order they will be joined.
5. Click **Process folder**.

The **first time**, AudioRefine asks to download its speech-recognition model
(Whisper large-v3, 3.1 GB; plus 1.3 GB of acceleration files if you have a suitable NVIDIA
graphics card). The download shows its progress, resumes if interrupted, is verified with
checksums and is reused afterwards. After that, no internet connection is needed.

Windows SmartScreen may warn about an unknown publisher the first time, because the program is
not code-signed. Choose *More info → Run anyway*.

### System requirements

| | Minimum | Recommended |
|---|---|---|
| Windows | 10 or 11, 64-bit | |
| Memory | 8 GB RAM | 16 GB RAM |
| Disk | 4 GB for the program and model, plus temporary space of about 1.3 GB per hour of audio (2.5 GB per hour for a single very long recording), and ~0.2 GB per hour for the results | SSD |
| Processor | any 64-bit CPU with AVX2 (2013 or newer) | 8+ cores |
| Graphics (optional) | | NVIDIA GeForce RTX 30/40/50 series with ≥ 4 GB (used automatically; see note) |

GPU note: acceleration is used only if a short test transcription on the graphics card
succeeds; otherwise the CPU is used automatically. GPU acceleration has not yet been verified on
real hardware (see TEST_REPORT.md).

Processing time is dominated by speech recognition, which only runs on the parts that contain
speech. Measured on CPU: 1.1–1.7× the speech duration on a 4-core cloud machine; expect roughly
0.5–1× on a modern 8-core desktop CPU (1 hour of conversation ≈ 30–60 minutes of processing). A supported
NVIDIA GPU is typically 10–20× faster. Speech enhancement adds about 5 % of the audio duration.

---

## What you get

Given this folder:

```
C:\Audio dumps\
    Recording001.m4a
    Recording002.m4a
    Recording003.wav
```

after processing:

```
C:\Audio dumps\
    processed\
        2026-11-26_14-35-02\
            Recording001.m4a          ← originals, moved unchanged (verified by checksum)
            Recording002.m4a
            Recording003.wav
    refined\
        2026-11-26_14-35-02\
            Audio_001.flac            ← 00:00:00 – 00:10:00
            Audio_001.txt
            Audio_001.srt
            Audio_002.flac            ← 00:10:00 – 00:20:00
            Audio_002.txt
            Audio_002.srt
            Audio_003.flac            ← 00:20:00 – 00:24:17 (last file may be shorter)
            Audio_003.txt
            Audio_003.srt
            Transcript_Complete.txt
            _AudioRefine_details.json ← technical record (order, positions, checksums, settings used)
            _AudioRefine_log.txt      ← diagnostic log of this run (contains no transcript text)
```

Each processing run gets its own **batch folder** named after the date and time it started, in
both `refined` and `processed`. If you process the same folder again later, only new recordings
lying directly in the folder are processed; earlier batches are never touched.

### Transcript format

`Audio_002.txt`:

```
Recording timeline: 00:10:00 – 00:20:00
Audio file: Audio_002.flac
Source recordings: Recording002.m4a, Recording003.wav

[00:10:04.120 – 00:10:08.420]
Det var egentlig det, jeg prøvede at forklare dig tidligere.

── Recording003.wav begins at 00:15:20.000 ──

[00:15:22.810 – 00:15:25.320] [uncertain]
Ja, det kan jeg godt forstå.
```

* Times in `.txt` files and `Transcript_Complete.txt` count from the start of the whole
  (joined) recording; they do not restart for each 10-minute file.
* Times in `.srt` files count from the start of *their own* audio file, so subtitles play in
  sync when you open e.g. `Audio_002.flac` with `Audio_002.srt` in VLC.
* A sentence that crosses a 10-minute boundary is split at the word where the boundary falls;
  the two halves are marked with `…`. `Transcript_Complete.txt` keeps it in one piece.
* `[uncertain]` marks passages the recogniser was not confident about.
  `[unclear speech]` marks places where someone spoke but no words could be recognised.
  The transcript is automatic and has not been checked by a person; it is not a certified or
  forensic record.
* `── Recording003.wav begins at … ──` shows where each original recording starts, so you can
  go back to the source.

---

## How the order is decided

AudioRefine joins the recordings in this order:

1. If **every** file name contains a date and time (e.g. `20240315_143022.m4a`,
   `2024-03-15 14.30.22.m4a`, `Recording 2024-03-15 at 14.30.22.m4a`), by that date and time.
2. Otherwise by **file name, sorted naturally** — the same order Windows Explorer shows when
   sorting by name: `Recording2` comes before `Recording10`.

Dates stored inside the files are only used to warn you when they disagree with this order
(devices write them inconsistently), and file-system dates are never used (copying changes them).
The order is shown before you press Process. To change it, rename the files.

Recordings are joined back-to-back: no silence is inserted between them and nothing is removed,
so timestamps are the elapsed time within the joined recording, not clock time.

---

## How your files are handled (safety)

* Original recordings are **never modified**. They are only read.
* All results are first written to a hidden temporary folder (`refined\.incomplete-…`). Only
  when every audio file and transcript has been written **and verified**, the folder is renamed
  to its final name in one step.
* Only after that are the originals moved to `processed\<batch>`. Each move is verified with a
  SHA-256 checksum. A file is never overwritten.
* If anything fails, or you press **Cancel**, or the PC crashes: your recordings stay where they
  were and nothing that looks like a finished result is left behind. Just process again.
* If an original cannot be moved (for example because a media player has it open), the results
  are still kept, the file stays where it is, it is **not** processed again, and it is moved the
  next time you click Process.
* Files that are not audio, damaged audio files and subfolders are left alone. Damaged files
  are listed in red with the reason.
* Two AudioRefine windows cannot process the same folder at the same time.

---

## Supported formats

WAV, MP3, M4A/AAC, FLAC, OGG (Vorbis/Opus), OPUS, WMA, AMR (AMR-NB/WB, `.amr`, `.3ga`, `.3gp`),
AIFF, CAF, MKA, WavPack, AC-3, Speex — any sample rate, mono or stereo or more channels, any
bit rate. Files with video are skipped. Output is always 48 kHz 16-bit mono FLAC
(lossless, about 3–5 MB per minute; plays in Windows Media Player, VLC and most players).

---

## Troubleshooting

* **Logs:** click **Logs** in the window. Each run writes a log file (also copied into the batch
  folder as `_AudioRefine_log.txt`). Logs contain technical information only — never the
  transcribed words — and can be shared when reporting a problem.
* **Data location:** models, logs and temporary files are stored in
  `%LOCALAPPDATA%\AudioRefine`. To keep everything next to the program instead (fully portable),
  create an empty folder named `data` next to `AudioRefine.exe`.
* **Command line:** `AudioRefineEngine.exe process "C:\Audio dumps"` processes a folder without the
  window (`AudioRefineEngine.exe --help` lists all commands).
* **Uninstall:** delete the program folder and `%LOCALAPPDATA%\AudioRefine`.

---

## For developers

* Technology choices: [TECHNICAL_DECISIONS.md](TECHNICAL_DECISIONS.md)
* Audio and speech processing in detail: [AUDIO_PROCESSING.md](AUDIO_PROCESSING.md)
* Tests and measured results: [TEST_REPORT.md](TEST_REPORT.md)

```bash
# development setup (Python 3.13, Rust stable)
cargo build --release --manifest-path native/arfilter/Cargo.toml
pip install -r requirements-dev.txt
PYTHONPATH=src python -m audiorefine            # desktop window
PYTHONPATH=src python -m audiorefine.cli --help # engine
AUDIOREFINE_TEST_MODEL=/path/to/faster-whisper-tiny python -m pytest
pyinstaller packaging/audiorefine.spec --noconfirm
```

The GitHub Actions workflow `.github/workflows/build-windows.yml` builds the helper, runs the
test suite on Windows, packages the portable folder, performs a packaged end-to-end run on real
Danish and English speech and publishes the ZIP as a build artifact.

Licences of bundled components: [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).
AudioRefine itself is MIT-licensed.
