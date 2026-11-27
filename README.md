# AudioRefine

**One folder. One button. Clean audio. Timestamped transcripts.**

AudioRefine is a small Windows utility that turns a folder of raw recordings (phone recordings,
dictaphone files, voice memos …) into speech-enhanced audio and timestamped transcripts,
**organised by the date and clock hour in which the recordings were made**:

* it works through the recordings **one clock hour at a time, oldest first**, and finishes each
  hour completely before starting the next – you can listen to and read the first hours while
  later ones are still being processed;
* each hour gets enhanced audio in exact **10-minute FLAC files**, a transcript and subtitle file
  for each of them, and **`Transcript_Hour_Complete.txt`**;
* each day gets **`Transcript_Day_Complete.txt`**;
* the `refined` folder holds **`Transcript_Master_Complete.txt`** – everything ever transcribed
  in this folder, in chronological order, ready to read or to upload to ChatGPT or Claude. It is
  updated after every finished hour and grows when you add more recordings later;
* original recordings are moved, unchanged, into `processed\<date>\<hour>` once every part of
  them has been processed.

Everything runs locally on your computer. Recordings are never uploaded anywhere.
Danish and English are recognised automatically (also mixed in the same batch).

---

## Getting started

1. Download the portable ZIP (`AudioRefine-1.0.0-win-x64`) and extract it anywhere,
   e.g. to `C:\Tools\AudioRefine`.
2. Start **`AudioRefine.exe`**.
3. Click **Browse…** and choose the folder with your recordings
   (or drag the folder onto the window).
4. Check the list: it shows each recording's detected date and time and the hour it belongs to.
5. Click **Process folder**. The window shows the current day and hour, how many hours are done,
   and the overall progress. **Open master transcript** becomes available after the first hour.

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

Given this folder (recordings from two days):

```
C:\Audio dumps\
    20251125_080412.m4a      (recorded 08:04)
    20251125_083000.m4a      (08:30)
    20251125_085500.m4a      (08:55, 20 minutes long – runs into the 09:00 hour)
    20251125_103200.m4a      (10:32)
    20251126_071200.m4a      (next day, 07:12)
```

after processing:

```
C:\Audio dumps\
    processed\
        2025-11-25\08-00\   20251125_080412.m4a, 20251125_083000.m4a, 20251125_085500.m4a
        2025-11-25\10-00\   20251125_103200.m4a
        2025-11-26\07-00\   20251126_071200.m4a
    refined\
        Transcript_Master_Complete.txt        ← all days and hours, chronological
        2025-11-25\
            Transcript_Day_Complete.txt
            08-00\
                Audio_001.flac  Audio_001.txt  Audio_001.srt    (first 10 minutes of this hour's audio)
                Audio_002.flac  Audio_002.txt  Audio_002.srt
                Transcript_Hour_Complete.txt
                _AudioRefine_hour.json                           (technical record of the hour)
            09-00\                                              (the part of the 08:55 recording after 09:00)
            10-00\
        2025-11-26\
            Transcript_Day_Complete.txt
            07-00\
```

* Folders exist only for hours that contain recordings (`10-00` but no empty `09-00`… unless a
  recording runs into it).
* Inside an hour, the recordings are joined back-to-back in recording order and cut into exact
  10-minute files; the last file of an hour may be shorter. No silence is added to fill an hour,
  and audio of different hours is never mixed in one file.
* A recording that runs across a full hour (or midnight) is split **virtually** at the boundary:
  the part before 09:00 is in `08-00`, the rest in `09-00`. The original file is never cut; it is
  archived under the hour in which it **started**, and only after all its parts are finished.

### Transcript formats

`Transcript_Hour_Complete.txt` and `Audio_NNN.txt`:

```
Recording date: 2025-11-25
Recording hour: 08:00–09:00

── 20251125_080412.m4a begins (2025-11-25 08:04:12) ──

[2025-11-25 08:04:15 | Audio_001 00:00:03]
Det var egentlig det, jeg prøvede at forklare dig.

[2025-11-25 08:04:19 | Audio_001 00:00:07] [uncertain]
Ja, det kan jeg godt forstå.
```

* Every entry shows the **recording date and clock time** and the audio file with the position
  inside it. Clock times are computed per recording (gaps between recordings are respected).
* `≈2025-11-26 14:32:19` – a `≈` marks times that had to be **estimated** from the file's dates
  because neither the file name nor the file itself says when it was recorded.
* `.srt` subtitle times count from the start of *their own* audio file, so they play in sync in
  VLC.
* A sentence that crosses an hour boundary or a 10-minute file boundary is split at the word
  where the boundary falls; the halves are marked with `…`. No word is lost or repeated.
* `[uncertain]` marks passages the recogniser was not confident about; `[unclear speech]` marks
  places where someone spoke but no words could be recognised. The transcript is automatic and
  has not been checked by a person; it is not a certified or forensic record.

`Transcript_Day_Complete.txt` contains all finished hours of the day under hour headings;
`Transcript_Master_Complete.txt` contains all finished days and hours under date and hour
headings, with a status line at the top: **in progress**, **interrupted** (process the folder
again to continue) or **complete**. Both are rebuilt from the hour records after every finished
hour – nothing is transcribed twice.

---

## How the recording time is determined

AudioRefine uses the most reliable information available, in this order:

1. **Date and time in the file name** (e.g. `20251125_080412.m4a`, `2025-11-25 08.04.12.m4a`,
   `Recording 2025-11-25 at 08.04.12.m4a`) – written by the recorder, survives copying.
2. **Recording time stored inside the file** (most phones and recorders write one).
3. **File creation time** – only if it matches the modification time minus the recording length
   (i.e. it really is when the recording started, not when the file was copied). *Estimated.*
4. **File modification time minus the recording length** (a file is last written when the
   recording stops). *Estimated* – copying from a phone can change these dates.

Estimated times are marked `≈` in the window and in every transcript. The time at which you run
AudioRefine is never used. Times are shown in this PC's local time zone; daylight-saving changes
are handled by Windows' time-zone rules.

The list in the window shows each recording's detected time and hour. If a time is wrong, the
simplest fix is to rename the file with its correct date and time (`YYYYMMDD_HHMMSS`).

---

## Interruptions, adding recordings later, processing again

* Every finished hour is a checkpoint. If you press **Cancel**, close the window, or the PC
  crashes, finished hours, the day transcripts and the master transcript stay complete and
  usable; the hour being worked on is discarded and simply redone next time. Process the same
  folder again and AudioRefine continues with the remaining hours.
* Add new recordings to the folder at any time and press **Process folder** again. Only the
  affected hours are processed: a recording for an hour that is already finished is merged into
  that hour (the existing audio is reused bit-for-bit and its transcript is kept, only the new
  recording is enhanced and transcribed), recordings from older dates are placed in their
  correct chronological position, and the master transcript is regenerated.
* Processing a folder with nothing new does nothing. A copy of a recording that was already
  processed is recognised by its content, reported and left in the folder.

## How your files are handled (safety)

* Original recordings are **never modified**. They are only read.
* Each hour is built in a hidden temporary folder (`refined\<date>\.incomplete-HH-00`) and appears
  under its final name only when all its audio files and transcripts were written **and
  verified**. Updating an existing hour uses a recorded, crash-safe swap.
* An original is moved to `processed\<date>\<hour>` only after **every** part of it (a long
  recording can span several hours) is finished. Each move is verified with SHA-256 before and
  after. Nothing is ever overwritten – a different recording with the same name gets a
  numbered name.
* If an original cannot be moved (for example because a media player has it open), it stays in
  the folder, is **not** processed again, and is moved the next time you click Process.
* Files that are not audio, damaged audio files and subfolders are left alone. Damaged files are
  listed in red with the reason.
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
* **Folders processed with AudioRefine 1.0** (batch folders such as `refined\2026-11-26_14-35-02`)
  are left as they are; they are not part of the master transcript.
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
