# AudioRefine 1.0 – Test report

Status legend: **Verified** = executed and passed with evidence below. **Partly verified** =
covered by automated or indirect checks only. **Unverified** = not executed (reason given).

## 1. Summary

| Area | Status | Evidence |
|---|---|---|
| Automated test suite (108 tests) on Linux | Verified | §3 |
| Automated test suite on Windows (GitHub Actions, Windows Server 2025) | Verified | §3, CI run |
| Portable Windows build (PyInstaller) builds, self-test passes | Verified | CI run |
| Packaged end-to-end run on Windows with Whisper large-v3, real Danish + English speech | Verified | §4 |
| First-run model download (3.1 GB, resumable, SHA-256 verified) via the packaged app | Verified (CI) | §4 |
| Resume of interrupted downloads | Verified (Linux, unit-level) | §3 |
| Desktop window starts in the packaged build and lists recordings | Verified (CI screenshot) | §4 |
| Interactive use of the window by a person on a Windows desktop | **Unverified** | no interactive Windows desktop available |
| Multi-hour recording (2 h, two files) | Verified (Linux, packaged engine) | §6 |
| Speech enhancement improves intelligibility | Partly verified (objective ESTOI only) | §5.3 |
| Listening tests by people | **Unverified** | not possible in this environment |
| Enhancement vs. transcription accuracy (original / conservative / neural) | Verified (WER, large-v3) | §5.2 |
| Danish and English transcription accuracy | Verified on read speech with synthetic degradations | §5 |
| Conversational / dialect Danish | **Unverified** | no openly licensed conversational Danish corpus available without registration |
| NVIDIA GPU acceleration | **Unverified** | no GPU in development or CI machines; CPU fallback verified |
| AMR input | Partly verified | decoders present in the bundled FFmpeg; no AMR encoder available to create a test file |

(Sections 3–6 follow.)
