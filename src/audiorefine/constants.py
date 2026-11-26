"""Fixed engineering decisions. Nothing here is exposed to the user as a setting.

The reasoning behind each value is documented in AUDIO_PROCESSING.md and TECHNICAL_DECISIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- timeline / audio formats
MASTER_SR = 48000          # internal + output sample rate (DeepFilterNet3 native rate)
ASR_SR = 16000             # Whisper input rate; exactly MASTER_SR / 3
DECIMATION = MASTER_SR // ASR_SR
SEGMENT_SECONDS = 600      # 10-minute output segments
SEGMENT_SAMPLES = SEGMENT_SECONDS * MASTER_SR
OUTPUT_EXT = ".flac"       # lossless, ~50 % of WAV size, plays natively on Windows 10/11
OUTPUT_SUBTYPE = "PCM_16"
FRAME = 480                # 10 ms analysis frame at 48 kHz
JUNCTION_FADE_S = 0.005    # 5 ms raised-cosine fade at file junctions (click prevention, no samples removed)

# ---------------------------------------------------------------- folders and names
REFINED_DIR = "refined"
PROCESSED_DIR = "processed"
STAGING_PREFIX = ".incomplete-"
STAGING_MARKER = ".audiorefine-staging"
LOCK_NAME = ".audiorefine.lock"
DETAILS_NAME = "_AudioRefine_details.json"
BATCH_LOG_NAME = "_AudioRefine_log.txt"
COMPLETE_TRANSCRIPT = "Transcript_Complete.txt"
SEGMENT_STEM = "Audio_{:03d}"

# Recognised by extension; actual decodability is verified by probing each file.
AUDIO_EXTENSIONS = frozenset(
    {
        ".wav", ".wave", ".bwf", ".mp3", ".mp2", ".m4a", ".m4b", ".aac", ".adts", ".flac",
        ".ogg", ".oga", ".opus", ".wma", ".asf", ".amr", ".awb", ".3ga", ".3gp", ".aif", ".aiff",
        ".aifc", ".caf", ".mka", ".wv", ".ac3", ".spx", ".weba",
    }
)


# ---------------------------------------------------------------- speech recognition model
@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    repo: str
    revision: str
    files: tuple[ModelFile, ...] = field(default_factory=tuple)

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)

    def url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{filename}"


WHISPER_LARGE_V3 = ModelSpec(
    key="whisper-large-v3",
    display="Whisper large-v3 (CTranslate2)",
    repo="Systran/faster-whisper-large-v3",
    revision="edaa852ec7e145841d8ffdb056a99866b5f0a478",
    files=(
        ModelFile("config.json", 2394, "a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9"),
        ModelFile("preprocessor_config.json", 340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
        ModelFile("tokenizer.json", 2480617, "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca"),
        ModelFile("vocabulary.json", 1068114, "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1"),
        ModelFile("model.bin", 3087284237, "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1"),
    ),
)

WHISPER_LARGE_V3_TURBO = ModelSpec(
    key="whisper-large-v3-turbo",
    display="Whisper large-v3-turbo (CTranslate2)",
    repo="dropbox-dash/faster-whisper-large-v3-turbo",
    revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    files=(
        ModelFile("config.json", 2263, "b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e"),
        ModelFile("preprocessor_config.json", 340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
        ModelFile("tokenizer.json", 2710337, "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"),
        ModelFile("vocabulary.json", 1068114, "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1"),
        ModelFile("model.bin", 1617884929, "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"),
    ),
)

# The model AudioRefine uses. Chosen from the measurements in TEST_REPORT.md.
ACTIVE_MODEL = WHISPER_LARGE_V3
