"""Inspecting candidate recordings (container, codec, duration, embedded recording time)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .ordering import parse_filename_datetime, parse_metadata_datetime


@dataclass
class SourceInfo:
    path: Path
    name: str
    size: int
    mtime: float
    birthtime: float | None = None   # file-system creation time where the OS provides it (Windows)
    ok: bool = False
    error: str = ""
    container: str = ""
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    bit_rate: int = 0
    duration: float = 0.0            # estimate from the container (exact value known after decoding)
    meta_datetime: datetime | None = None
    meta_datetime_tag: str = ""
    name_datetime: datetime | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        d["meta_datetime"] = self.meta_datetime.isoformat() if self.meta_datetime else None
        d["name_datetime"] = self.name_datetime.isoformat() if self.name_datetime else None
        return d


_DATE_TAGS = ("com.apple.quicktime.creationdate", "creation_time", "date", "origination_date",
              "date_recorded", "recording_time", "tdrc", "tdat", "icrd", "year")


def probe(path: Path, check_decode: bool = True) -> SourceInfo:
    st = path.stat()
    info = SourceInfo(path=path, name=path.name, size=st.st_size, mtime=st.st_mtime,
                      birthtime=getattr(st, "st_birthtime", None))
    info.name_datetime = parse_filename_datetime(path.stem)
    if st.st_size == 0:
        info.error = "The file is empty."
        return info
    try:
        import av
    except Exception as e:  # pragma: no cover - packaging error
        info.error = f"Audio decoder unavailable: {e}"
        return info
    try:
        with av.open(str(path)) as c:
            videos = [s for s in c.streams.video if not (s.disposition & av.stream.Disposition.attached_pic)]
            audios = list(c.streams.audio)
            if not audios:
                info.error = "No audio stream found."
                return info
            if videos:
                info.error = "Contains video; only audio recordings are processed."
                return info
            s = audios[0]
            info.container = c.format.name or ""
            info.codec = s.codec_context.name or ""
            info.sample_rate = int(s.rate or s.codec_context.sample_rate or 0)
            info.channels = int(s.channels or 0)
            info.bit_rate = int(s.bit_rate or c.bit_rate or 0)
            if s.duration is not None and s.time_base:
                info.duration = float(s.duration * s.time_base)
            elif c.duration:
                info.duration = c.duration / 1_000_000
            tags = {k.lower(): v for k, v in (c.metadata or {}).items()}
            tags.update({k.lower(): v for k, v in (s.metadata or {}).items() if k.lower() not in tags})
            for t in _DATE_TAGS:
                if t in tags:
                    dt = parse_metadata_datetime(str(tags[t]))
                    if dt is not None:
                        info.meta_datetime, info.meta_datetime_tag = dt, t
                        break
            if check_decode:
                got = 0
                for frame in c.decode(s):
                    got += frame.samples
                    if got >= info.sample_rate * 2 or got > 200_000:
                        break
                if got == 0:
                    info.error = "No decodable audio data (the file may be damaged)."
                    return info
            if info.sample_rate <= 0:
                info.error = "Unknown sample rate."
                return info
            info.ok = True
    except Exception as e:
        info.error = _friendly_error(e)
    return info


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "invalid data" in low or "moov atom" in low:
        return "The file is damaged or not a supported audio format."
    if "permission" in low or isinstance(e, PermissionError):
        return "The file cannot be opened (it may be in use or access is denied)."
    return f"Cannot read the file ({type(e).__name__}: {msg[:160]})."


def file_fingerprint(path: Path) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime": st.st_mtime}


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
