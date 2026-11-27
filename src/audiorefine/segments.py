"""Exact 10-minute segmentation of the stitched master audio into FLAC files."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .constants import MASTER_SR, OUTPUT_EXT, SEGMENT_SAMPLES, SEGMENT_STEM
from .rawaudio import iter_raw, raw_length
from .transcript import SegmentFile


def plan_segments(total_samples: int, segment_samples: int = SEGMENT_SAMPLES) -> list[SegmentFile]:
    """Every segment is exactly ``segment_samples`` long except the last, which may be shorter."""
    files = []
    n = max(1, -(-total_samples // segment_samples))
    for i in range(n):
        a = i * segment_samples
        b = min(total_samples, a + segment_samples)
        stem = SEGMENT_STEM.format(i + 1)
        files.append(SegmentFile(index=i + 1, stem=stem, audio_name=stem + OUTPUT_EXT, start=a / MASTER_SR,
                                 end=b / MASTER_SR, start_sample=a, end_sample=b))
    return files


def _to_int16_dithered(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    tpdf = rng.random(x.size) - rng.random(x.size)  # triangular, +-1 LSB
    q = np.round(x.astype(np.float64) * 32767.0 + tpdf)
    return np.clip(q, -32768, 32767).astype(np.int16)


def write_segments(master: Path, out_dir: Path, files: list[SegmentFile], progress: Callable[[float], None] | None = None,
                   cancel: Callable[[], None] | None = None,
                   exact_ranges: list[tuple[int, int]] | None = None) -> list[dict]:
    """Write the FLAC segments. Samples inside ``exact_ranges`` already are 16-bit values (audio
    reused from an earlier output) and are written back unchanged, without new dither."""
    total = raw_length(master)
    exact_ranges = sorted(exact_ranges or [])
    rng = np.random.default_rng(1234567)
    results = []
    for f in files:
        path = out_dir / f.audio_name
        tmp = path.with_name(path.stem + ".partial" + path.suffix)
        h = hashlib.sha256()
        count = 0
        with sf.SoundFile(tmp, "w", samplerate=MASTER_SR, channels=1, subtype="PCM_16", format="FLAC") as out:
            for pos, x in iter_raw(master, MASTER_SR * 20, f.start_sample, f.end_sample):
                if cancel:
                    cancel()
                q = _to_int16_dithered(x, rng)
                for a, b in exact_ranges:
                    lo, hi = max(a, pos), min(b, pos + len(x))
                    if hi > lo:
                        q[lo - pos:hi - pos] = np.clip(np.round(x[lo - pos:hi - pos].astype(np.float64) * 32767.0),
                                                       -32768, 32767).astype(np.int16)
                out.write(q)
                h.update(q.tobytes())
                count += len(q)
                if progress:
                    progress((f.start_sample + count) / max(total, 1))
        tmp.replace(path)
        results.append({"file": f.audio_name, "samples": count, "pcm_sha256": h.hexdigest()})
    return results


def read_int16(dir_: Path, segs: list[dict], a: int, b: int, block: int = MASTER_SR * 20):
    """Yield int16 blocks of an earlier hour output between hour-timeline samples a and b."""
    for sg in segs:
        lo, hi = max(a, sg["start_sample"]), min(b, sg["end_sample"])
        if hi <= lo:
            continue
        with sf.SoundFile(str(dir_ / sg["audio"])) as f:
            f.seek(lo - sg["start_sample"])
            left = hi - lo
            while left > 0:
                blk = f.read(min(block, left), dtype="int16")
                if blk.size == 0:
                    raise IOError(f"{sg['audio']} is shorter than recorded in its manifest")
                left -= blk.size
                yield blk


def verify_segment(path: Path, expected_samples: int, expected_pcm_sha256: str | None = None) -> str | None:
    """Re-read a written segment; returns an error message or None."""
    try:
        info = sf.info(str(path))
        if info.samplerate != MASTER_SR or info.channels != 1:
            return f"{path.name}: unexpected format {info.samplerate} Hz / {info.channels} ch"
        if info.frames != expected_samples:
            return f"{path.name}: {info.frames} samples, expected {expected_samples}"
        if expected_pcm_sha256:
            h = hashlib.sha256()
            n = 0
            with sf.SoundFile(str(path)) as f:
                while True:
                    blk = f.read(MASTER_SR * 20, dtype="int16")
                    if blk.size == 0:
                        break
                    h.update(blk.tobytes())
                    n += blk.size
            if n != expected_samples or h.hexdigest() != expected_pcm_sha256:
                return f"{path.name}: decoded audio does not match what was written"
    except Exception as e:
        return f"{path.name}: cannot be read back ({e})"
    return None
