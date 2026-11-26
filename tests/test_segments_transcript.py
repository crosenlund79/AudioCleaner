import numpy as np
import pytest
import soundfile as sf

from audiorefine import segments
from audiorefine.constants import MASTER_SR, SEGMENT_SAMPLES
from audiorefine.rawaudio import RawWriter
from audiorefine.transcript import (Piece, SegmentFile, SourceSpan, TSegment, TWord, fmt_srt, fmt_ts, render_complete,
                                    render_segment_srt, render_segment_txt, split_into_files, words_for_pieces,
                                    write_transcripts)

TEN = SEGMENT_SAMPLES


@pytest.mark.parametrize("total,expected", [
    (1, [1]),
    (TEN - 1, [TEN - 1]),
    (TEN, [TEN]),
    (TEN + 1, [TEN, 1]),
    (2 * TEN, [TEN, TEN]),
    (3 * TEN + 17 * 60 * MASTER_SR // 4, [TEN, TEN, TEN, 17 * 60 * MASTER_SR // 4]),
])
def test_plan_segments_exact(total, expected):
    files = segments.plan_segments(total)
    assert [f.end_sample - f.start_sample for f in files] == expected
    assert files[0].start_sample == 0 and files[-1].end_sample == total
    for a, b in zip(files, files[1:]):
        assert a.end_sample == b.start_sample  # no gap, no overlap
    assert [f.audio_name for f in files] == sorted(f.audio_name for f in files)


def test_plan_segments_multi_hour():
    total = int(5.5 * 3600 * MASTER_SR) + 123
    files = segments.plan_segments(total)
    assert len(files) == 34
    assert all(f.end_sample - f.start_sample == TEN for f in files[:-1])
    assert sum(f.end_sample - f.start_sample for f in files) == total
    assert files[0].audio_name == "Audio_001.flac" and files[-1].audio_name == "Audio_034.flac"


def test_write_segments_no_loss_no_duplication(tmp_path):
    seg = 48000  # 1-second segments keep the test fast; same code path as 10 minutes
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(seg * 3 + 4321) * 0.2).astype(np.float32)
    x[seg - 1], x[seg], x[seg + 1] = 0.9, -0.9, 0.5  # samples right before/at/after a boundary
    master = tmp_path / "m.f32"
    with RawWriter(master) as w:
        w.write(x)
    files = segments.plan_segments(len(x), seg)
    res = segments.write_segments(master, tmp_path, files)
    got = []
    for f, r in zip(files, res):
        y, sr = sf.read(tmp_path / f.audio_name, dtype="float32")
        assert sr == MASTER_SR
        assert len(y) == f.end_sample - f.start_sample == r["samples"]
        assert segments.verify_segment(tmp_path / f.audio_name, r["samples"], r["pcm_sha256"]) is None
        got.append(y)
    y = np.concatenate(got)
    assert len(y) == len(x)
    assert np.max(np.abs(y - x)) < 3 / 32768  # 16-bit quantisation + dither only
    assert abs(y[seg - 1] - 0.9) < 1e-3 and abs(y[seg] + 0.9) < 1e-3 and abs(y[seg + 1] - 0.5) < 1e-3


def test_verify_segment_detects_problems(tmp_path):
    p = tmp_path / "a.flac"
    sf.write(p, np.zeros(1000, np.int16), MASTER_SR, subtype="PCM_16")
    assert segments.verify_segment(p, 1001) is not None
    assert segments.verify_segment(tmp_path / "missing.flac", 10) is not None


# ------------------------------------------------------------------ transcript formatting
def test_timestamp_formats():
    assert fmt_ts(0) == "00:00:00.000"
    assert fmt_ts(604.1206) == "00:10:04.121"
    assert fmt_ts(3600 * 5 + 59.9996) == "05:01:00.000"
    assert fmt_srt(4.12) == "00:00:04,120"


def make_files(n, seg=600.0, last=None):
    files = []
    for i in range(n):
        end = (i + 1) * seg if (i < n - 1 or last is None) else i * seg + last
        files.append(SegmentFile(i + 1, f"Audio_{i + 1:03d}", f"Audio_{i + 1:03d}.flac", i * seg, end,
                                 int(i * seg * MASTER_SR), int(end * MASTER_SR)))
    return files


def seg_from_words(words, text=None):
    ws = [TWord(s, e, t) for s, e, t in words]
    return TSegment(ws[0].start, ws[-1].end, text or "".join(w.text for w in ws).strip(), ws)


def test_segment_crossing_boundary_split_by_words():
    files = make_files(2)
    s = seg_from_words([(598.0, 598.5, " Men"), (598.6, 599.0, " vi"), (599.2, 599.9, " bliver"),
                        (600.1, 600.6, " nødt"), (600.7, 601.0, " til."), ])
    pieces = split_into_files([s], files)
    assert [p.text for p in pieces[0]] == ["Men vi bliver"]
    assert [p.text for p in pieces[1]] == ["nødt til."]
    assert pieces[0][0].cont_next and pieces[1][0].cont_prev
    words0 = sum(len(p.text.split()) for p in pieces[0]) + sum(len(p.text.split()) for p in pieces[1])
    assert words0 == 5  # nothing lost, nothing duplicated
    srt2 = render_segment_srt(files[1], pieces[1], words_for_pieces([s], files)[1])
    assert "00:00:00,100 --> 00:00:01,000" in srt2  # relative to Audio_002
    txt2 = render_segment_txt(files[1], pieces[1], [SourceSpan("a.m4a", 0, 1200)])
    assert "[00:10:00.100 – 00:10:01.000]" in txt2  # continuous timeline
    assert "… nødt til." in txt2
    assert txt2.startswith("Recording timeline: 00:10:00 – 00:20:00")


def test_boundary_word_at_exact_boundary_goes_to_later_file():
    files = make_files(2)
    s = seg_from_words([(599.5, 599.99, " ja"), (600.0, 600.4, " nej")])
    pieces = split_into_files([s], files)
    assert pieces[0][0].text == "ja" and pieces[1][0].text == "nej"


def test_srt_relative_and_txt_absolute_example():
    files = make_files(3, last=257.0)
    s = seg_from_words([(604.12, 605.0, " Det"), (605.1, 608.42, " var egentlig det.")])
    pieces = split_into_files([s], files)
    assert pieces[0] == [] and pieces[2] == []
    srt = render_segment_srt(files[1], pieces[1], words_for_pieces([s], files)[1])
    assert srt.splitlines()[1] == "00:00:04,120 --> 00:00:08,420"
    txt = render_segment_txt(files[1], pieces[1], [])
    assert "[00:10:04.120 – 00:10:08.420]" in txt


def test_srt_cues_split_long_segments_and_dont_overlap():
    files = make_files(1)
    words = [(i * 0.5, i * 0.5 + 0.4, f" ord{i}") for i in range(60)]
    s = seg_from_words(words)
    pieces = split_into_files([s], files)
    srt = render_segment_srt(files[0], pieces[0], words_for_pieces([s], files)[0])
    blocks = [b for b in srt.strip().split("\n\n")]
    assert len(blocks) >= 4
    times = []
    for b in blocks:
        line = b.splitlines()[1]
        a, z = line.split(" --> ")
        to_s = lambda t: int(t[:2]) * 3600 + int(t[3:5]) * 60 + float(t[6:].replace(",", "."))
        times.append((to_s(a), to_s(z)))
        assert len(b.splitlines()) <= 4
    for (a1, z1), (a2, z2) in zip(times, times[1:]):
        assert a1 < z1 <= a2 < z2
    allwords = " ".join(" ".join(b.splitlines()[2:]) for b in blocks).split()
    assert allwords == [f"ord{i}" for i in range(60)]


def test_unclear_marker_and_uncertain_flag_rendered():
    files = make_files(2)
    m = TSegment(598.0, 603.0, "[unclear speech]", [], uncertain=True, marker=True)
    u = seg_from_words([(10.0, 11.0, " måske")])
    u.uncertain = True
    pieces = split_into_files([u, m], files)
    t0 = render_segment_txt(files[0], pieces[0], [])
    assert "[00:00:10.000 – 00:00:11.000] [uncertain]" in t0
    assert "[unclear speech]" in t0
    t1 = render_segment_txt(files[1], pieces[1], [])
    assert "[00:10:00.000 – 00:10:03.000]" in t1


def test_complete_transcript_continuous_and_files_written(tmp_path):
    files = make_files(2, last=100.0)
    segs = [seg_from_words([(1.0, 2.0, " Hej.")]), seg_from_words([(650.0, 651.5, " Farvel.")])]
    spans = [SourceSpan("Recording001.m4a", 0, 300), SourceSpan("Recording002.m4a", 300, 700)]
    out = write_transcripts(tmp_path, segs, spans, files, {"batch": "b", "duration": 700, "version": "1", "model": "m",
                                                           "created": "now", "languages": ["Danish (da)"]})
    names = sorted(p.name for p in out)
    assert names == ["Audio_001.srt", "Audio_001.txt", "Audio_002.srt", "Audio_002.txt", "Transcript_Complete.txt"]
    full = (tmp_path / "Transcript_Complete.txt").read_text(encoding="utf-8")
    assert full.index("[00:00:01.000") < full.index("Recording002.m4a begins at 00:05:00.000") < full.index("[00:10:50.000")
    assert any(l.strip().startswith("1. Recording001.m4a") and l.endswith("00:00:00.000 – 00:05:00.000")
               for l in full.splitlines())
    raw = (tmp_path / "Audio_002.txt").read_bytes()
    raw.decode("utf-8")  # valid UTF-8
    assert "No speech" not in (tmp_path / "Audio_001.txt").read_text(encoding="utf-8")


def test_empty_segment_file_says_no_speech(tmp_path):
    files = make_files(1, last=30.0)
    write_transcripts(tmp_path, [], [SourceSpan("a.wav", 0, 30)], files, {})
    assert "No speech was detected" in (tmp_path / "Audio_001.txt").read_text(encoding="utf-8")
    assert (tmp_path / "Audio_001.srt").read_text(encoding="utf-8") == ""
