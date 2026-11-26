from dataclasses import dataclass
from datetime import datetime

from audiorefine.ordering import determine_order, natural_key, parse_filename_datetime, parse_metadata_datetime


@dataclass
class Item:
    name: str
    name_datetime: datetime | None = None
    meta_datetime: datetime | None = None


def names(items):
    return [i.name for i in items]


def test_natural_sort_numbers():
    files = ["Recording10.m4a", "Recording2.m4a", "Recording9.m4a", "Recording1.m4a"]
    assert sorted(files, key=natural_key) == ["Recording1.m4a", "Recording2.m4a", "Recording9.m4a", "Recording10.m4a"]


def test_natural_sort_case_spaces_leading_zeros():
    files = ["rec 10.wav", "Rec 2.wav", "REC 1.wav", "rec 002.wav", "Voice 001.m4a", "voice 01.m4a"]
    out = sorted(files, key=natural_key)
    assert out.index("REC 1.wav") < out.index("Rec 2.wav") < out.index("rec 10.wav")
    assert out.index("rec 002.wav") < out.index("rec 10.wav")
    assert sorted(files, key=natural_key) == sorted(reversed(files), key=natural_key)  # deterministic


def test_natural_sort_multiple_numbers():
    files = ["Day 2 part 10.mp3", "Day 2 part 9.mp3", "Day 10 part 1.mp3", "Day 2 part 1.mp3"]
    assert sorted(files, key=natural_key) == ["Day 2 part 1.mp3", "Day 2 part 9.mp3", "Day 2 part 10.mp3", "Day 10 part 1.mp3"]


def test_filename_datetime_patterns():
    cases = {
        "20240315_143022": datetime(2024, 3, 15, 14, 30, 22),
        "REC_20240315-143022": datetime(2024, 3, 15, 14, 30, 22),
        "20240315 143022": datetime(2024, 3, 15, 14, 30, 22),
        "Recording_20240315T1430": datetime(2024, 3, 15, 14, 30, 0),
        "2024-03-15 14.30.22": datetime(2024, 3, 15, 14, 30, 22),
        "2024-03-15_14-30-22": datetime(2024, 3, 15, 14, 30, 22),
        "Optagelse 2024-03-15 at 14.30.22": datetime(2024, 3, 15, 14, 30, 22),
        "15-03-2024 14.30": datetime(2024, 3, 15, 14, 30, 0),
    }
    for s, dt in cases.items():
        assert parse_filename_datetime(s) == dt, s


def test_filename_datetime_rejects_non_dates():
    for s in ["Recording001", "Recording 10", "ZOOM0001", "20241345_250000", "4512345678", "Voice 2024", "track 12345678"]:
        assert parse_filename_datetime(s) is None, s


def test_metadata_datetime():
    assert parse_metadata_datetime("2024-03-15T14:30:22.000000Z") == datetime(2024, 3, 15, 14, 30, 22)
    assert parse_metadata_datetime("2024-03-15T15:30:22+01:00") == datetime(2024, 3, 15, 14, 30, 22)
    assert parse_metadata_datetime("2024-03-15 14:30:22") == datetime(2024, 3, 15, 14, 30, 22)
    assert parse_metadata_datetime("1970-01-01T00:00:00Z") is None
    assert parse_metadata_datetime("garbage") is None


def test_order_by_filename_datetime_when_all_have_one():
    items = [Item("b_20240316_090000.m4a", datetime(2024, 3, 16, 9)), Item("a_20240315_100000.m4a", datetime(2024, 3, 15, 10)),
             Item("z_20240315_080000.m4a", datetime(2024, 3, 15, 8))]
    ordered, basis, warnings = determine_order(items)
    assert names(ordered) == ["z_20240315_080000.m4a", "a_20240315_100000.m4a", "b_20240316_090000.m4a"]
    assert "date and time" in basis


def test_order_natural_when_not_all_dated():
    items = [Item("Recording10.m4a"), Item("Recording2.m4a", datetime(2024, 1, 1, 1, 1)), Item("Recording1.m4a")]
    ordered, basis, _ = determine_order(items)
    assert names(ordered) == ["Recording1.m4a", "Recording2.m4a", "Recording10.m4a"]
    assert "natural" in basis


def test_metadata_contradiction_warns_but_does_not_reorder():
    items = [Item("Rec1.m4a", meta_datetime=datetime(2024, 1, 2)), Item("Rec2.m4a", meta_datetime=datetime(2024, 1, 1))]
    ordered, _, warnings = determine_order(items)
    assert names(ordered) == ["Rec1.m4a", "Rec2.m4a"]
    assert warnings and "different order" in warnings[0]


def test_filesystem_times_never_used():
    # determine_order has no access to file-system times at all: same input gives same output
    items = [Item("b.wav"), Item("a.wav")]
    assert names(determine_order(items)[0]) == ["a.wav", "b.wav"]
