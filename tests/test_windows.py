from datetime import datetime, timedelta

import pytest

from netflow_prototype.windows import (
    file_timestamp,
    parse_duration,
    parse_time,
    resolve_files,
    resolve_range,
)


def _touch(directory, *stamps):
    for s in stamps:
        (directory / f"netflow.{s}.txt.gz").write_bytes(b"")


def test_parse_time_formats():
    expected = datetime(2026, 10, 6, 11, 50)
    assert parse_time("2026-10-06 11:50") == expected
    assert parse_time("2026-10-06T11:50") == expected
    assert parse_time("20261006.11.50") == expected
    with pytest.raises(ValueError):
        parse_time("yesterday")


def test_parse_duration():
    assert parse_duration("90m") == timedelta(minutes=90)
    assert parse_duration("6h") == timedelta(hours=6)
    assert parse_duration("2d") == timedelta(days=2)


def test_resolve_range_requires_end_or_duration():
    with pytest.raises(ValueError):
        resolve_range("2026-10-06 11:00")
    t0, t1 = resolve_range("2026-10-06 11:00", duration="1h")
    assert t1 - t0 == timedelta(hours=1)


def test_file_timestamp():
    from pathlib import Path

    assert file_timestamp(Path("netflow.20261006.11.50.txt.gz")) == datetime(2026, 10, 6, 11, 50)
    assert file_timestamp(Path("netflow.20261006.11.50.txt")) == datetime(2026, 10, 6, 11, 50)
    assert file_timestamp(Path("other.txt.gz")) is None


def test_resolve_files_half_open_window_and_missing(tmp_path):
    _touch(tmp_path, "20261006.10.50", "20261006.11.00", "20261006.11.10",
           "20261006.11.30", "20261006.11.40", "20261006.11.50", "20261006.12.00")
    fs = resolve_files(tmp_path, datetime(2026, 10, 6, 11), datetime(2026, 10, 6, 12))
    assert [t.minute for t in fs.timestamps] == [0, 10, 30, 40, 50]
    assert fs.missing == [datetime(2026, 10, 6, 11, 20)]
