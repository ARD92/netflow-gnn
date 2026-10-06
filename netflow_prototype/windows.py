"""Resolve a user-supplied time window to the NetFlow files that cover it.

Files are named ``netflow.YYYYMMDD.HH.MM.txt[.gz]`` and are produced every
10 minutes. A window ``[start, end)`` selects every file whose timestamp is
``>= start`` and ``< end``; for example 11:00-12:00 selects 11:00 ... 11:50.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

FILE_PATTERN = re.compile(r"^netflow\.(\d{8})\.(\d{2})\.(\d{2})\.txt(\.gz)?$")
DEFAULT_INTERVAL = timedelta(minutes=10)

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([mhd])\s*$", re.IGNORECASE)
_TIME_FORMATS = ("%Y%m%d.%H.%M", "%Y%m%d%H%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M")


@dataclass
class FileSet:
    """Files resolved for a time window, ordered by timestamp."""

    start: datetime
    end: datetime
    files: list[tuple[datetime, Path]]
    missing: list[datetime] = field(default_factory=list)

    @property
    def paths(self) -> list[Path]:
        return [p for _, p in self.files]

    @property
    def timestamps(self) -> list[datetime]:
        return [t for t, _ in self.files]


def parse_time(value: str) -> datetime:
    """Parse a user-supplied timestamp (ISO or file-style ``YYYYMMDD.HH.MM``)."""
    value = value.strip()
    # Explicit formats first: fromisoformat reads '20261006.11.50' as a fractional hour.
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    raise ValueError(
        f"Unrecognized time '{value}'. Use 'YYYY-MM-DD HH:MM' or 'YYYYMMDD.HH.MM'."
    )


def parse_duration(value: str) -> timedelta:
    """Parse durations such as '30m', '6h', '2d'."""
    match = _DURATION.match(value)
    if not match:
        raise ValueError(f"Unrecognized duration '{value}'. Use e.g. 30m, 6h, 2d.")
    amount, unit = float(match.group(1)), match.group(2).lower()
    return {"m": timedelta(minutes=amount), "h": timedelta(hours=amount),
            "d": timedelta(days=amount)}[unit]


def resolve_range(
    start: str, end: str | None = None, duration: str | None = None
) -> tuple[datetime, datetime]:
    """Turn CLI inputs (start + end, or start + duration) into a datetime range."""
    t0 = parse_time(start)
    if end and duration:
        raise ValueError("Pass either --end or --duration, not both.")
    if end:
        t1 = parse_time(end)
    elif duration:
        t1 = t0 + parse_duration(duration)
    else:
        raise ValueError("Pass --end or --duration with --start.")
    if t1 <= t0:
        raise ValueError(f"End time {t1} must be after start time {t0}.")
    return t0, t1


def file_timestamp(path: Path) -> datetime | None:
    """Extract the window timestamp from a NetFlow file name."""
    match = FILE_PATTERN.match(path.name)
    if not match:
        return None
    day, hour, minute = match.group(1), match.group(2), match.group(3)
    return datetime.strptime(f"{day}{hour}{minute}", "%Y%m%d%H%M")


def file_name(ts: datetime, gz: bool = True) -> str:
    """Build the canonical file name for a window timestamp."""
    return f"netflow.{ts:%Y%m%d.%H.%M}.txt" + (".gz" if gz else "")


def index_directory(data_dir: str | Path) -> dict[datetime, Path]:
    """Map timestamp -> path for every NetFlow file in a directory."""
    index: dict[datetime, Path] = {}
    for path in Path(data_dir).iterdir():
        ts = file_timestamp(path)
        if ts is not None:
            # Prefer the compressed file when both exist.
            if ts not in index or path.suffix == ".gz":
                index[ts] = path
    return index


def resolve_files(
    data_dir: str | Path,
    start: datetime,
    end: datetime,
    interval: timedelta = DEFAULT_INTERVAL,
) -> FileSet:
    """Select the files covering ``[start, end)`` and report missing slots."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    index = index_directory(data_dir)
    files = sorted((ts, p) for ts, p in index.items() if start <= ts < end)

    missing: list[datetime] = []
    slot = _align(start, interval)
    if slot < start:
        slot += interval
    while slot < end:
        if slot not in index:
            missing.append(slot)
        slot += interval
    return FileSet(start=start, end=end, files=files, missing=missing)


def files_from_paths(paths: list[str | Path]) -> FileSet:
    """Build a FileSet from explicit file paths (for ad-hoc inference)."""
    files = []
    for p in map(Path, paths):
        ts = file_timestamp(p)
        if ts is None:
            raise ValueError(f"{p.name} does not match netflow.YYYYMMDD.HH.MM.txt[.gz]")
        files.append((ts, p))
    files.sort()
    return FileSet(start=files[0][0], end=files[-1][0] + DEFAULT_INTERVAL, files=files)


def _align(ts: datetime, interval: timedelta) -> datetime:
    """Round a timestamp down to the interval grid."""
    midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    steps = (ts - midnight) // interval
    return midnight + steps * interval
