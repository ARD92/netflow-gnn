"""Cache of aggregated flows, so each export is parsed only once.

Parsing and aggregating a gzipped export takes several seconds per window
(about 7 s for 2 million flows); loading the cached result takes about 0.2 s.
Each export ``netflow.YYYYMMDD.HH.MM.txt.gz`` is cached as
``<cache_dir>/netflow.YYYYMMDD.HH.MM.flows.v1.pkl.gz``: the aggregated flow
table with key columns stored as categories, gzip level 1 (about half the
size of the raw export).

A cache entry is used only while it is newer than its source file, so a
re-delivered export is re-read automatically. Bump ``CACHE_VERSION`` when
aggregation changes. The cache uses pickle: keep the cache directory writable
only by trusted users.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from netflow_prototype.schema import FLOW_KEY, load_window

CACHE_VERSION = 1
CACHE_SUFFIX = f".flows.v{CACHE_VERSION}.pkl.gz"


def cache_path(cache_dir: str | Path, source: str | Path) -> Path:
    name = Path(source).name
    for ext in (".gz", ".txt"):
        name = name.removesuffix(ext)
    return Path(cache_dir) / f"{name}{CACHE_SUFFIX}"


def is_fresh(cache_dir: str | Path, source: str | Path) -> bool:
    cached = cache_path(cache_dir, source)
    return cached.exists() and cached.stat().st_mtime >= Path(source).stat().st_mtime


def write(flows: pd.DataFrame, cache_dir: str | Path, source: str | Path) -> Path:
    """Write atomically (temp file + rename), so readers never see a partial entry."""
    target = cache_path(cache_dir, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    compact = flows.copy()
    for col in FLOW_KEY:
        compact[col] = compact[col].astype("category")
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    compact.to_pickle(tmp, compression={"method": "gzip", "compresslevel": 1})
    # Never older than its source, even if the source's timestamp is ahead (clock skew,
    # timestamps preserved on copy); otherwise the entry would look stale forever.
    stamp = max(tmp.stat().st_mtime, Path(source).stat().st_mtime)
    os.utime(tmp, (stamp, stamp))
    tmp.replace(target)
    return target


def read(cache_dir: str | Path, source: str | Path) -> pd.DataFrame:
    flows = pd.read_pickle(cache_path(cache_dir, source), compression="gzip")
    for col in FLOW_KEY:
        flows[col] = flows[col].astype(str)
    return flows


def load_window_cached(path: str | Path, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Aggregated flows for one export, from the cache when fresh, else parsed (and cached)."""
    if cache_dir is None:
        return load_window(path)
    if is_fresh(cache_dir, path):
        return read(cache_dir, path)
    flows = load_window(path)
    if not flows.empty:
        write(flows, cache_dir, path)
    return flows


def _prepare_one(path: str, cache_dir: str) -> str:
    """Cache one export; returns 'cached', 'fresh' or 'skipped' (small result for the parent)."""
    if is_fresh(cache_dir, path):
        return "fresh"
    try:
        flows = load_window(path)
    except (EOFError, OSError, ValueError):
        return "skipped"
    if flows.empty:
        return "skipped"
    write(flows, cache_dir, path)
    return "cached"


def prepare(paths: list[Path], cache_dir: str | Path, workers: int = 4,
            progress=None) -> dict[str, list[str]]:
    """Parse and cache every export not already cached (in parallel)."""
    from concurrent.futures import ProcessPoolExecutor

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    result: dict[str, list[str]] = {"cached": [], "fresh": [], "skipped": []}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        statuses = pool.map(_prepare_one, map(str, paths), [str(cache_dir)] * len(paths))
        for i, (path, status) in enumerate(zip(paths, statuses, strict=True), start=1):
            result[status].append(Path(path).name)
            if progress:
                progress(i, len(paths))
    return result
