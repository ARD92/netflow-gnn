"""Load a resolved set of NetFlow files into a sequence of window graphs."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from netflow_prototype.graph import HashConfig, WindowGraph, build_window_graph
from netflow_prototype.schema import load_window
from netflow_prototype.windows import DEFAULT_INTERVAL, file_name

logger = logging.getLogger(__name__)


def context_file(data_dir: Path, first: datetime,
                 interval: timedelta = DEFAULT_INTERVAL) -> tuple[datetime, Path] | None:
    """Return the file immediately before ``first`` (used only for lag features)."""
    ts = first - interval
    for gz in (True, False):
        path = Path(data_dir) / file_name(ts, gz=gz)
        if path.exists():
            return ts, path
    return None


def load_graphs(
    files: list[tuple[datetime, Path]],
    hashes: HashConfig,
    context: tuple[datetime, Path] | None = None,
    keep_flows: bool = False,
    context_flows: np.ndarray | None = None,
    max_flows_per_window: int | None = None,
    interval: timedelta = DEFAULT_INTERVAL,
    seed: int = 0,
) -> list[WindowGraph]:
    """Build one WindowGraph per file, wiring previous-window lag features.

    Lag features are only used when the previous file is exactly one interval
    earlier; a gap in the data resets them.
    """
    rng = np.random.default_rng(seed)
    graphs: list[WindowGraph] = []
    skipped: list[str] = []
    prev_ts, prev_flows = None, None
    if context is not None:
        logger.info("Reading previous-window context %s", context[1].name)
        ctx_flows = _safe_load(context[1])
        if ctx_flows is not None:
            prev_ts, prev_flows = context[0], ctx_flows

    for i, (ts, path) in enumerate(files):
        t0 = time.perf_counter()
        flows = _safe_load(path)
        t_read = time.perf_counter() - t0
        if flows is None:
            skipped.append(path.name)
            continue
        lag = prev_flows if prev_ts is not None and ts - prev_ts == interval else None
        sample = flows
        if max_flows_per_window and len(flows) > max_flows_per_window:
            idx = rng.choice(len(flows), size=max_flows_per_window, replace=False)
            sample = flows.iloc[np.sort(idx)]
        t0 = time.perf_counter()
        g = build_window_graph(sample, ts, lag, hashes, keep_flows=keep_flows,
                               context_flows=context_flows)
        t_graph = time.perf_counter() - t0
        graphs.append(g)
        prev_ts, prev_flows = ts, flows
        logger.info("[%d/%d] %s: %d routers, %d flows (read+aggregate %.1fs, graph %.1fs)",
                    i + 1, len(files), path.name, g.num_nodes, g.num_edges, t_read, t_graph)

    if skipped:
        logger.warning("Skipped %d unreadable files: %s", len(skipped), ", ".join(skipped))
    if not graphs:
        raise ValueError("None of the selected files could be read.")
    return graphs


def _safe_load(path: Path) -> pd.DataFrame | None:
    """Load a window, returning None for truncated, corrupt, or malformed files."""
    try:
        flows = load_window(path)
    except (EOFError, OSError, ValueError) as exc:
        # EOFError: truncated gzip (file still being written or partially copied).
        logger.warning("Skipping %s: %s: %s", path.name, type(exc).__name__, exc)
        return None
    if flows.empty:
        logger.warning("Skipping %s: no valid flow records", path.name)
        return None
    return flows
