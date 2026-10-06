"""Load a resolved set of NetFlow files into a sequence of window graphs."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

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
    prev_ts, prev_flows = None, None
    if context is not None:
        prev_ts, prev_flows = context[0], load_window(context[1])

    for i, (ts, path) in enumerate(files):
        flows = load_window(path)
        lag = prev_flows if prev_ts is not None and ts - prev_ts == interval else None
        sample = flows
        if max_flows_per_window and len(flows) > max_flows_per_window:
            idx = rng.choice(len(flows), size=max_flows_per_window, replace=False)
            sample = flows.iloc[np.sort(idx)]
        g = build_window_graph(sample, ts, lag, hashes, keep_flows=keep_flows)
        graphs.append(g)
        prev_ts, prev_flows = ts, flows
        logger.info("[%d/%d] %s: %d routers, %d flows",
                    i + 1, len(files), path.name, g.num_nodes, g.num_edges)
    return graphs
