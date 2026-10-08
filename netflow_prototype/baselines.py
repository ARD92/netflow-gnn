"""Router-pair route_miles baseline.

route_miles is a property of the path between an ingress and an egress
router, not of the traffic: for a given pair it stays at one value (or a few,
with multiple paths) until routing changes. Learning it through hashed router
embeddings blurs pairs together, so a correct 460.6 could be "expected" as
136. Instead, the usual range of route_miles per (ingress, egress) pair is
measured directly from the files, and a flow is flagged on route_miles only
when it leaves that range by a meaningful margin. The GNN still scores bytes.

Per file, each pair is summarized (median, min, max of its flows'
route_miles). Across files, the baseline keeps the median of the medians, the
1st percentile of the minimums and the 99th percentile of the maximums (so a
rare excursion in the training window does not widen the range), and the
number of windows the pair was seen in.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PAIR_BASELINE_NAME = "pair_baseline.csv"
_SEP = "|"  # never part of a router name: the export itself is pipe-delimited


@dataclass
class PairMilesConfig:
    min_change: float = 0.10        # flag when outside the usual range by >= 10% of the median...
    min_change_miles: float = 40.0  # ...and by at least 40 miles
    min_windows: int = 3            # pairs seen in fewer windows fall back to the model


def pair_summary(flows: pd.DataFrame) -> pd.DataFrame:
    """Per (ingress, egress) pair in one window: median, min and max route_miles."""
    return (flows.groupby(["ingress", "egress"], sort=False)["route_miles"]
            .agg(["median", "min", "max"]).reset_index())


class PairBaselineBuilder:
    """Accumulates per-window pair summaries and builds the baseline table."""

    def __init__(self) -> None:
        self._index = pd.Index([], dtype=object)
        self._parts: list[np.ndarray] = []

    def add(self, summary: pd.DataFrame) -> None:
        keys = (summary["ingress"] + _SEP + summary["egress"]).to_numpy(dtype=object)
        codes = self._index.get_indexer(keys)
        if (codes < 0).any():
            self._index = self._index.append(pd.Index(pd.unique(keys[codes < 0])))
            codes = self._index.get_indexer(keys)
        self._parts.append(np.column_stack([
            codes, summary["median"], summary["min"], summary["max"],
        ]).astype(np.float64))

    def build(self) -> pd.DataFrame:
        if not self._parts:
            return pd.DataFrame(columns=["ingress", "egress", "median_miles", "low_miles",
                                         "high_miles", "windows"])
        data = pd.DataFrame(np.concatenate(self._parts),
                            columns=["code", "median", "min", "max"])
        grouped = data.groupby("code")
        out = pd.DataFrame({
            "median_miles": grouped["median"].median(),
            "low_miles": grouped["min"].quantile(0.01),
            "high_miles": grouped["max"].quantile(0.99),
            "windows": grouped.size(),
        })
        names = self._index[out.index.astype(int)].str.split(_SEP, n=1, expand=True)
        out.insert(0, "ingress", names.get_level_values(0))
        out.insert(1, "egress", names.get_level_values(1))
        return out.reset_index(drop=True)


class WindowBaselines:
    """Builds every file-derived baseline (router pairs, applications) in one pass."""

    def __init__(self, ports: dict[int, str]) -> None:
        from netflow_prototype.appbaseline import AppBaselineBuilder

        self.ports = ports
        self.pairs = PairBaselineBuilder()
        self.apps = AppBaselineBuilder()

    def __call__(self, ts, flows: pd.DataFrame) -> None:
        """Observer for ``load_graphs``: summarize one full window."""
        from netflow_prototype.appbaseline import app_summaries

        self.add(ts, pair_summary(flows), *app_summaries(flows, self.ports))

    def add(self, ts, pairs: pd.DataFrame, app_totals: pd.DataFrame,
            app_pairs: pd.DataFrame) -> None:
        self.pairs.add(pairs)
        self.apps.add(ts, app_totals, app_pairs)

    def save(self, model_dir: str | Path) -> dict[str, int]:
        from netflow_prototype import appbaseline

        pair_table = self.pairs.build()
        save(pair_table, model_dir)
        apps, app_pairs = self.apps.build()
        appbaseline.save(apps, app_pairs, model_dir)
        return {"router_pairs": len(pair_table),
                "applications": int((apps["hour"] == -1).sum()) if len(apps) else 0,
                "application_router_pairs": len(app_pairs)}


def _summarize_file(path: str, cache_dir: str | None, ports: dict[int, str]):
    from netflow_prototype.appbaseline import app_summaries
    from netflow_prototype.data import _safe_load  # local import: runs in worker processes
    from netflow_prototype.windows import file_timestamp

    flows = _safe_load(Path(path), cache_dir)
    if flows is None:
        return None
    return (file_timestamp(Path(path)), pair_summary(flows), *app_summaries(flows, ports))


def build_from_files(paths: list[Path], ports: dict[int, str], workers: int = 4,
                     cache_dir: str | Path | None = None) -> WindowBaselines:
    """Build pair and application baselines by reading files in parallel (no model)."""
    result = WindowBaselines(ports)
    n = len(paths)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        summaries = pool.map(_summarize_file, map(str, paths),
                             [None if cache_dir is None else str(cache_dir)] * n, [ports] * n)
        for i, summary in enumerate(summaries, start=1):
            if summary is not None:
                result.add(*summary)
            if i % 10 == 0 or i == n:
                logger.info("Baselines: %d/%d files read", i, n)
    return result


def save(table: pd.DataFrame, model_dir: str | Path) -> Path:
    path = Path(model_dir) / PAIR_BASELINE_NAME
    table.to_csv(path, index=False, float_format="%.3f")
    return path


def load(model_dir: str | Path) -> pd.DataFrame | None:
    """Pair baseline indexed by (ingress, egress), or None when absent."""
    path = Path(model_dir) / PAIR_BASELINE_NAME
    if not path.exists():
        return None
    table = pd.read_csv(path, dtype={"ingress": str, "egress": str}, keep_default_na=False)
    return table.set_index(["ingress", "egress"])


def apply_pair_baseline(df: pd.DataFrame, pairs: pd.DataFrame | None,
                        cfg: PairMilesConfig, edge_threshold: float) -> pd.DataFrame:
    """Replace the model's route_miles expectation with the router pair's usual range.

    For pairs in the baseline (seen in at least ``cfg.min_windows`` windows):
    ``expected_route_miles`` is the pair median, and ``z_miles`` is scaled so
    that leaving the usual range by exactly the tolerance (the larger of
    ``min_change`` x median and ``min_change_miles``) equals the edge
    threshold. Inside the range, z_miles is 0. ``score`` is recomputed.
    """
    df["miles_baseline"] = "model"
    df["usual_miles_low"] = np.nan
    df["usual_miles_high"] = np.nan
    if pairs is None or pairs.empty:
        return df
    idx = pairs.index.get_indexer(pd.MultiIndex.from_arrays([df["ingress"], df["egress"]]))
    found = idx >= 0
    rows = pairs.iloc[np.where(found, idx, 0)]
    med = rows["median_miles"].to_numpy(dtype=float)
    lo = rows["low_miles"].to_numpy(dtype=float)
    hi = rows["high_miles"].to_numpy(dtype=float)
    use = found & (rows["windows"].to_numpy() >= cfg.min_windows)

    for col in ("z_miles", "expected_route_miles", "z_bytes"):  # model outputs are float32
        df[col] = df[col].astype(np.float64)
    miles = df["route_miles"].to_numpy(dtype=float)
    deviation = np.where(miles > hi, miles - hi, np.where(miles < lo, miles - lo, 0.0))
    tolerance = np.maximum(cfg.min_change * med, cfg.min_change_miles)
    z = edge_threshold * deviation / tolerance

    df.loc[use, "z_miles"] = z[use]
    df.loc[use, "expected_route_miles"] = med[use]
    df.loc[use, "usual_miles_low"] = lo[use]
    df.loc[use, "usual_miles_high"] = hi[use]
    df.loc[use, "miles_baseline"] = "pair"
    df["score"] = np.maximum(df["z_bytes"].abs(), df["z_miles"].abs())
    return df
