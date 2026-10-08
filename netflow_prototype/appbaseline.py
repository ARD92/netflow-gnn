"""Application byte baselines and application-level anomaly rules.

Two baselines are measured from the training (or baseline) window:

- Per application, network-wide: bytes per window (median and 99th
  percentile, overall and per hour of day), median flows, and the share of
  windows in which the application appears.
- Per application on each router pair: statistics of log(bytes) over the
  windows where it appears, plus how often it appears.

Rules (asymmetric on purpose). Untyped buckets (``any`` for dstPort ``*`` and
``other-*``) mix unrelated traffic and are never judged.

- A tracked critical application (DNS, NTP, RADIUS, Diameter, BGP, GTP, PFCP,
  NGAP, ...; ``tracked``) that is normally always present (default: in 90%
  of windows) and drops below 20% of its usual bytes is ``application_drop``;
  if it vanishes entirely it is ``application_disappeared``.
- A surge above 5x usual (and above its 99th percentile) is
  ``application_surge``, except for burst-tolerant applications (https, http,
  ...), whose bursts are expected.
- On a single router pair, any application, https included, above 20x its
  usual bytes (and 4 standard deviations in log space) is ``app_pair_surge``.
- A tracked application that is on a router pair in almost every window (95%,
  seen in 12+ windows) with several flows each time (5+ on average) and drops below
  10% or vanishes is ``app_pair_drop`` / ``app_pair_disappeared``. Pairs
  carried by one or two flows are left to the flow-level model: one flow
  pausing would otherwise look like the application vanishing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from netflow_prototype.enrich import TRACKED_APPLICATIONS, UNTYPED_APPLICATIONS, applications

APP_BASELINE_NAME = "app_baseline.csv"
APP_PAIR_BASELINE_NAME = "app_pair_baseline.csv"
_SEP = "|"

EVENT_COLUMNS = ["window", "level", "kind", "application", "ingress", "egress", "bytes",
                 "usual_bytes", "ratio", "flows", "reason"]


@dataclass
class AppRuleConfig:
    tracked: tuple[str, ...] = TRACKED_APPLICATIONS       # drop/disappear rules apply
    burst_tolerant: tuple[str, ...] = ("https", "http", "http-alt", "https-alt", "gtp-u")
    untyped: tuple[str, ...] = UNTYPED_APPLICATIONS       # never judged
    regular_presence: float = 0.9
    drop_fraction: float = 0.2
    surge_factor: float = 5.0
    pair_surge_factor: float = 20.0
    pair_surge_z: float = 4.0
    pair_min_std: float = 0.5        # floor on log-space std for pair surges
    pair_regular_presence: float = 0.95
    pair_min_windows: int = 12
    pair_min_flows: float = 5.0      # average flows per window for pair drop rules
    pair_drop_fraction: float = 0.1
    min_hour_windows: int = 3        # hourly reference needs this many windows


def app_summaries(flows: pd.DataFrame, ports: dict[int, str]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One window: bytes and flows per application, and bytes per application x router pair."""
    f = flows[["ingress", "egress", "dstPort", "bytes"]].assign(
        application=applications(flows["dstPort"], ports))
    totals = f.groupby("application").agg(bytes=("bytes", "sum"),
                                          flows=("bytes", "size")).reset_index()
    pairs = f.groupby(["application", "ingress", "egress"]).agg(
        bytes=("bytes", "sum"), flows=("bytes", "size")).reset_index()
    return totals, pairs


@dataclass
class AppBaselineBuilder:
    """Accumulates per-window application summaries (streaming for router pairs)."""

    windows: list[datetime] = field(default_factory=list)
    _totals: list[pd.DataFrame] = field(default_factory=list)
    _index: pd.Index = field(default_factory=lambda: pd.Index([], dtype=object))
    _n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _m2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _max: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _flows: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def add(self, ts: datetime, totals: pd.DataFrame, pairs: pd.DataFrame) -> None:
        self.windows.append(ts)
        self._totals.append(totals.assign(window=ts))
        keys = (pairs["application"] + _SEP + pairs["ingress"] + _SEP
                + pairs["egress"]).to_numpy(dtype=object)
        codes = self._index.get_indexer(keys)
        if (codes < 0).any():
            self._index = self._index.append(pd.Index(pd.unique(keys[codes < 0])))
            codes = self._index.get_indexer(keys)
            grow = len(self._index) - len(self._n)
            self._n, self._mean, self._m2, self._max, self._flows = (
                np.concatenate([a, np.zeros(grow)])
                for a in (self._n, self._mean, self._m2, self._max, self._flows))
        # Welford update of log1p(bytes), one observation per key per window.
        x = np.log1p(pairs["bytes"].to_numpy(dtype=float))
        n = self._n[codes] + 1
        delta = x - self._mean[codes]
        mean = self._mean[codes] + delta / n
        self._m2[codes] += delta * (x - mean)
        self._mean[codes] = mean
        self._n[codes] = n
        self._max[codes] = np.maximum(self._max[codes], pairs["bytes"].to_numpy(dtype=float))
        self._flows[codes] += pairs["flows"].to_numpy(dtype=float)

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        total_windows = len(self.windows)
        if not total_windows:
            return pd.DataFrame(), pd.DataFrame()
        t = pd.concat(self._totals, ignore_index=True)
        grid = pd.MultiIndex.from_product([sorted(t["application"].unique()), self.windows],
                                          names=["application", "window"])
        t = (t.set_index(["application", "window"]).reindex(grid, fill_value=0)
             .reset_index())  # absent windows count as zero bytes
        t["hour"] = pd.to_datetime(t["window"]).dt.hour

        def stats(group: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "windows": len(group),
                "median_bytes": group["bytes"].median(),
                "p99_bytes": group["bytes"].quantile(0.99),
                "median_flows": group["flows"].median(),
                "presence": (group["flows"] > 0).mean(),
            })

        overall = t.groupby("application").apply(stats, include_groups=False).reset_index()
        overall.insert(1, "hour", -1)
        hourly = t.groupby(["application", "hour"]).apply(stats, include_groups=False)
        apps = pd.concat([overall, hourly.reset_index()], ignore_index=True)

        n = self._n
        std = np.sqrt(np.where(n > 1, self._m2 / np.maximum(n - 1, 1), 0.0))
        names = self._index.str.split(_SEP, n=2, expand=True)
        pairs = pd.DataFrame({
            "application": names.get_level_values(0),
            "ingress": names.get_level_values(1),
            "egress": names.get_level_values(2),
            "windows": n.astype(int),
            "mean_log_bytes": self._mean,
            "std_log_bytes": std,
            "max_bytes": self._max,
            "mean_flows": self._flows / np.maximum(n, 1),
            "presence": n / total_windows,
        })
        return apps, pairs


def save(apps: pd.DataFrame, pairs: pd.DataFrame, model_dir: str | Path) -> None:
    apps.to_csv(Path(model_dir) / APP_BASELINE_NAME, index=False, float_format="%.4f")
    pairs.to_csv(Path(model_dir) / APP_PAIR_BASELINE_NAME, index=False, float_format="%.6f")


@dataclass
class AppBaseline:
    apps: pd.DataFrame   # indexed by (application, hour); hour -1 = all hours
    pairs: pd.DataFrame  # indexed by (application, ingress, egress)

    @classmethod
    def load(cls, model_dir: str | Path) -> AppBaseline | None:
        a, p = Path(model_dir) / APP_BASELINE_NAME, Path(model_dir) / APP_PAIR_BASELINE_NAME
        if not (a.exists() and p.exists()):
            return None
        apps = pd.read_csv(a, dtype={"application": str}, keep_default_na=False)
        pairs = pd.read_csv(p, dtype={"application": str, "ingress": str, "egress": str},
                            keep_default_na=False)
        return cls(apps.set_index(["application", "hour"]),
                   pairs.set_index(["application", "ingress", "egress"]))


def _fmt(value: float) -> str:
    return f"{value:,.0f}"


def detect_app_events(edges: pd.DataFrame, ts: datetime, base: AppBaseline,
                      cfg: AppRuleConfig) -> pd.DataFrame:
    """Application and application-pair events for one window (``edges`` of that window)."""
    events: list[dict] = []
    totals = edges.groupby("application").agg(bytes=("bytes", "sum"), flows=("bytes", "size"))

    overall = base.apps.xs(-1, level="hour")
    hour = pd.Timestamp(ts).hour
    for app, ref in overall.iterrows():
        hourly = base.apps.loc[(app, hour)] if (app, hour) in base.apps.index else None
        if hourly is not None and hourly["windows"] >= cfg.min_hour_windows:
            ref = hourly
        usual = float(ref["median_bytes"])
        cur_bytes = float(totals["bytes"].get(app, 0.0))
        cur_flows = int(totals["flows"].get(app, 0))
        common = {"window": ts, "level": "application", "application": app,
                  "ingress": "", "egress": "", "bytes": cur_bytes, "usual_bytes": usual,
                  "ratio": cur_bytes / usual if usual > 0 else np.nan, "flows": cur_flows}
        if app in cfg.untyped:
            continue
        regular = (app in cfg.tracked and overall.loc[app, "presence"] >= cfg.regular_presence
                   and usual > 0)
        if regular and cur_flows == 0:
            events.append({**common, "kind": "application_disappeared",
                           "reason": f"{app} traffic disappeared (usually {_fmt(usual)} bytes)"})
        elif regular and cur_bytes < cfg.drop_fraction * usual:
            events.append({**common, "kind": "application_drop",
                           "reason": f"{app} bytes {100 * (1 - cur_bytes / usual):.0f}% below "
                                     f"usual ({_fmt(cur_bytes)} vs {_fmt(usual)})"})
        elif (app not in cfg.burst_tolerant and usual > 0
              and cur_bytes > cfg.surge_factor * usual and cur_bytes > float(ref["p99_bytes"])):
            events.append({**common, "kind": "application_surge",
                           "reason": f"{app} bytes {cur_bytes / usual:.1f}x usual "
                                     f"({_fmt(cur_bytes)} vs {_fmt(usual)})"})

    # Per application on each router pair.
    cur = edges.groupby(["application", "ingress", "egress"])["bytes"].agg(["sum", "size"])
    joined = cur.join(base.pairs, how="left")
    seen = joined["windows"].fillna(0) >= 3
    log_b = np.log1p(joined["sum"].to_numpy(dtype=float))
    std = np.maximum(joined["std_log_bytes"].fillna(0).to_numpy(), cfg.pair_min_std)
    z = (log_b - joined["mean_log_bytes"].fillna(0).to_numpy()) / std
    usual = np.expm1(joined["mean_log_bytes"].fillna(0).to_numpy())
    typed = ~joined.index.get_level_values("application").isin(cfg.untyped)
    surge = seen.to_numpy() & typed & (z >= cfg.pair_surge_z) & (
        joined["sum"].to_numpy() >= cfg.pair_surge_factor * usual)
    for (app, ing, egr), row, u in zip(joined.index[surge], joined[surge].itertuples(),
                                       usual[surge], strict=True):
        events.append({"window": ts, "level": "application_pair", "kind": "app_pair_surge",
                       "application": app, "ingress": ing, "egress": egr, "bytes": row.sum,
                       "usual_bytes": u, "ratio": row.sum / u if u > 0 else np.nan,
                       "flows": int(row.size),
                       "reason": f"{app} on {ing} -> {egr}: {row.sum / max(u, 1):.1f}x usual "
                                 f"bytes ({_fmt(row.sum)} vs {_fmt(u)})"})

    mean_flows = (base.pairs["mean_flows"] if "mean_flows" in base.pairs.columns
                  else pd.Series(np.inf, index=base.pairs.index))
    tracked = base.pairs.index.get_level_values("application").isin(cfg.tracked)
    regular_pairs = base.pairs[tracked
                               & (base.pairs["presence"] >= cfg.pair_regular_presence)
                               & (base.pairs["windows"] >= cfg.pair_min_windows)
                               & (mean_flows >= cfg.pair_min_flows)]
    if len(regular_pairs):
        now = cur["sum"].reindex(regular_pairs.index)
        usual_p = np.expm1(regular_pairs["mean_log_bytes"].to_numpy())
        now_b = now.fillna(0).to_numpy(dtype=float)
        gone = now.isna().to_numpy()
        low = ~gone & (now_b < cfg.pair_drop_fraction * usual_p)
        for mask, kind in ((gone, "app_pair_disappeared"), (low, "app_pair_drop")):
            for (app, ing, egr), b, u in zip(regular_pairs.index[mask], now_b[mask],
                                             usual_p[mask], strict=True):
                text = ("disappeared" if kind == "app_pair_disappeared"
                        else f"{100 * (1 - b / u):.0f}% below usual")
                events.append({"window": ts, "level": "application_pair", "kind": kind,
                               "application": app, "ingress": ing, "egress": egr,
                               "bytes": b, "usual_bytes": u, "ratio": b / u if u > 0 else np.nan,
                               "flows": 0 if kind == "app_pair_disappeared"
                               else int(cur.loc[(app, ing, egr), "size"]),
                               "reason": f"{app} on {ing} -> {egr} {text} "
                                         f"(usually {_fmt(u)} bytes)"})
    return pd.DataFrame(events, columns=EVENT_COLUMNS)
