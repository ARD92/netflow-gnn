"""Cluster anomalies into patterns.

For each window, and for the whole run, anomalous flows and
application-pair events are grouped along each dimension: router pair,
ingress router, egress router, application, service and customer. For every
dimension the most common value is reported with:

- share: the fraction of the anomalies that have this value;
- traffic_share: the fraction of all flows in scope that have it;
- lift: share / traffic_share. A lift well above 1 means the anomalies are
  concentrated there, not just following where most traffic is. "90% of
  anomalies are https" is unremarkable when 85% of all flows are https.

A value is a pattern when its share and its lift both pass the thresholds.
Every dimension's top value is still listed (``is_pattern`` false) so the file
also answers "are all anomalies on the same X?" when the answer is not news.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DIMENSIONS: dict[str, tuple[list[str], str]] = {
    "router_pair": (["ingress", "egress"], "are on router pair"),
    "ingress": (["ingress"], "enter at ingress router"),
    "egress": (["egress"], "leave at egress router"),
    "application": (["application"], "are application"),
    "service": (["service"], "are on service"),
    "customer": (["customer"], "belong to customer"),
}

PATTERN_COLUMNS = ["scope", "dimension", "value", "anomalies", "anomalies_with_dimension",
                   "share", "traffic_share", "lift", "anomalous_bytes", "is_pattern",
                   "statement"]


@dataclass
class PatternConfig:
    min_anomalies: int = 3
    min_share: float = 0.5
    min_lift: float = 2.0


def _items(edges: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Anomalous flows plus application-pair events, one row per anomaly."""
    cols = ["window", "ingress", "egress", "application", "service", "customer", "bytes"]
    flows = edges.loc[edges["is_anomalous"], [c for c in cols if c in edges.columns]]
    pair_events = events.loc[events["level"] == "application_pair",
                             ["window", "ingress", "egress", "application", "bytes"]] \
        if len(events) else pd.DataFrame(columns=cols)
    return pd.concat([flows, pair_events], ignore_index=True)


def _value(key, cols: list[str]) -> str:
    return " -> ".join(map(str, key)) if len(cols) > 1 else str(key)


def find_patterns(edges: pd.DataFrame, events: pd.DataFrame,
                  cfg: PatternConfig | None = None) -> pd.DataFrame:
    """Top value per dimension for each window and for the whole run."""
    cfg = cfg or PatternConfig()
    items = _items(edges, events)
    rows: list[dict] = []
    scopes = [("all windows", items, edges)]
    for ts in sorted(items["window"].unique()) if len(items) else []:
        scopes.append((f"{pd.Timestamp(ts):%Y-%m-%d %H:%M}", items[items["window"] == ts],
                       edges[edges["window"] == ts]))

    for scope, its, traffic in scopes:
        if len(its) < cfg.min_anomalies:
            continue
        for dim, (cols, phrase) in DIMENSIONS.items():
            if not all(c in its.columns and c in traffic.columns for c in cols):
                continue
            known = its.dropna(subset=cols)
            known = known[(known[cols] != "").all(axis=1)]
            if known.empty:
                continue
            counts = known.groupby(cols).agg(n=("bytes", "size"), b=("bytes", "sum"))
            key = counts["n"].idxmax()
            n, nbytes = int(counts.loc[key, "n"]), float(counts.loc[key, "b"])
            match = np.ones(len(traffic), dtype=bool)
            for c, v in zip(cols, key if len(cols) > 1 else (key,), strict=True):
                match &= (traffic[c] == v).to_numpy()
            share = n / len(known)
            traffic_share = match.mean() if len(traffic) else 0.0
            lift = share / traffic_share if traffic_share > 0 else np.inf
            is_pattern = share >= cfg.min_share and lift >= cfg.min_lift
            value = _value(key, cols)
            lead = "All" if share == 1.0 else f"{share:.0%} of"
            rows.append({
                "scope": scope, "dimension": dim, "value": value, "anomalies": n,
                "anomalies_with_dimension": len(known), "share": share,
                "traffic_share": traffic_share, "lift": lift, "anomalous_bytes": nbytes,
                "is_pattern": is_pattern,
                "statement": f"{lead} {len(known)} anomalies {phrase} {value} "
                             f"({traffic_share:.1%} of flows)",
            })
    out = pd.DataFrame(rows, columns=PATTERN_COLUMNS)
    # "all windows" first, then windows in time order; patterns before non-patterns.
    out["_run"] = out["scope"] != "all windows"
    out["_dim"] = out["dimension"].map({d: i for i, d in enumerate(DIMENSIONS)})
    out = out.sort_values(["_run", "scope", "is_pattern", "_dim"],
                          ascending=[True, True, False, True])
    return out.drop(columns=["_run", "_dim"]).reset_index(drop=True)
