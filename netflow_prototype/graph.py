"""Graph construction from aggregated NetFlow windows.

One 10-minute file becomes one directed multigraph:

    Nodes  = routers seen in ``ingress`` or ``egress``
    Edges  = flows (srcIpPrefix -> dstIpPrefix, typed by dstPort) carried
             from the ingress router to the egress router

Edge properties are ``bytes`` and ``route_miles``. The model predicts these
for every edge; the edge's input features are its flow identity (prefixes,
port, flow type) and the values the same flow had in the previous window.

Node features describe the router's structure in the window (fan-in/out,
prefix diversity, flow-type mix). They deliberately exclude current byte
volumes so that an anomalous edge cannot leak its own value into the
context used to predict it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from netflow_prototype.schema import FLOW_KEY, flow_hash

# ---------------------------------------------------------------------------
# Flow typing by destination port
# ---------------------------------------------------------------------------

PORT_CLASSES = [
    "any", "web", "dns", "ntp", "mail", "remote_access", "voip",
    "database", "well_known_other", "registered", "dynamic",
]
NUM_PORT_CLASSES = len(PORT_CLASSES)

_PORT_TO_CLASS = {
    **dict.fromkeys([80, 443, 8080, 8443], "web"),
    **dict.fromkeys([53, 853], "dns"),
    123: "ntp",
    **dict.fromkeys([25, 110, 143, 465, 587, 993, 995], "mail"),
    **dict.fromkeys([22, 23, 3389, 5900], "remote_access"),
    **dict.fromkeys([5060, 5061], "voip"),
    **dict.fromkeys([1433, 1521, 3306, 5432, 6379, 27017], "database"),
}


def port_class(port: str) -> str:
    """Classify a dstPort value into a flow type."""
    port = str(port).strip()
    if port in ("*", "", "0"):
        return "any"
    try:
        num = int(float(port))
    except ValueError:
        return "any"
    if num in _PORT_TO_CLASS:
        return _PORT_TO_CLASS[num]
    if num < 1024:
        return "well_known_other"
    if num < 49152:
        return "registered"
    return "dynamic"


def port_class_index(ports: pd.Series) -> np.ndarray:
    codes, uniques = pd.factorize(ports)
    lookup = np.array([PORT_CLASSES.index(port_class(u)) for u in uniques], dtype=np.int64)
    return lookup[codes] if len(uniques) else np.zeros(0, dtype=np.int64)


def isin_sorted(values: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    """``np.isin`` for a pre-sorted set, via binary search.

    ``np.isin`` sorts both arrays on every call; with tens of millions of
    training flows that took ~35 s per call. Binary search takes well under 1 s.
    """
    if len(sorted_set) == 0:
        return np.zeros(len(values), dtype=bool)
    idx = np.searchsorted(sorted_set, values)
    idx[idx == len(sorted_set)] = 0
    return sorted_set[idx] == values


def bucket_hash(values: pd.Series, buckets: int) -> np.ndarray:
    """Stable hash of string values into ``[0, buckets)``."""
    h = pd.util.hash_pandas_object(values, index=False).to_numpy(np.uint64)
    return (h % np.uint64(buckets)).astype(np.int64)


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------

NODE_FEATURE_NAMES = [
    "log_out_flows", "log_in_flows", "log_out_neighbors", "log_in_neighbors",
    "log_src_prefixes", "log_dst_prefixes",
    *[f"type_frac_{c}" for c in PORT_CLASSES],
    "port_entropy",
]
TIME_FEATURE_NAMES = ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]


@dataclass
class HashConfig:
    """Bucket sizes for hashed identity embeddings."""

    router_buckets: int = 1024
    prefix_buckets: int = 16384
    port_buckets: int = 4096


@dataclass
class WindowGraph:
    """One 10-minute window as a graph. Arrays are raw (un-normalized)."""

    timestamp: datetime
    routers: list[str]
    src: np.ndarray            # (E,) ingress node index
    dst: np.ndarray            # (E,) egress node index
    node_x: np.ndarray         # (N, F) structural node features
    router_hash: np.ndarray    # (N,)
    port_class: np.ndarray     # (E,)
    port_hash: np.ndarray      # (E,)
    src_prefix_hash: np.ndarray  # (E,)
    dst_prefix_hash: np.ndarray  # (E,)
    prev: np.ndarray           # (E, 3) [log1p prev bytes, prev route_miles, prev present]
    y: np.ndarray              # (E, 2) [log1p bytes, route_miles]
    time_x: np.ndarray         # (4,)
    flow_hash: np.ndarray      # (E,) uint64
    flows: pd.DataFrame | None = None  # aggregated flows aligned with edges
    context: np.ndarray | None = None  # (E,) bool: flow feeds router context (None = all)

    @property
    def num_nodes(self) -> int:
        return len(self.routers)

    @property
    def num_edges(self) -> int:
        return len(self.src)


def time_features(ts: datetime) -> np.ndarray:
    minute_of_day = ts.hour * 60 + ts.minute
    tod = 2 * math.pi * minute_of_day / 1440.0
    dow = 2 * math.pi * ts.weekday() / 7.0
    return np.array([math.sin(tod), math.cos(tod), math.sin(dow), math.cos(dow)],
                    dtype=np.float32)


def _node_features(flows: pd.DataFrame, src: np.ndarray, dst: np.ndarray,
                   pclass: np.ndarray, n: int) -> np.ndarray:
    out_flows = np.bincount(src, minlength=n)
    in_flows = np.bincount(dst, minlength=n)

    pairs = np.unique(src.astype(np.int64) * n + dst)
    out_nbrs = np.bincount(pairs // n, minlength=n)
    in_nbrs = np.bincount(pairs % n, minlength=n)

    src_prefixes = pd.Series(flows["srcIpPrefix"].to_numpy()).groupby(src).nunique()
    dst_prefixes = pd.Series(flows["dstIpPrefix"].to_numpy()).groupby(dst).nunique()
    src_pref = np.zeros(n)
    dst_pref = np.zeros(n)
    src_pref[src_prefixes.index.to_numpy()] = src_prefixes.to_numpy()
    dst_pref[dst_prefixes.index.to_numpy()] = dst_prefixes.to_numpy()

    # Flow-type mix over all incident flows.
    type_counts = np.zeros((n, NUM_PORT_CLASSES))
    np.add.at(type_counts, (src, pclass), 1.0)
    np.add.at(type_counts, (dst, pclass), 1.0)
    type_frac = type_counts / np.maximum(type_counts.sum(axis=1, keepdims=True), 1.0)

    # Shannon entropy of raw dstPort usage over incident flows.
    ports = flows["dstPort"].to_numpy()
    inc = pd.DataFrame({"node": np.concatenate([src, dst]),
                        "port": np.concatenate([ports, ports])})
    counts = inc.groupby(["node", "port"]).size()
    probs = counts / counts.groupby(level="node").transform("sum")
    entropy_s = (-(probs * np.log2(probs))).groupby(level="node").sum()
    entropy = np.zeros(n)
    entropy[entropy_s.index.to_numpy()] = entropy_s.to_numpy()

    return np.column_stack([
        np.log1p(out_flows), np.log1p(in_flows),
        np.log1p(out_nbrs), np.log1p(in_nbrs),
        np.log1p(src_pref), np.log1p(dst_pref),
        type_frac, entropy,
    ]).astype(np.float32)


def build_window_graph(
    flows: pd.DataFrame,
    timestamp: datetime,
    prev_flows: pd.DataFrame | None = None,
    hashes: HashConfig | None = None,
    keep_flows: bool = False,
    context_flows: np.ndarray | None = None,
) -> WindowGraph:
    """Convert one aggregated window into a WindowGraph.

    Args:
        flows: Output of ``schema.aggregate_flows`` for this window.
        timestamp: Window timestamp (from the file name).
        prev_flows: Aggregated flows from the immediately preceding window,
            used as lag features. ``None`` when unavailable.
        hashes: Bucket sizes for identity hashing.
        keep_flows: Retain the flow table on the graph for reporting.
        context_flows: SORTED hashes of flows with a training baseline. When given, only
            those flows build the router features and messages; other (new)
            flows are still scored but cannot distort the context used to
            judge established flows.
    """
    hashes = hashes or HashConfig()
    flows = flows.reset_index(drop=True)

    routers = pd.Index(pd.unique(pd.concat([flows["ingress"], flows["egress"]],
                                           ignore_index=True)))
    src = routers.get_indexer(flows["ingress"]).astype(np.int64)
    dst = routers.get_indexer(flows["egress"]).astype(np.int64)
    n = len(routers)

    pclass = port_class_index(flows["dstPort"])
    fhash = flow_hash(flows)
    ctx = (np.ones(len(flows), dtype=bool) if context_flows is None
           else isin_sorted(fhash, context_flows))

    prev = np.zeros((len(flows), 3), dtype=np.float32)
    if prev_flows is not None and len(prev_flows):
        lag = pd.DataFrame(
            {"bytes": prev_flows["bytes"].to_numpy(),
             "route_miles": prev_flows["route_miles"].to_numpy()},
            index=flow_hash(prev_flows),
        )
        lag = lag[~lag.index.duplicated()].reindex(fhash)
        present = lag["bytes"].notna().to_numpy()
        prev[present, 0] = np.log1p(lag["bytes"].to_numpy()[present])
        prev[present, 1] = lag["route_miles"].to_numpy()[present]
        prev[present, 2] = 1.0

    y = np.column_stack([
        np.log1p(flows["bytes"].to_numpy(dtype=np.float64)),
        flows["route_miles"].to_numpy(dtype=np.float64),
    ]).astype(np.float32)

    return WindowGraph(
        timestamp=timestamp,
        routers=list(routers),
        src=src,
        dst=dst,
        node_x=_node_features(flows[ctx].reset_index(drop=True), src[ctx], dst[ctx],
                              pclass[ctx], n),
        router_hash=bucket_hash(pd.Series(routers), hashes.router_buckets),
        port_class=pclass,
        port_hash=bucket_hash(flows["dstPort"], hashes.port_buckets),
        src_prefix_hash=bucket_hash(flows["srcIpPrefix"], hashes.prefix_buckets),
        dst_prefix_hash=bucket_hash(flows["dstIpPrefix"], hashes.prefix_buckets),
        prev=prev,
        y=y,
        time_x=time_features(timestamp),
        flow_hash=fhash,
        flows=flows[FLOW_KEY + ["bytes", "packets", "route_miles"]] if keep_flows else None,
        context=None if context_flows is None else ctx,
    )


# ---------------------------------------------------------------------------
# Normalization statistics (fit on training windows, reused at inference)
# ---------------------------------------------------------------------------

@dataclass
class FeatureStats:
    node_mean: list[float]
    node_std: list[float]
    bytes_mean: float
    bytes_std: float
    miles_mean: float
    miles_std: float

    @classmethod
    def fit(cls, graphs: list[WindowGraph]) -> FeatureStats:
        node_x = np.concatenate([g.node_x for g in graphs])
        y = np.concatenate([g.y for g in graphs])
        node_std = node_x.std(axis=0)
        node_std[node_std < 1e-6] = 1.0
        return cls(
            node_mean=node_x.mean(axis=0).tolist(),
            node_std=node_std.tolist(),
            bytes_mean=float(y[:, 0].mean()),
            bytes_std=float(max(y[:, 0].std(), 1e-6)),
            miles_mean=float(y[:, 1].mean()),
            miles_std=float(max(y[:, 1].std(), 1e-6)),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def target_mean(self) -> np.ndarray:
        return np.array([self.bytes_mean, self.miles_mean], dtype=np.float32)

    @property
    def target_std(self) -> np.ndarray:
        return np.array([self.bytes_std, self.miles_std], dtype=np.float32)
