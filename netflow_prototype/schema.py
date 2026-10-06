"""NetFlow file schema, parsing, and flow aggregation.

Files are pipe-delimited (optionally gzip-compressed) exports with the header:

    time|protocol|ingress|srcIpPrefix|srcPort|srcAs|egress|dstIpPrefix|dstPort|dstAs|
    packetsamplingfactor|vpn_label|src_snrc|dst_snrc|route_miles|rawbytes|rawpackets|bytes|packets

Raw records are aggregated into *flows*. A flow is identified by the router pair it
traverses (ingress -> egress), the source and destination prefixes, and the
destination port. ``dstPort`` may be ``*`` (any port).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMNS = [
    "time", "protocol", "ingress", "srcIpPrefix", "srcPort", "srcAs",
    "egress", "dstIpPrefix", "dstPort", "dstAs", "packetsamplingfactor",
    "vpn_label", "src_snrc", "dst_snrc", "route_miles", "rawbytes",
    "rawpackets", "bytes", "packets",
]

# Columns the model actually needs.
REQUIRED_COLUMNS = [
    "ingress", "srcIpPrefix", "egress", "dstIpPrefix", "dstPort",
    "route_miles", "bytes", "packets",
]
NUMERIC_COLUMNS = ["route_miles", "bytes", "packets"]

# Identity of an aggregated flow (one graph edge).
FLOW_KEY = ["ingress", "egress", "srcIpPrefix", "dstIpPrefix", "dstPort"]


def read_netflow_file(path: str | Path) -> pd.DataFrame:
    """Read one NetFlow export file and return the required columns.

    Numeric columns are coerced to float; rows with unparseable bytes or
    route_miles are dropped.
    """
    df = pd.read_csv(
        path,
        sep="|",
        usecols=lambda c: c in REQUIRED_COLUMNS,
        dtype=str,
        keep_default_na=False,
        compression="infer",
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in FLOW_KEY:
        df[col] = df[col].str.strip()
    df.loc[df["dstPort"] == "", "dstPort"] = "*"

    before = len(df)
    df = df.dropna(subset=["bytes", "route_miles"])
    if len(df) < before:
        logger.debug("%s: dropped %d rows with invalid numerics", path, before - len(df))
    return df


def aggregate_flows(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw records into one row per flow key.

    bytes and packets are summed; route_miles is the bytes-weighted mean
    (falling back to the plain mean when a flow carries zero bytes).
    """
    df = records[FLOW_KEY + NUMERIC_COLUMNS].copy()
    df["_bytes_x_miles"] = df["bytes"] * df["route_miles"]
    agg = (
        df.groupby(FLOW_KEY, sort=False)
        .agg(
            bytes=("bytes", "sum"),
            packets=("packets", "sum"),
            _bm=("_bytes_x_miles", "sum"),
            _miles_mean=("route_miles", "mean"),
            records=("bytes", "size"),
        )
        .reset_index()
    )
    agg["route_miles"] = np.where(
        agg["bytes"] > 0, agg["_bm"] / agg["bytes"].where(agg["bytes"] > 0, 1.0),
        agg["_miles_mean"],
    )
    return agg.drop(columns=["_bm", "_miles_mean"])


def load_window(path: str | Path) -> pd.DataFrame:
    """Read and aggregate one 10-minute file."""
    return aggregate_flows(read_netflow_file(path))


def flow_hash(flows: pd.DataFrame) -> np.ndarray:
    """Stable 64-bit hash of each flow key (deterministic across runs)."""
    return pd.util.hash_pandas_object(flows[FLOW_KEY], index=False).to_numpy(np.uint64)
