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
    """Read one NetFlow export or snapshot parquet file and return the required columns.

    Numeric columns are coerced to float; rows with unparseable bytes or
    route_miles are dropped.
    """
    path_str = str(path)
    if path_str.endswith(".parquet"):
        df = pd.read_parquet(path)
        if "ingress" not in df.columns and "src_router" in df.columns:
            df["ingress"] = df["src_router"]
        if "egress" not in df.columns and "dst_router" in df.columns:
            df["egress"] = df["dst_router"]
        if "router_pair" in df.columns and ("ingress" not in df.columns or "egress" not in df.columns):
            pairs = df["router_pair"].astype(str).str.split(" -> ", expand=True)
            if "ingress" not in df.columns:
                df["ingress"] = pairs[0]
            if "egress" not in df.columns:
                df["egress"] = pairs[1]
        if "srcIpPrefix" not in df.columns:
            df["srcIpPrefix"] = "*"
        if "dstIpPrefix" not in df.columns:
            df["dstIpPrefix"] = "*"
        if "dstPort" not in df.columns:
            if "application" in df.columns:
                _app_map = {
                    "https": "443", "http": "80", "dns": "53", "ssh": "22",
                    "snmp": "161", "bgp": "179", "sip": "5060", "ntp": "123",
                    "dhcp": "67", "radius": "1812", "diameter": "3868",
                    "gtp-u": "2152", "gtp-c": "2123", "pfcp": "8805", "ngap": "38412",
                }
                df["dstPort"] = df["application"].astype(str).str.lower().map(_app_map).fillna("*")
            else:
                df["dstPort"] = "*"
        if "packets" not in df.columns:
            df["packets"] = 1
        if "route_miles" not in df.columns:
            # Deterministic pseudo-distance (50 to 2000 miles) per router pair
            pair_hash = pd.util.hash_pandas_object(df[["ingress", "egress"]], index=False).to_numpy(np.uint64)
            df["route_miles"] = 50.0 + (pair_hash % 19500) / 10.0
    else:
        try:
            df = pd.read_csv(
                path,
                sep="|",
                engine="pyarrow",
                usecols=REQUIRED_COLUMNS,
                dtype=str,
                keep_default_na=False,
                compression="infer",
            )
        except Exception:
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
        df[col] = df[col].astype(str).str.strip()
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
    extra_cols = [c for c in ("customer", "service", "application", "anomaly_type", "is_anomaly")
                  if c in records.columns]
    df = records[FLOW_KEY + NUMERIC_COLUMNS + extra_cols].copy()
    df["_bytes_x_miles"] = df["bytes"] * df["route_miles"]

    agg_dict = {
        "bytes": ("bytes", "sum"),
        "packets": ("packets", "sum"),
        "_bm": ("_bytes_x_miles", "sum"),
        "_miles_mean": ("route_miles", "mean"),
        "records": ("bytes", "size"),
    }
    for c in extra_cols:
        if c == "is_anomaly":
            agg_dict[c] = (c, "any")
        elif c == "anomaly_type":
            agg_dict[c] = (c, lambda s: next((x for x in s if x), ""))
        else:
            agg_dict[c] = (c, "first")

    agg = df.groupby(FLOW_KEY, sort=False).agg(**agg_dict).reset_index()
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
