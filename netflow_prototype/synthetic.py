"""Synthetic NetFlow generator in the production file schema.

Produces ``netflow.YYYYMMDD.HH.MM.txt.gz`` files every 10 minutes with the
same 19 pipe-delimited columns as the real exports, plus label sidecars in
``<out_dir>/labels/netflow.YYYYMMDD.HH.MM.labels.csv`` (flow key +
anomaly_type) for evaluation. Router names, prefixes and AS numbers are
fictitious.

Normal traffic: a fixed population of flows with diurnal volume, per-flow
noise and stable route_miles per router pair. Anomalies are injected only in
the final ``anomaly_hours`` so that the earlier period can be used for
training:

    volume_spike   all flows to one destination prefix grow 8-25x
    route_change   one ingress/egress pair is rerouted (+400-1200 route miles)
    traffic_drop   all flows to one egress router fall to 2-6% of normal
    port_scan      one source prefix opens ~40 small flows on random high ports
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from netflow_prototype.schema import COLUMNS, FLOW_KEY
from netflow_prototype.windows import DEFAULT_INTERVAL, file_name

logger = logging.getLogger(__name__)

ANOMALY_TYPES = ["volume_spike", "route_change", "traffic_drop", "port_scan"]
_PORTS = ["443", "80", "*", "53", "123", "25", "22", "8080", "REG", "DYN"]
_PORT_WEIGHTS = [0.38, 0.12, 0.15, 0.08, 0.03, 0.02, 0.03, 0.03, 0.12, 0.04]
_DST_AS = [15169, 16509, 32934, 36692, 45102, 396982, 2906, 13335, 8075, 20940]
_SRC_AS = [64777, 65535, 2386]
_ROUTER_ROLES = ["CR1", "PE2", "IGX", "ME9"]


@dataclass
class SyntheticConfig:
    start: datetime
    hours: float = 48.0
    anomaly_hours: float = 12.0
    anomaly_window_rate: float = 0.35
    num_sites: int = 8
    routers_per_site: int = 3
    src_prefixes_per_router: int = 25
    num_dst_prefixes: int = 150
    num_flows: int = 3000
    presence: float = 0.95
    seed: int = 7


def _random_prefix(rng: np.random.Generator, min_len: int, max_len: int) -> str:
    length = int(rng.integers(min_len, max_len + 1))
    addr = int(rng.integers(1 << 24, 223 << 24))
    net = ipaddress.ip_network((addr, length), strict=False)
    return str(net)


class _Topology:
    """Fixed routers, prefixes, and persistent flow population."""

    def __init__(self, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
        sites = [f"SITE{chr(65 + i)}" for i in range(cfg.num_sites)]
        coords = rng.uniform(0, 1500, size=(cfg.num_sites, 2))
        self.routers: list[str] = []
        self.router_site: list[int] = []
        for s, site in enumerate(sites):
            for k in range(cfg.routers_per_site):
                self.routers.append(f"{site}{401 + k}{_ROUTER_ROLES[k % len(_ROUTER_ROLES)]}")
                self.router_site.append(s)
        self.sites = sites
        n_r = len(self.routers)

        # route_miles per router pair: path stretch over great-circle-like distance.
        site_xy = coords[self.router_site]
        dist = np.linalg.norm(site_xy[:, None, :] - site_xy[None, :, :], axis=2)
        stretch = rng.uniform(1.05, 1.3, size=(n_r, n_r))
        self.pair_miles = dist * stretch + rng.uniform(5, 30, size=(n_r, n_r))

        self.src_prefixes = [
            [_random_prefix(rng, 21, 25) for _ in range(cfg.src_prefixes_per_router)]
            for _ in range(n_r)
        ]
        self.dst_prefixes = [_random_prefix(rng, 16, 24) for _ in range(cfg.num_dst_prefixes)]
        self.dst_home = rng.integers(0, n_r, size=cfg.num_dst_prefixes)
        self.dst_as = rng.choice(_DST_AS, size=cfg.num_dst_prefixes)

        n = cfg.num_flows
        ingress = rng.integers(0, n_r, size=n)
        dst_idx = np.minimum(rng.zipf(1.3, size=n) - 1, cfg.num_dst_prefixes - 1)
        dst_idx = rng.permutation(cfg.num_dst_prefixes)[dst_idx]
        ports = rng.choice(_PORTS, size=n, p=_PORT_WEIGHTS)
        ports = np.where(ports == "REG", rng.integers(1024, 49152, size=n).astype(str), ports)
        ports = np.where(ports == "DYN", rng.integers(49152, 65536, size=n).astype(str), ports)
        base = rng.normal(18.5, 1.2, size=n) - np.isin(ports, ["53", "123"]) * 2.0

        flows = pd.DataFrame({
            "ingress_i": ingress,
            "egress_i": self.dst_home[dst_idx],
            "srcIpPrefix": [self.src_prefixes[r][rng.integers(len(self.src_prefixes[r]))]
                            for r in ingress],
            "dst_i": dst_idx,
            "dstPort": ports,
            "log_bytes": base,
            "srcAs": rng.choice(_SRC_AS, size=n),
        })
        flows["dstIpPrefix"] = np.asarray(self.dst_prefixes)[flows["dst_i"]]
        self.flows = flows.drop_duplicates(
            subset=["ingress_i", "egress_i", "srcIpPrefix", "dst_i", "dstPort"]
        ).reset_index(drop=True)


def _diurnal(ts: datetime) -> float:
    hour = ts.hour + ts.minute / 60.0
    weekend = 0.85 if ts.weekday() >= 5 else 1.0
    return weekend * (1.0 + 0.45 * np.sin(2 * np.pi * (hour - 9.0) / 24.0))


def _window_frame(topo: _Topology, rng: np.random.Generator, ts: datetime,
                  presence: float) -> pd.DataFrame:
    f = topo.flows[rng.random(len(topo.flows)) < presence].copy()
    router_noise = rng.lognormal(0.0, 0.05, size=len(topo.routers))
    f["bytes"] = (np.exp(f["log_bytes"]) * _diurnal(ts) * router_noise[f["ingress_i"]]
                  * rng.lognormal(0.0, 0.25, size=len(f)))
    f["route_miles"] = (topo.pair_miles[f["ingress_i"], f["egress_i"]]
                        * rng.uniform(0.995, 1.005, size=len(f)))
    return f


def _inject(kind: str, f: pd.DataFrame, topo: _Topology, rng: np.random.Generator,
            target: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Apply an anomaly to a window; return the frame and a mask of affected rows."""
    if kind == "volume_spike":
        mask = f["dst_i"] == target["dst_i"]
        f.loc[mask, "bytes"] *= target["factor"]
    elif kind == "route_change":
        mask = (f["ingress_i"] == target["ingress_i"]) & (f["egress_i"] == target["egress_i"])
        f.loc[mask, "route_miles"] += target["extra_miles"]
    elif kind == "traffic_drop":
        mask = f["egress_i"] == target["egress_i"]
        f.loc[mask, "bytes"] *= target["factor"]
    else:  # port_scan
        n = 40
        dst_idx = rng.integers(0, len(topo.dst_prefixes), size=n)
        scan = pd.DataFrame({
            "ingress_i": target["ingress_i"],
            "egress_i": topo.dst_home[dst_idx],
            "srcIpPrefix": target["src_prefix"],
            "dst_i": dst_idx,
            "dstPort": rng.integers(1024, 65536, size=n).astype(str),
            "log_bytes": 0.0,
            "srcAs": _SRC_AS[0],
            "dstIpPrefix": np.asarray(topo.dst_prefixes)[dst_idx],
            "bytes": rng.lognormal(12.0, 0.5, size=n),
        })
        scan["route_miles"] = topo.pair_miles[scan["ingress_i"], scan["egress_i"]]
        f = pd.concat([f, scan], ignore_index=True)
        mask = pd.Series(False, index=f.index)
        mask.iloc[-n:] = True
    return f, mask


def _pick_target(kind: str, f: pd.DataFrame, topo: _Topology,
                 rng: np.random.Generator) -> dict:
    if kind == "volume_spike":
        counts = f["dst_i"].value_counts()
        candidates = counts[counts >= 3].index.to_numpy()
        return {"dst_i": int(rng.choice(candidates)), "factor": float(rng.uniform(8, 25))}
    if kind == "route_change":
        pairs = f.groupby(["ingress_i", "egress_i"]).size()
        pairs = pairs[pairs >= 3].index.to_numpy()
        ingress_i, egress_i = pairs[rng.integers(len(pairs))]
        return {"ingress_i": int(ingress_i), "egress_i": int(egress_i),
                "extra_miles": float(rng.uniform(400, 1200))}
    if kind == "traffic_drop":
        return {"egress_i": int(rng.integers(len(topo.routers))),
                "factor": float(rng.uniform(0.02, 0.06))}
    ingress_i = int(rng.integers(len(topo.routers)))
    prefixes = topo.src_prefixes[ingress_i]
    return {"ingress_i": ingress_i, "src_prefix": prefixes[rng.integers(len(prefixes))]}


def _to_records(f: pd.DataFrame, topo: _Topology, rng: np.random.Generator,
                ts: datetime) -> pd.DataFrame:
    routers = np.asarray(topo.routers)
    sites = np.asarray([topo.sites[s] for s in topo.router_site])
    out = pd.DataFrame({
        "time": f"{ts:%Y-%m-%d %H:%M:%S}.000",
        "protocol": rng.choice([1, 6, 17], size=len(f), p=[0.5, 0.35, 0.15]),
        "ingress": routers[f["ingress_i"]],
        "srcIpPrefix": f["srcIpPrefix"].to_numpy(),
        "srcPort": 0,
        "srcAs": f["srcAs"].to_numpy(),
        "egress": routers[f["egress_i"]],
        "dstIpPrefix": f["dstIpPrefix"].to_numpy(),
        "dstPort": f["dstPort"].to_numpy(),
        "dstAs": topo.dst_as[f["dst_i"]],
        "packetsamplingfactor": 4096,
        "vpn_label": "",
        "src_snrc": sites[f["ingress_i"]],
        "dst_snrc": sites[f["egress_i"]],
        "route_miles": f["route_miles"].round(2).to_numpy(),
        "bytes": f["bytes"].round(1).to_numpy(),
    })
    out["packets"] = (out["bytes"] / rng.uniform(400, 1400, size=len(out))).round(1)
    out["rawbytes"] = (out["bytes"] / 4096 / 350).round(1).clip(lower=1.0)
    out["rawpackets"] = (out["packets"] / 4096 / 350).round(1).clip(lower=1.0)

    # Split ~20% of flows into two records to exercise aggregation.
    split = rng.random(len(out)) < 0.2
    if split.any():
        extra = out[split].copy()
        frac = rng.uniform(0.2, 0.8, size=len(extra))
        for col in ["bytes", "packets"]:
            extra[col] = (out.loc[split, col].to_numpy() * frac).round(1)
            out.loc[split, col] = (out.loc[split, col].to_numpy() * (1 - frac)).round(1)
        out = pd.concat([out, extra], ignore_index=True)
    return out[COLUMNS].sample(frac=1.0, random_state=int(rng.integers(1 << 31)))


def generate(cfg: SyntheticConfig, out_dir: str | Path) -> dict:
    """Write synthetic NetFlow files and label sidecars; return a summary."""
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(out_dir)
    labels_dir = out_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    topo = _Topology(cfg, rng)
    num_windows = int(round(cfg.hours * 60 / 10))
    anomaly_start = cfg.start + timedelta(hours=cfg.hours - cfg.anomaly_hours)
    active: dict | None = None
    counts = dict.fromkeys(ANOMALY_TYPES, 0)

    for w in range(num_windows):
        ts = cfg.start + w * DEFAULT_INTERVAL
        f = _window_frame(topo, rng, ts, cfg.presence)

        if active is None and ts >= anomaly_start and rng.random() < cfg.anomaly_window_rate:
            kind = ANOMALY_TYPES[int(rng.integers(len(ANOMALY_TYPES)))]
            duration = 1 if kind == "port_scan" else int(rng.integers(1, 4))
            active = {"kind": kind, "left": duration,
                      "target": _pick_target(kind, f, topo, rng)}
            counts[kind] += 1

        labels = pd.DataFrame(columns=[*FLOW_KEY, "anomaly_type"])
        if active is not None:
            f, mask = _inject(active["kind"], f, topo, rng, active["target"])
            hit = f[mask]
            labels = pd.DataFrame({
                "ingress": np.asarray(topo.routers)[hit["ingress_i"]],
                "egress": np.asarray(topo.routers)[hit["egress_i"]],
                "srcIpPrefix": hit["srcIpPrefix"].to_numpy(),
                "dstIpPrefix": hit["dstIpPrefix"].to_numpy(),
                "dstPort": hit["dstPort"].to_numpy(),
                "anomaly_type": active["kind"],
            })
            active["left"] -= 1
            if active["left"] == 0:
                active = None

        records = _to_records(f, topo, rng, ts)
        records.to_csv(out_dir / file_name(ts), sep="|", index=False, compression="gzip")
        if len(labels):
            labels.drop_duplicates().to_csv(
                labels_dir / f"netflow.{ts:%Y%m%d.%H.%M}.labels.csv", index=False)

    summary = {
        "files": num_windows,
        "start": str(cfg.start),
        "end": str(cfg.start + num_windows * DEFAULT_INTERVAL),
        "anomaly_period_start": str(anomaly_start),
        "routers": len(topo.routers),
        "flows": len(topo.flows),
        "anomalies_started": counts,
    }
    logger.info("Synthetic data: %s", summary)
    return summary
