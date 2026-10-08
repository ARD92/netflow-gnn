"""Enrichment: destination port -> application, IP prefix -> ASN / customer / service.

Port mapping
------------
Every flow gets an ``application`` from its dstPort: well-known names (22 ->
ssh, 53 -> dns, 443 -> https, ...), ``any`` for ``*``, and ``other-well-known``
/ ``other-registered`` / ``other-dynamic`` for unlisted ports. A file with
``port`` and ``application`` columns (pipe- or comma-separated) adds to or
overrides the built-in list.

Prefix mapping
--------------
A pipe-delimited file with the header
``PREFIX|ASN|ASN_CUSTOMER|SERVICE|IP_MODE`` maps prefixes to an ASN, a
customer (ASN_CUSTOMER) and services. Each flow prefix is matched to the most
specific (longest) file prefix that contains it, IPv4 and IPv6. A prefix
listed with several services (one row each) gets them joined with ``;``.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

import numpy as np
import pandas as pd

PORT_APPLICATIONS: dict[int, str] = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    67: "dhcp", 68: "dhcp", 69: "tftp", 80: "http", 88: "kerberos", 110: "pop3",
    119: "nntp", 123: "ntp", 135: "msrpc", 137: "netbios", 138: "netbios",
    139: "netbios", 143: "imap", 161: "snmp", 162: "snmp-trap", 179: "bgp",
    389: "ldap", 443: "https", 445: "smb", 465: "smtps", 500: "ike", 514: "syslog",
    515: "printer", 554: "rtsp", 587: "smtp-submission", 636: "ldaps", 853: "dns-over-tls",
    873: "rsync", 989: "ftps", 990: "ftps", 993: "imaps", 995: "pop3s",
    1080: "socks", 1194: "openvpn", 1433: "mssql", 1521: "oracle", 1701: "l2tp",
    1723: "pptp", 1812: "radius", 1813: "radius-acct", 1883: "mqtt", 1935: "rtmp",
    2049: "nfs", 2083: "cpanel", 2181: "zookeeper", 3128: "http-proxy", 3268: "ldap-gc",
    3306: "mysql", 3389: "rdp", 3478: "stun", 4500: "ipsec-nat-t", 5060: "sip",
    5061: "sips", 5222: "xmpp", 5353: "mdns", 5432: "postgres", 5671: "amqps",
    5672: "amqp", 5900: "vnc", 6379: "redis", 6443: "kubernetes-api", 8080: "http-alt",
    8443: "https-alt", 8883: "mqtts", 9092: "kafka", 9200: "elasticsearch",
    11211: "memcached", 27017: "mongodb",
}


def load_port_map(path: str | Path | None) -> dict[int, str]:
    """Built-in port map plus overrides from a ``port,application`` file."""
    ports = dict(PORT_APPLICATIONS)
    if path is None:
        return ports
    table = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    table.columns = [c.strip().lower() for c in table.columns]
    if not {"port", "application"} <= set(table.columns):
        raise ValueError(f"{path}: needs 'port' and 'application' columns")
    for port, app in zip(table["port"], table["application"], strict=True):
        ports[int(port)] = app.strip()
    return ports


def save_port_overrides(path_in: str | Path | None, model_dir: str | Path) -> None:
    """Keep the port file used for the baselines with the model, so inference matches it."""
    if path_in is not None:
        target = Path(model_dir) / "port_map.csv"
        target.write_text(Path(path_in).read_text())


def model_port_map(model_dir: str | Path, override: str | Path | None = None) -> dict[int, str]:
    """Port map for a model: explicit file, else the one saved with the model, else built-in."""
    if override is not None:
        return load_port_map(override)
    saved = Path(model_dir) / "port_map.csv"
    return load_port_map(saved if saved.exists() else None)


def _application(port: str, ports: dict[int, str]) -> str:
    port = str(port).strip()
    if port in ("*", "", "0"):
        return "any"
    try:
        num = int(float(port))
    except ValueError:
        return "any"
    if num in ports:
        return ports[num]
    if num < 1024:
        return "other-well-known"
    return "other-registered" if num < 49152 else "other-dynamic"


def applications(dst_ports: pd.Series, ports: dict[int, str]) -> np.ndarray:
    """Application name for each dstPort value."""
    codes, uniques = pd.factorize(dst_ports)
    if not len(uniques):
        return np.array([], dtype=object)
    names = np.array([_application(u, ports) for u in uniques], dtype=object)
    return names[codes]


class PrefixEnricher:
    """Longest-prefix match of flow prefixes against a prefix -> ASN/customer/service file."""

    COLUMNS = ["asn", "customer", "service", "ip_mode", "matched_prefix"]

    def __init__(self, table: pd.DataFrame) -> None:
        table = table.rename(columns=lambda c: c.strip().upper())
        if "PREFIX" not in table.columns:
            raise ValueError("Prefix file needs a PREFIX column")
        for col in ("ASN", "ASN_CUSTOMER", "SERVICE", "IP_MODE"):
            if col not in table.columns:
                table[col] = ""
        table = table.apply(lambda s: s.str.strip())
        grouped = table.groupby("PREFIX", sort=False).agg(
            asn=("ASN", "first"),
            customer=("ASN_CUSTOMER", "first"),
            service=("SERVICE", lambda s: ";".join(sorted({v for v in s if v}))),
            ip_mode=("IP_MODE", "first"),
        ).reset_index()

        rows, v4, v6 = [], {}, {}
        for prefix, asn, customer, service, ip_mode in grouped.itertuples(index=False):
            try:
                net = ipaddress.ip_network(prefix, strict=False)
            except ValueError:
                continue
            row = len(rows)
            rows.append((asn, customer, service, ip_mode, str(net)))
            bucket = v4 if net.version == 4 else v6
            bucket.setdefault(net.prefixlen, {})[int(net.network_address)] = row
        self._rows = rows
        self._v4 = {plen: (pd.Index(np.fromiter(d.keys(), dtype=np.uint64, count=len(d))),
                           np.fromiter(d.values(), dtype=np.int64, count=len(d)))
                    for plen, d in sorted(v4.items(), reverse=True)}
        self._v6 = dict(sorted(v6.items(), reverse=True))
        self._memo: dict[str, int] = {}
        # Row table with a trailing empty row for "no match".
        self._table = np.empty((len(rows) + 1, len(self.COLUMNS)), dtype=object)
        for i, row in enumerate([*rows, ("", "", "", "", "")]):
            self._table[i] = row

    @classmethod
    def from_file(cls, path: str | Path) -> PrefixEnricher:
        return cls(pd.read_csv(path, sep="|", dtype=str, keep_default_na=False))

    @property
    def size(self) -> int:
        return len(self._rows)

    def _match_new(self, prefixes: list[str]) -> None:
        """Longest-prefix match for prefixes not seen before (results memoized)."""
        addr4, plen4, idx4 = [], [], []
        for i, prefix in enumerate(prefixes):
            try:
                net = ipaddress.ip_network(prefix, strict=False)
            except ValueError:
                self._memo[prefix] = -1
                continue
            if net.version == 4:
                addr4.append(int(net.network_address))
                plen4.append(net.prefixlen)
                idx4.append(i)
                continue
            found, addr = -1, int(net.network_address)
            for plen, table in self._v6.items():
                if plen > net.prefixlen:
                    continue
                key = addr & (((1 << plen) - 1) << (128 - plen))
                if key in table:
                    found = table[key]
                    break
            self._memo[prefix] = found

        if idx4:
            addr = np.array(addr4, dtype=np.uint64)
            plen = np.array(plen4)
            result = np.full(len(addr), -1, dtype=np.int64)
            for length, (index, values) in self._v4.items():
                todo = (result < 0) & (plen >= length)
                if not todo.any():
                    continue
                mask = np.uint64(((1 << length) - 1) << (32 - length))
                hits = index.get_indexer(addr[todo] & mask)
                sel = np.flatnonzero(todo)[hits >= 0]
                result[sel] = values[hits[hits >= 0]]
            for i, row in zip(idx4, result, strict=True):
                self._memo[prefixes[i]] = int(row)

    def lookup(self, prefixes: pd.Series) -> pd.DataFrame:
        """ASN, customer, service, ip_mode and matched prefix per input prefix ('' if none)."""
        codes, uniques = pd.factorize(prefixes)
        new = [u for u in uniques if u not in self._memo]
        if new:
            self._match_new(new)
        rows = np.array([self._memo[u] for u in uniques], dtype=np.int64)
        picked = (self._table[np.where(rows[codes] >= 0, rows[codes], len(self._rows))]
                  if len(codes) else np.empty((0, len(self.COLUMNS)), dtype=object))
        return pd.DataFrame(picked, columns=self.COLUMNS, index=prefixes.index)


UNKNOWN_CUSTOMERS = {"", "UNKNOWN"}


def enrich_prefixes(edges: pd.DataFrame, enricher: PrefixEnricher) -> pd.DataFrame:
    """Add src_/dst_ asn, customer, service and flow-level ``customer`` / ``service``.

    The flow's customer and service come from the side whose prefix maps to a
    known customer (source first); if neither does, from the source side.
    """
    for side, col in (("src", "srcIpPrefix"), ("dst", "dstIpPrefix")):
        match = enricher.lookup(edges[col])
        edges[f"{side}_asn"] = match["asn"].to_numpy()
        edges[f"{side}_customer"] = match["customer"].to_numpy()
        edges[f"{side}_service"] = match["service"].to_numpy()
    src_known = ~edges["src_customer"].isin(UNKNOWN_CUSTOMERS)
    dst_known = ~edges["dst_customer"].isin(UNKNOWN_CUSTOMERS)
    use_src = src_known | ~dst_known
    edges["customer"] = np.where(use_src, edges["src_customer"], edges["dst_customer"])
    edges["service"] = np.where(use_src, edges["src_service"], edges["dst_service"])
    for col in ("customer", "service"):
        edges[col] = edges[col].replace("", "unmatched")
    return edges
