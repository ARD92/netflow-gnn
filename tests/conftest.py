"""Shared fixtures."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

HEADER = ("time|protocol|ingress|srcIpPrefix|srcPort|srcAs|egress|dstIpPrefix|dstPort|dstAs|"
          "packetsamplingfactor|vpn_label|src_snrc|dst_snrc|route_miles|rawbytes|rawpackets|"
          "bytes|packets")

# Fictitious rows in the production format.
ROWS = [
    "2026-10-06 11:50:00.000|1|SITEA401CR1|10.1.0.0/24|0|64777|SITEB401PE2|192.0.2.0/24|443|15169|4096||SITEA|SITEB|21.39|264.0|1.0|300000000.0|1000.0",
    "2026-10-06 11:50:00.000|6|SITEA401CR1|10.1.0.0/24|0|64777|SITEB401PE2|192.0.2.0/24|443|15169|4096||SITEA|SITEB|23.39|100.0|1.0|100000000.0|500.0",
    "2026-10-06 11:50:00.000|1|SITEA401CR1|10.1.4.0/22|0|64777|SITEB401PE2|198.51.100.53/32|*|796|4096||SITEA|SITEB|21.39|216.0|1.0|375000000.0|1736250.0",
    "2026-10-06 11:50:00.000|1|SITEC402PE2|10.2.0.0/22|0|65535|SITEB401PE2|203.0.113.0/24|53|36692|4096||SITEC|SITEB|140.36|84.0|1.0|375000000.0|4464375.0",
    "2026-10-06 11:50:00.000|1|SITEC402PE2|10.2.0.0/22|0|65535|SITED403IGX|198.18.0.0/16|771|396982|4096||SITEC|SITED|587.17|54.0|1.0|375000000.0|6944625.0",
]


def write_file(path: Path, rows: list[str] = ROWS) -> Path:
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join([HEADER, *rows]) + "\n")
    return path


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    return write_file(tmp_path / "netflow.20261006.11.50.txt.gz")
