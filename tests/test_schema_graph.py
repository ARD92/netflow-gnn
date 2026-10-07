from datetime import datetime

import numpy as np
import pytest

from netflow_prototype.graph import PORT_CLASSES, build_window_graph, port_class
from netflow_prototype.schema import aggregate_flows, load_window, read_netflow_file


def test_read_file_matches_schema(sample_file):
    df = read_netflow_file(sample_file)
    assert len(df) == 5
    assert df["bytes"].dtype.kind == "f"
    assert "*" in set(df["dstPort"])


def test_aggregation_sums_bytes_and_weights_miles(sample_file):
    flows = aggregate_flows(read_netflow_file(sample_file))
    assert len(flows) == 4  # the first two rows share a flow key
    row = flows[(flows["dstIpPrefix"] == "192.0.2.0/24")].iloc[0]
    assert row["bytes"] == pytest.approx(4e8)
    assert row["route_miles"] == pytest.approx((21.39 * 3 + 23.39 * 1) / 4)


@pytest.mark.parametrize("port,expected", [
    ("*", "any"), ("443", "web"), ("53", "dns"), ("123", "ntp"),
    ("771", "well_known_other"), ("8443", "web"), ("30000", "registered"),
    ("60000", "dynamic"),
])
def test_port_class(port, expected):
    assert port_class(port) == expected


def test_graph_nodes_are_routers_and_edges_are_flows(sample_file):
    flows = load_window(sample_file)
    g = build_window_graph(flows, datetime(2026, 10, 6, 11, 50))
    assert set(g.routers) == {"SITEA401CR1", "SITEB401PE2", "SITEC402PE2", "SITED403IGX"}
    assert g.num_edges == 4
    assert g.y.shape == (4, 2)
    assert np.all(g.prev[:, 2] == 0)  # no previous window
    assert PORT_CLASSES[g.port_class[0]] == "web"


def test_lag_features_use_previous_window(sample_file):
    flows = load_window(sample_file)
    g = build_window_graph(flows, datetime(2026, 10, 6, 12), prev_flows=flows.iloc[:2])
    assert g.prev[:, 2].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert g.prev[0, 0] == pytest.approx(np.log1p(flows["bytes"].iloc[0]))


def test_truncated_gzip_is_skipped(tmp_path, sample_file):
    from netflow_prototype.data import load_graphs
    from netflow_prototype.graph import HashConfig

    good = sample_file
    bad = tmp_path / "netflow.20261006.12.00.txt.gz"
    bad.write_bytes(good.read_bytes()[:-20])  # cut off the gzip trailer
    files = [(datetime(2026, 10, 6, 11, 50), good), (datetime(2026, 10, 6, 12, 0), bad)]
    graphs = load_graphs(files, HashConfig())
    assert [g.timestamp for g in graphs] == [datetime(2026, 10, 6, 11, 50)]


def test_isin_sorted_matches_np_isin():
    from netflow_prototype.graph import isin_sorted

    rng = np.random.default_rng(0)
    seen = np.unique(rng.integers(0, 2**63, 5000).astype(np.uint64))
    values = np.concatenate([seen[::3], rng.integers(0, 2**63, 2000).astype(np.uint64),
                             np.array([0, 2**64 - 1], dtype=np.uint64)])
    assert (isin_sorted(values, seen) == np.isin(values, seen)).all()
    assert not isin_sorted(values, np.zeros(0, dtype=np.uint64)).any()
