from datetime import datetime

import pandas as pd

from netflow_prototype.visualize import render_window, router_pairs, router_sites
from netflow_prototype.schema import load_window


def test_router_pairs_and_sites(sample_file):
    pairs = router_pairs(load_window(sample_file))
    assert len(pairs) == 3  # A->B, C->B, C->D
    assert router_sites(sample_file)["SITEC402PE2"] == "SITEC"


def test_render_with_results(sample_file, tmp_path):
    ts = datetime(2026, 10, 6, 11, 50)
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame([{
        "window": str(ts), "ingress": "SITEC402PE2", "egress": "SITED403IGX",
        "srcIpPrefix": "10.2.0.0/22", "dstIpPrefix": "198.18.0.0/16", "dstPort": "771",
        "score": 9.5, "reason": "route_miles 900.0 vs expected 587.2",
    }]).to_csv(results / "edge_anomalies.csv", index=False)
    pd.DataFrame([{"window": str(ts), "router": "SITED403IGX", "is_anomalous": True}]).to_csv(
        results / "node_scores.csv", index=False)

    out = tmp_path / "graph.svg"
    summary = render_window(sample_file, out, results_dir=results, timestamp=ts)
    assert out.exists() and out.stat().st_size > 0
    assert summary["flagged_pairs_shown"] == 1
    table = pd.read_csv(tmp_path / "graph.pairs.csv")
    assert table.iloc[0]["flagged_flows"] == 1  # flagged pairs listed first


def test_render_single_router(sample_file, tmp_path):
    summary = render_window(sample_file, tmp_path / "g.png", router="SITEA401CR1")
    assert summary["router_pairs"] == 1
