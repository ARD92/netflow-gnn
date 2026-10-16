"""Rules on scored flows: flapping, new flows, bursts, baselines, graph table, report."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from netflow_prototype.infer import _node_scores, _window_scores
from netflow_prototype.report import write_report
from netflow_prototype.rules import (
    CATEGORY_BLACKHOLE,
    CATEGORY_FIBER_REROUTE,
    CATEGORY_INGRESS_BURST,
    CATEGORY_NORMAL,
    CATEGORY_REROUTE_CONGESTION,
    CATEGORY_ROUTE_FLAPPING,
    CATEGORY_TRAFFIC_BURST,
    STATUS_ANOMALY,
    STATUS_NEW,
    STATUS_NORMAL,
    STATUS_REPEATED,
    RuleConfig,
    add_baselines,
    apply_rules,
    categorize_anomalies,
    graph_edges,
)

T0 = datetime(2026, 10, 6, 10, 0)
TEN = timedelta(minutes=10)


def _row(flow_id, minutes, miles=851.0, nbytes=1e6, flag=False, new=False, prev=None,
         ingress="PE-DAL-01", egress="PE-LAX-01", score=None, z_bytes=0.0, z_miles=None):
    zm = (9.0 if flag else 0.5) if z_miles is None else z_miles
    sc = score if score is not None else max(abs(z_bytes), abs(zm))
    return {
        "window": T0 + timedelta(minutes=minutes), "ingress": ingress, "egress": egress,
        "srcIpPrefix": f"10.0.{flow_id % 250}.0/24", "dstIpPrefix": "203.0.113.0/24",
        "dstPort": "443", "flow_type": "web", "bytes": nbytes, "packets": 10.0,
        "route_miles": miles, "z_bytes": z_bytes, "z_miles": zm,
        "score": sc,
        "expected_bytes": 1e6, "expected_route_miles": 851.0,
        "flow_id": np.uint64(flow_id), "is_new_flow": new, "model_flag": flag,
        "prev_present": prev is not None,
        "prev_bytes": np.nan if prev is None else prev[0],
        "prev_route_miles": np.nan if prev is None else prev[1],
    }


def _edges(rows):
    return pd.DataFrame(rows)


def test_flapping_route_miles_is_a_repeated_violation():
    rows = [
        # Flow 1 flaps low -> high -> low within 20 minutes.
        _row(1, 0, 851), _row(1, 10, 1810, flag=True), _row(1, 20, 852),
        # Flow 2 is stable; flow 3 moves once and stays; flow 4 has a gap.
        _row(2, 0), _row(2, 10), _row(2, 20),
        _row(3, 0, 851), _row(3, 10, 1810, flag=True), _row(3, 20, 1812, flag=True),
        _row(4, 0, 851), _row(4, 20, 1810), _row(4, 30, 851),
    ]
    edges, _ = apply_rules(_edges(rows), RuleConfig())
    status = edges.set_index(["flow_id", "window"])["status"]
    assert status[(1, T0 + 2 * TEN)] == STATUS_REPEATED
    assert status[(1, T0 + TEN)] == STATUS_ANOMALY  # one change is not yet flapping
    assert (edges.loc[edges["flow_id"] == 2, "status"] == STATUS_NORMAL).all()
    assert STATUS_REPEATED not in set(edges.loc[edges["flow_id"].isin([3, 4]), "status"])
    flapped = edges[edges["status"] == STATUS_REPEATED].iloc[0]
    assert flapped["miles_history"] == "10:00 851.0 -> 10:10 1810.0 -> 10:20 852.0"
    assert flapped["is_anomalous"]


def test_min_reversals_requires_repeated_flips():
    once = [_row(1, 0, 851), _row(1, 10, 1810), _row(1, 20, 852)]
    twice = [_row(2, 0, 851), _row(2, 10, 1810), _row(2, 20, 852), _row(2, 30, 1811)]
    cfg = RuleConfig(flap_window=timedelta(minutes=30), flap_min_reversals=2)
    edges, _ = apply_rules(_edges(once + twice), cfg)
    flagged = edges[edges["repeated_violation"]]
    assert list(zip(flagged["flow_id"], flagged["window"])) == [(2, T0 + 3 * TEN)]


def test_small_route_miles_wobble_is_not_flapping():
    rows = [_row(1, 0, 851), _row(1, 10, 870), _row(1, 20, 851)]
    edges, _ = apply_rules(_edges(rows), RuleConfig())
    assert not edges["repeated_violation"].any()


def test_new_flow_is_observed_not_anomalous():
    rows = [_row(1, 0, 1810, flag=True, new=True), _row(2, 0, flag=True)]
    edges, bursts = apply_rules(_edges(rows), RuleConfig())
    by_flow = edges.set_index("flow_id")
    assert by_flow.loc[1, "status"] == STATUS_NEW and not by_flow.loc[1, "is_anomalous"]
    assert by_flow.loc[2, "status"] == STATUS_ANOMALY and by_flow.loc[2, "is_anomalous"]
    assert bursts.empty


def test_burst_of_new_flows_from_one_ingress():
    burst = [_row(1000 + i, 0, new=True, ingress="PE-CHI-01") for i in range(12)]
    again = [_row(1000 + i, 10, new=True, ingress="PE-CHI-01") for i in range(12)]
    other = [_row(5000 + i, 0, new=True, ingress="PE-ATL-02") for i in range(5)]
    edges, bursts = apply_rules(_edges(burst + again + other), RuleConfig(new_flow_burst=10))
    assert bursts.to_dict("records") == [
        {"window": T0, "ingress": "PE-CHI-01", "new_flows": 12}]
    assert edges.loc[edges["window"] == T0 + TEN, "in_new_flow_burst"].sum() == 0

    nodes = _node_scores(edges, bursts)
    chi = nodes[(nodes["router"] == "PE-CHI-01") & (nodes["window"] == T0)].iloc[0]
    assert chi["new_flow_burst"] and chi["is_anomalous"] and chi["new_flows_first_seen"] == 12
    windows = _window_scores(edges, bursts, thr=0.5)
    assert windows.set_index("window").loc[T0, "is_anomalous"]


def test_baseline_reports_original_and_detected_values():
    rows = [
        _row(1, 0, nbytes=100.0), _row(1, 10, nbytes=5000.0, flag=True),
        _row(1, 20, nbytes=5100.0, flag=True),
        _row(2, 0, 1810, flag=True, prev=(900.0, 851.0)),  # baseline from previous file
        _row(3, 0, 1810, flag=True),                       # no observation: model baseline
    ]
    cfg = RuleConfig()
    edges = add_baselines(apply_rules(_edges(rows), cfg)[0], cfg)
    f1 = edges[(edges["flow_id"] == 1) & edges["is_anomalous"]]
    assert (f1["baseline_time"] == T0).all() and (f1["baseline_bytes"] == 100.0).all()
    assert (f1["anomaly_start_time"] == T0 + TEN).all()
    f2 = edges[edges["flow_id"] == 2].iloc[0]
    assert f2["baseline_time"] == T0 - TEN and f2["baseline_route_miles"] == 851.0
    assert f2["baseline_source"] == "observed"
    f3 = edges[edges["flow_id"] == 3].iloc[0]
    assert f3["baseline_source"] == "model" and pd.isna(f3["baseline_time"])
    assert pd.isna(edges[edges["flow_id"] == 1].iloc[0]["baseline_time"])  # normal row


def test_graph_edges_use_a_node_z_node():
    rows = [_row(1, 0, nbytes=100.0), _row(2, 0, nbytes=300.0, flag=True),
            _row(3, 0, egress="PE-NYC-02")]
    edges, _ = apply_rules(_edges(rows), RuleConfig())
    g = graph_edges(edges)
    assert list(g.columns[:3]) == ["window", "a_node", "z_node"]
    lax = g[g["z_node"] == "PE-LAX-01"].iloc[0]
    assert lax["flows"] == 2 and lax["bytes"] == 400.0 and lax["anomalous_flows"] == 1
    assert lax["status"] == "anomalous"


def test_readable_report(tmp_path):
    rows = [_row(1, 0, 851), _row(1, 10, 1810, flag=True), _row(1, 20, 852),
            _row(2, 0, nbytes=100.0), _row(2, 10, nbytes=9e6, flag=True)]
    rows += [_row(1000 + i, 10, new=True, ingress="PE-CHI-01") for i in range(10)]
    cfg = RuleConfig(new_flow_burst=10)
    edges, bursts = apply_rules(_edges(rows), cfg)
    edges = add_baselines(edges, cfg)
    nodes = _node_scores(edges, bursts)
    windows = _window_scores(edges, bursts, thr=0.5)
    summary = {"thresholds": {"edge_score": 4.0}, "missing_windows": []}
    path = write_report(tmp_path / "anomaly_report.txt", edges, nodes, windows, bursts,
                        summary, cfg, {"train_start": "a", "train_end": "b"})
    text = path.read_text()
    assert "[REPEATED VIOLATION]" in text and "10:00 851.0 -> 10:10 1810.0 -> 10:20 852.0" in text
    assert "[NEW-FLOW BURST] ingress PE-CHI-01: 10 new flows" in text
    assert "route_miles: 851.0 at 2026-10-06 10:00  ->  1,810.0 at 2026-10-06 10:10" in text


def test_anomaly_categorization_buckets():
    rows = [
        _row(1, 0, flag=True, z_miles=8.5, z_bytes=0.2),         # Fiber reroute
        _row(2, 0, flag=True, z_miles=0.1, z_bytes=6.0),         # Traffic burst
        _row(3, 0, flag=True, z_miles=0.0, z_bytes=-5.2),        # Blackhole / Drop
        _row(4, 0, flag=True, z_miles=7.0, z_bytes=5.5),         # Reroute with congestion
        _row(5, 0, flag=False, z_miles=0.2, z_bytes=0.3),        # Normal
    ]
    edges, _ = apply_rules(_edges(rows), RuleConfig(edge_threshold=4.0))
    cat = edges.set_index("flow_id")["anomaly_category"]
    assert cat[1] == CATEGORY_FIBER_REROUTE
    assert cat[2] == CATEGORY_TRAFFIC_BURST
    assert cat[3] == CATEGORY_BLACKHOLE
    assert cat[4] == CATEGORY_REROUTE_CONGESTION
    assert cat[5] == CATEGORY_NORMAL


def test_anomaly_categorization_route_flapping():
    rows = [
        _row(1, 0, 851),
        _row(1, 10, 1810, flag=True, z_miles=9.0),
        _row(1, 20, 852),
    ]
    edges, _ = apply_rules(_edges(rows), RuleConfig(edge_threshold=4.0))
    flapped = edges[(edges["flow_id"] == 1) & (edges["window"] == T0 + 2 * TEN)].iloc[0]
    assert flapped["status"] == STATUS_REPEATED
    assert flapped["anomaly_category"] == CATEGORY_ROUTE_FLAPPING


def test_anomaly_categorization_ingress_scan_burst():
    burst = [_row(1000 + i, 0, new=True, ingress="PE-CHI-01") for i in range(12)]
    edges, bursts = apply_rules(_edges(burst), RuleConfig(new_flow_burst=10))
    burst_flows = edges[edges["in_new_flow_burst"]]
    assert len(burst_flows) == 12
    assert (burst_flows["anomaly_category"] == CATEGORY_INGRESS_BURST).all()


def test_report_groups_by_category(tmp_path):
    rows = [
        _row(1, 0, flag=True, z_miles=8.5, z_bytes=0.2),  # Fiber reroute
        _row(2, 0, flag=True, z_miles=0.1, z_bytes=6.0),  # Traffic burst
    ]
    cfg = RuleConfig(edge_threshold=4.0)
    edges, bursts = apply_rules(_edges(rows), cfg)
    edges = add_baselines(edges, cfg)
    nodes = _node_scores(edges, bursts)
    windows = _window_scores(edges, bursts, thr=0.5)
    summary = {"thresholds": {"edge_score": 4.0}, "missing_windows": []}
    path = write_report(tmp_path / "report_cat.txt", edges, nodes, windows, bursts,
                        summary, cfg, {"train_start": "t0", "train_end": "t1"})
    text = path.read_text()
    assert "Categories       :" in text
    assert "Fiber Reroute" in text
    assert "Traffic Burst / Congestion" in text
    assert "-- Category: Fiber Reroute (1 flows) --" in text
    assert "-- Category: Traffic Burst / Congestion (1 flows) --" in text
