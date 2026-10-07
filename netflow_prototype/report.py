"""Plain-text anomaly report that is easy to scan in a terminal or editor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from netflow_prototype.rules import STATUS_REPEATED, RuleConfig

RULE = "=" * 78


def _t(value) -> str:
    return "" if pd.isna(value) else f"{pd.Timestamp(value):%Y-%m-%d %H:%M}"


def _flow_line(r) -> str:
    return (f"{r.ingress} -> {r.egress} | {r.srcIpPrefix} -> {r.dstIpPrefix} : "
            f"{r.dstPort} ({r.flow_type})")


def _change_lines(r, edge_thr: float) -> list[str]:
    """Original vs detected values for the metrics that moved."""
    if r.baseline_source == "model":
        origin = "expected (no earlier normal observation)"
    else:
        origin = f"at {_t(r.baseline_time)}"
    lines = []
    if abs(r.z_miles) >= edge_thr or r.status == STATUS_REPEATED:
        lines.append(f"route_miles: {r.baseline_route_miles:,.1f} {origin}  ->  "
                     f"{r.route_miles:,.1f} at {_t(r.window)}  "
                     f"(expected {r.expected_route_miles:,.1f}, z={r.z_miles:+.1f})")
    if abs(r.z_bytes) >= edge_thr:
        lines.append(f"bytes:       {r.baseline_bytes:,.0f} {origin}  ->  "
                     f"{r.bytes:,.0f} at {_t(r.window)}  "
                     f"(expected {r.expected_bytes:,.0f}, z={r.z_bytes:+.1f})")
    return lines


def write_report(
    path: str | Path,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    windows: pd.DataFrame,
    bursts: pd.DataFrame,
    summary: dict,
    rules: RuleConfig,
    meta: dict,
    max_flows_per_window: int = 25,
) -> Path:
    """Write ``anomaly_report.txt``: a summary, then each anomalous window in time order."""
    edge_thr = summary["thresholds"]["edge_score"]
    anomalies = edges[edges["is_anomalous"]]
    new_first = edges[edges["is_new_flow"] & edges["first_seen"]]
    lines = [
        RULE,
        "NetFlow anomaly report",
        RULE,
        f"Model trained on : {meta.get('train_start', '?')} -> {meta.get('train_end', '?')}",
        f"Windows scored   : {_t(windows['window'].min())} -> {_t(windows['window'].max())} "
        f"({len(windows)} windows, {len(edges):,} flows)",
        f"Anomalous windows: {int(windows['is_anomalous'].sum())}",
        f"Anomalous flows  : {len(anomalies):,} "
        f"({int((anomalies['status'] == STATUS_REPEATED).sum())} repeated violations)",
        f"New flows        : {len(new_first):,} first observed (not anomalies), "
        f"{len(bursts)} new-flow burst{'' if len(bursts) == 1 else 's'} "
        f"({rules.new_flow_burst:,}+ from one ingress in one window)",
        f"Flow threshold   : z >= {edge_thr:.1f}; flapping = route_miles reversal within "
        f"{int(rules.flap_window.total_seconds() // 60)} min",
    ]
    if summary.get("missing_windows"):
        lines.append(f"Missing windows  : {', '.join(summary['missing_windows'])}")

    shown_any = False
    for _, w in windows.sort_values("window").iterrows():
        ts = w["window"]
        win_anoms = anomalies[anomalies["window"] == ts].sort_values(
            ["status", "score"], ascending=[True, False])  # repeated_violation sorts after anomaly
        win_bursts = bursts[bursts["window"] == ts]
        if not (w["is_anomalous"] or len(win_anoms) or len(win_bursts)):
            continue
        shown_any = True
        verdict = "ANOMALOUS" if w["is_anomalous"] else "flows flagged"
        lines += ["", RULE,
                  f"{_t(ts)}  [{verdict}]  {int(w['flagged_flows'])} of {int(w['flows']):,} "
                  f"flows flagged, {int(w['new_flows']):,} new flows",
                  RULE]

        repeated = win_anoms[win_anoms["status"] == STATUS_REPEATED]
        single = win_anoms[win_anoms["status"] != STATUS_REPEATED]
        for _, r in win_bursts.iterrows():
            lines += ["", f"[NEW-FLOW BURST] ingress {r['ingress']}: {int(r['new_flows']):,} "
                          f"new flows first seen (threshold {rules.new_flow_burst:,})"]
        for r in repeated.itertuples():
            lines += ["", f"[REPEATED VIOLATION] {_flow_line(r)}",
                      f"    route_miles flapping: {r.miles_history} "
                      f"({int(r.flap_changes)} changes)",
                      f"    anomaly started: {_t(r.anomaly_start_time)}"]
        for r in single.head(max_flows_per_window).itertuples():
            lines += ["", f"[ANOMALY] {_flow_line(r)}   score {r.score:.1f}"]
            lines += [f"    {x}" for x in _change_lines(r, edge_thr)]
            lines.append(f"    anomaly started: {_t(r.anomaly_start_time)}")
        if len(single) > max_flows_per_window:
            lines.append(f"\n... and {len(single) - max_flows_per_window:,} more anomalous "
                         f"flows in this window (see edge_anomalies.csv)")

        flagged_routers = nodes[(nodes["window"] == ts) & nodes["is_anomalous"]]
        if len(flagged_routers):
            lines += ["", "Routers flagged:"]
            for r in flagged_routers.itertuples():
                lines.append(f"    {r.router}: {int(r.flagged_flows)} of {int(r.flows):,} "
                             f"flows flagged"
                             + (f", {int(r.new_flows_first_seen):,} new flows (burst)"
                                if r.new_flow_burst else ""))

    if not shown_any:
        lines += ["", "No anomalies found in the scored windows."]
    lines += ["", "Files: edge_anomalies.csv (all anomalous flows), new_flows.csv, "
                  "node_scores.csv, window_scores.csv, graph_edges.csv"]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path
