"""Inference: score a new time window (or explicit files) with a trained model.

Outputs (written to ``out_dir``):
    edge_anomalies.csv  anomalous flows: original (baseline) vs detected time and
                        values, expected values, status (anomaly or
                        repeated_violation) and a reason
    new_flows.csv       flows with no training baseline, at first observation
                        (reported, not anomalies)
    node_scores.csv     per-router scores per window, including new-flow bursts
    window_scores.csv   per-window summary and verdict
    graph_edges.csv     the scored graph: one row per window, a_node (ingress),
                        z_node (egress)
    anomaly_report.txt  readable report (only with ``write_report``)
    edge_scores.csv     every scored flow (only with ``write_all_edges``)
    metrics.json        evaluation against labels (only with ``labels_dir``)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from netflow_prototype import baselines
from netflow_prototype.appbaseline import EVENT_COLUMNS, AppBaseline, AppRuleConfig, detect_app_events
from netflow_prototype.enrich import PrefixEnricher, applications, enrich_prefixes, model_port_map
from netflow_prototype.patterns import PatternConfig, find_patterns
from netflow_prototype.baselines import PairMilesConfig, apply_pair_baseline
from netflow_prototype.data import load_graphs
from netflow_prototype.graph import PORT_CLASSES, FeatureStats, WindowGraph, isin_sorted
from netflow_prototype.model import DEFAULT_CHUNK, FlowGNN, ModelConfig, limit_gpu_memory
from netflow_prototype.report import write_report
from netflow_prototype.rules import (
    STATUS_REPEATED,
    RuleConfig,
    add_baselines,
    apply_rules,
    graph_edges,
)
from netflow_prototype.schema import FLOW_KEY
from netflow_prototype.train import (
    ARTIFACT_NAME,
    SEEN_FLOWS_NAME,
    TrainConfig,
    predict_with_backoff,
)
from netflow_prototype.windows import DEFAULT_INTERVAL, FileSet

logger = logging.getLogger(__name__)

# Column order of edge_anomalies.csv and new_flows.csv (most useful first).
ANOMALY_COLUMNS = [
    "detected_time", "status", "ingress", "egress", "srcIpPrefix", "dstIpPrefix",
    "dstPort", "flow_type", "application", "reason", "score",
    "baseline_time", "baseline_source", "anomaly_start_time",
    "baseline_route_miles", "route_miles", "expected_route_miles", "z_miles",
    "miles_baseline", "usual_miles_low", "usual_miles_high",
    "baseline_bytes", "bytes", "expected_bytes", "z_bytes",
    "flap_changes", "miles_history", "packets", "is_new_flow", "flow_id",
]
# Added when --enrich-prefixes is given.
ENRICHED_COLUMNS = ["service", "customer", "src_customer", "src_service", "src_asn",
                    "dst_customer", "dst_service", "dst_asn"]
NEW_FLOW_COLUMNS = [
    "first_seen_time", "ingress", "egress", "srcIpPrefix", "dstIpPrefix", "dstPort",
    "flow_type", "bytes", "route_miles", "in_new_flow_burst", "flow_id",
]

# Oldest last-normal value used as lag for a flow flagged in the previous window.
LAST_GOOD_MAX_AGE = pd.Timedelta(hours=1)

# A router is flagged when enough of its incident flows (or bytes) are anomalous,
# or when it is the ingress of a new-flow burst.
NODE_MIN_FLAGGED_FLOWS = 2
NODE_FLAGGED_FRACTION = 0.05
NODE_FLAGGED_BYTES_FRACTION = 0.10


@dataclass
class LoadedModel:
    model: FlowGNN
    config: ModelConfig
    stats: FeatureStats
    thresholds: dict[str, float]
    meta: dict
    seen_flows: np.ndarray
    pair_baseline: pd.DataFrame | None = None


def lm_has_pairs(model_dir: str | Path) -> bool:
    return (Path(model_dir) / baselines.PAIR_BASELINE_NAME).exists()


def load_model(model_dir: str | Path, device: str = "cpu") -> LoadedModel:
    model_dir = Path(model_dir)
    artifact = torch.load(model_dir / ARTIFACT_NAME, map_location=device, weights_only=True)
    cfg = ModelConfig(**artifact["model_config"])
    model = FlowGNN(cfg).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    seen_path = model_dir / SEEN_FLOWS_NAME
    seen = np.load(seen_path) if seen_path.exists() else np.zeros(0, dtype=np.uint64)
    if len(seen) > 1 and not np.all(seen[:-1] <= seen[1:]):
        seen = np.sort(seen)  # membership checks rely on a sorted set
    return LoadedModel(model, cfg, FeatureStats(**artifact["stats"]),
                       artifact["thresholds"], artifact["meta"], seen,
                       baselines.load(model_dir))


def _reason(row: pd.Series, thr: float) -> str:
    parts = []
    if row.status == STATUS_REPEATED:
        parts.append(f"route_miles flapping {row.miles_history} (repeated violation)")
    if abs(row.z_bytes) >= thr:
        ratio = row.bytes / max(row.expected_bytes, 1.0)
        if ratio >= 1:
            parts.append(f"bytes {ratio:.1f}x above expected")
        else:
            parts.append(f"bytes {100 * (1 - ratio):.0f}% below expected")
    if abs(row.z_miles) >= thr and row.status != STATUS_REPEATED:
        if row.miles_baseline == "pair":
            parts.append(f"route_miles {row.route_miles:.1f} vs usual "
                         f"{row.expected_route_miles:.1f} for this router pair "
                         f"(range {row.usual_miles_low:.1f}-{row.usual_miles_high:.1f})")
        else:
            parts.append(f"route_miles {row.route_miles:.1f} vs expected "
                         f"{row.expected_route_miles:.1f}")
    if row.is_new_flow:
        parts.append("new flow")
    return "; ".join(parts)


def score_window(lm: LoadedModel, g: WindowGraph, run: TrainConfig,
                 pair_cfg: PairMilesConfig | None = None) -> pd.DataFrame:
    """Return one row per flow with predictions, z-scores, and verdicts.

    ``run`` carries the device and evaluation chunk size (halved on CUDA OOM).
    route_miles is judged against the router pair's usual range when the model
    has a pair baseline (``pair_cfg`` None disables it), else by the model.
    """
    pred = predict_with_backoff(lm.model, g, lm.stats, run)
    df = g.flows.reset_index(drop=True).copy()
    df.insert(0, "window", g.timestamp)
    df.insert(6, "flow_type", np.asarray(PORT_CLASSES)[g.port_class])
    for key, values in pred.items():
        df[key] = values
    apply_pair_baseline(df, lm.pair_baseline if pair_cfg else None,
                        pair_cfg or PairMilesConfig(), lm.thresholds["edge_score"])
    df["flow_id"] = g.flow_hash
    # g.context already holds "has a training baseline" when the graph was built with it.
    known = g.context if g.context is not None else isin_sorted(g.flow_hash, lm.seen_flows)
    df["is_new_flow"] = ~known
    df["model_flag"] = df["score"] >= lm.thresholds["edge_score"]
    # Previous-window values used as lag (zeroed when that window was flagged).
    present = g.prev[:, 2] > 0
    df["prev_present"] = present
    df["prev_bytes"] = np.where(present, np.expm1(g.prev[:, 0]), np.nan)
    df["prev_route_miles"] = np.where(present, g.prev[:, 1], np.nan)
    return df


def _node_scores(edges: pd.DataFrame, bursts: pd.DataFrame) -> pd.DataFrame:
    incident = pd.concat([
        edges[["window", "ingress", "score", "is_anomalous", "bytes"]]
        .rename(columns={"ingress": "router"}).assign(role="ingress"),
        edges[["window", "egress", "score", "is_anomalous", "bytes"]]
        .rename(columns={"egress": "router"}).assign(role="egress"),
    ], ignore_index=True)
    incident["flagged_bytes"] = incident["bytes"].where(incident["is_anomalous"], 0.0)
    grouped = incident.groupby(["window", "router"])
    nodes = grouped.agg(
        flows=("score", "size"),
        flagged_flows=("is_anomalous", "sum"),
        max_score=("score", "max"),
        bytes=("bytes", "sum"),
        flagged_bytes=("flagged_bytes", "sum"),
    )
    nodes["node_score"] = grouped["score"].apply(lambda s: s.nlargest(3).mean())
    nodes["flagged_fraction"] = nodes["flagged_flows"] / nodes["flows"]
    nodes["flagged_bytes_fraction"] = nodes["flagged_bytes"] / nodes["bytes"].clip(lower=1.0)
    nodes = nodes.reset_index()
    burst_counts = bursts.rename(columns={"ingress": "router",
                                          "new_flows": "new_flows_first_seen"})
    nodes = nodes.merge(burst_counts, on=["window", "router"], how="left")
    nodes["new_flow_burst"] = nodes["new_flows_first_seen"].notna()
    nodes["new_flows_first_seen"] = nodes["new_flows_first_seen"].fillna(0).astype(int)
    nodes["is_anomalous"] = nodes["new_flow_burst"] | (
        (nodes["flagged_flows"] >= NODE_MIN_FLAGGED_FLOWS) & (
            (nodes["flagged_fraction"] >= NODE_FLAGGED_FRACTION)
            | (nodes["flagged_bytes_fraction"] >= NODE_FLAGGED_BYTES_FRACTION)
        )
    )
    return (nodes.drop(columns=["flagged_bytes"])
            .sort_values(["window", "node_score"], ascending=[True, False]))


def _window_scores(edges: pd.DataFrame, bursts: pd.DataFrame, thr: float,
                   events: pd.DataFrame | None = None) -> pd.DataFrame:
    edges = edges.assign(flagged_bytes=edges["bytes"].where(edges["is_anomalous"], 0.0),
                         repeated=edges["status"] == STATUS_REPEATED)
    win = edges.groupby("window").agg(
        flows=("score", "size"),
        flagged_flows=("is_anomalous", "sum"),
        repeated_violations=("repeated", "sum"),
        new_flows=("is_new_flow", "sum"),
        max_score=("score", "max"),
        bytes=("bytes", "sum"),
        flagged_bytes=("flagged_bytes", "sum"),
    )
    win["flagged_fraction"] = win["flagged_flows"] / win["flows"]
    win["flagged_bytes_fraction"] = win["flagged_bytes"] / win["bytes"].clip(lower=1.0)
    win["new_flow_bursts"] = bursts.groupby("window").size().reindex(win.index).fillna(0)
    win["new_flow_bursts"] = win["new_flow_bursts"].astype(int)
    counts = (events.groupby("window").size() if events is not None and len(events)
              else pd.Series(dtype=int))
    win["app_events"] = counts.reindex(win.index).fillna(0).astype(int)
    win["is_anomalous"] = ((win["flagged_fraction"] >= thr) | (win["new_flow_bursts"] > 0)
                           | (win["repeated_violations"] > 0) | (win["app_events"] > 0))
    return win.drop(columns=["flagged_bytes"]).reset_index()


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), no sklearn required."""
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _prf(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate(edges: pd.DataFrame, windows: pd.DataFrame, labels_dir: str | Path) -> dict:
    """Compare predictions with ``netflow.YYYYMMDD.HH.MM.labels.csv`` sidecar files."""
    labels_dir = Path(labels_dir)
    frames = []
    for ts in edges["window"].unique():
        path = labels_dir / f"netflow.{pd.Timestamp(ts):%Y%m%d.%H.%M}.labels.csv"
        if path.exists():
            lab = pd.read_csv(path, dtype=str, keep_default_na=False)
            frames.append(lab.assign(window=pd.Timestamp(ts)))
    labels = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=[*FLOW_KEY, "anomaly_type", "window"]))
    labels = labels.drop_duplicates(subset=["window", *FLOW_KEY])

    merged = edges.merge(labels, on=["window", *FLOW_KEY], how="left")
    truth = merged["anomaly_type"].notna().to_numpy()
    pred = merged["is_anomalous"].to_numpy(bool)

    per_type = {
        t: {"flows": int(len(grp)), "recall": float(grp["is_anomalous"].mean())}
        for t, grp in merged[truth].groupby("anomaly_type")
    }
    win_truth = windows["window"].isin(set(labels["window"])).to_numpy()
    return {
        "edge": {"auc": roc_auc(truth, merged["score"].to_numpy()), **_prf(truth, pred),
                 "labeled_anomalous_flows": int(truth.sum()), "flows": int(len(merged))},
        "edge_by_type": per_type,
        "window": {**_prf(win_truth, windows["is_anomalous"].to_numpy(bool)),
                   "labeled_anomalous_windows": int(win_truth.sum()),
                   "windows": int(len(windows))},
    }


def infer(
    model_dir: str | Path,
    fileset: FileSet,
    out_dir: str | Path,
    context: tuple | None = None,
    labels_dir: str | Path | None = None,
    write_all_edges: bool = False,
    device: str = "cpu",
    max_gpu_mem_gb: float | None = 16.0,
    chunk_size: int = DEFAULT_CHUNK,
    rules: RuleConfig | None = None,
    pair_miles: PairMilesConfig | None = None,
    use_pair_baseline: bool = True,
    write_readable_report: bool = False,
    report_max_flows: int = 25,
    cache_dir: str | Path | None = None,
    workers: int = 1,
    port_map: str | Path | None = None,
    enrich_prefixes_file: str | Path | None = None,
    app_rules: AppRuleConfig | None = None,
    patterns_cfg: PatternConfig | None = None,
) -> dict:
    """Score every window in ``fileset`` and write CSV reports."""
    rules = rules or RuleConfig()
    pair_miles = (pair_miles or PairMilesConfig()) if use_pair_baseline else None
    if pair_miles is not None and lm_has_pairs(model_dir):
        logger.info("route_miles judged against the router-pair baseline")
    elif use_pair_baseline:
        logger.warning("No %s in %s: route_miles is judged by the model. Build one with "
                       "the baselines command.", baselines.PAIR_BASELINE_NAME, model_dir)
    if not fileset.files:
        raise ValueError("No NetFlow files found in the requested time window.")
    limit_gpu_memory(device, max_gpu_mem_gb)
    run = TrainConfig(device=device, chunk_size=chunk_size)
    lm = load_model(model_dir, device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_thr = lm.thresholds["edge_score"]

    # Only flows with a training baseline build router context (new flows are scored
    # but cannot distort the context of established flows).
    baseline = lm.seen_flows if len(lm.seen_flows) else None
    # The file just before the window is scored as a warm-up (not reported), so the
    # first requested window gets masked lag and flapping history like the rest.
    files = list(fileset.files)
    warmup = None
    if context is not None:
        files, warmup = [context, *files], context[0]
    graphs = load_graphs(files, lm.config.hashes, keep_flows=True, context_flows=baseline,
                         cache_dir=cache_dir, workers=workers)
    scored: list[pd.DataFrame] = []
    prev_ts, prev_flagged = None, np.zeros(0, dtype=np.uint64)
    # Last normal (unflagged) observation per flow: [log1p(bytes), route_miles, time].
    last_good = pd.DataFrame({"log_bytes": pd.Series(dtype=float),
                              "miles": pd.Series(dtype=float),
                              "time": pd.Series(dtype="datetime64[ns]")},
                             index=pd.Index([], dtype=np.uint64))
    for g in graphs:
        # Anomaly-masked lag: a flow flagged in the previous window must not serve
        # as its own baseline, or persistent anomalies are absorbed and the
        # recovery window is flagged instead. Its lag becomes its last normal
        # value (if seen within LAST_GOOD_MAX_AGE), else it is hidden.
        if prev_ts is not None and g.timestamp - prev_ts == DEFAULT_INTERVAL:
            masked = np.flatnonzero(np.isin(g.flow_hash, prev_flagged))
            g.prev[masked] = 0.0
            good = last_good.reindex(g.flow_hash[masked])
            age = pd.Timestamp(g.timestamp) - pd.to_datetime(good["time"])
            fresh = (age <= LAST_GOOD_MAX_AGE).to_numpy()  # NaT (never normal) -> False
            rows = masked[fresh]
            g.prev[rows, 0] = good["log_bytes"].to_numpy(dtype=float)[fresh]
            g.prev[rows, 1] = good["miles"].to_numpy(dtype=float)[fresh]
            g.prev[rows, 2] = 1.0
        df = score_window(lm, g, run, pair_miles)
        scored.append(df)
        ok = ~df["model_flag"].to_numpy()
        update = pd.DataFrame({"log_bytes": np.log1p(df["bytes"].to_numpy()[ok]),
                               "miles": df["route_miles"].to_numpy()[ok],
                               "time": g.timestamp},
                              index=pd.Index(g.flow_hash[ok], dtype=np.uint64))
        last_good = pd.concat([last_good[~last_good.index.isin(update.index)], update])
        prev_ts, prev_flagged = g.timestamp, g.flow_hash[~ok]
    edges = pd.concat(scored, ignore_index=True)
    edges, bursts = apply_rules(edges, rules)
    edges = add_baselines(edges, rules)
    if warmup is not None:
        edges = edges[edges["window"] != warmup].reset_index(drop=True)
        bursts = bursts[bursts["window"] != warmup].reset_index(drop=True)
        # "First seen" refers to the reported windows.
        edges["first_seen"] = (~edges.sort_values("window", kind="stable")["flow_id"]
                               .duplicated()).reindex(edges.index)

    # Enrichment: application from dstPort; customer/service from prefixes (optional).
    edges["application"] = applications(edges["dstPort"], model_port_map(model_dir, port_map))
    if enrich_prefixes_file:
        enricher = PrefixEnricher.from_file(enrich_prefixes_file)
        logger.info("Enriching prefixes from %s (%d prefixes)", enrich_prefixes_file,
                    enricher.size)
        enrich_prefixes(edges, enricher)

    # Application-level events against the application baselines.
    app_base = AppBaseline.load(model_dir)
    if app_base is None:
        logger.warning("No application baselines in %s: application rules skipped. Build "
                       "them with the baselines command.", model_dir)
        events = pd.DataFrame(columns=EVENT_COLUMNS)
    else:
        found = [detect_app_events(edges[edges["window"] == ts], ts, app_base,
                                   app_rules or AppRuleConfig())
                 for ts in sorted(edges["window"].unique())]
        found = [f for f in found if len(f)]
        events = (pd.concat(found, ignore_index=True) if found
                  else pd.DataFrame(columns=EVENT_COLUMNS))

    nodes = _node_scores(edges, bursts)
    windows = _window_scores(edges, bursts, lm.thresholds["window_flagged_fraction"], events)
    patterns = find_patterns(edges, events, patterns_cfg)

    flagged = edges[edges["is_anomalous"]].sort_values("score", ascending=False).copy()
    flagged["reason"] = [_reason(r, edge_thr) for r in flagged.itertuples()]
    flagged = flagged.rename(columns={"window": "detected_time"})
    flagged = flagged[[c for c in ANOMALY_COLUMNS + ENRICHED_COLUMNS if c in flagged.columns]]
    flagged.to_csv(out_dir / "edge_anomalies.csv", index=False, float_format="%.4f")
    new_flows = edges[edges["is_new_flow"] & edges["first_seen"]]
    new_flows.rename(columns={"window": "first_seen_time"})[NEW_FLOW_COLUMNS].to_csv(
        out_dir / "new_flows.csv", index=False, float_format="%.4f")
    nodes.to_csv(out_dir / "node_scores.csv", index=False, float_format="%.4f")
    windows.to_csv(out_dir / "window_scores.csv", index=False, float_format="%.6f")
    graph_edges(edges).to_csv(out_dir / "graph_edges.csv", index=False, float_format="%.4f")
    events.to_csv(out_dir / "app_anomalies.csv", index=False, float_format="%.4f")
    patterns.to_csv(out_dir / "anomaly_patterns.csv", index=False, float_format="%.4f")
    if write_all_edges:
        edges.to_csv(out_dir / "edge_scores.csv", index=False, float_format="%.4f")

    summary = {
        "windows": len(windows),
        "anomalous_windows": int(windows["is_anomalous"].sum()),
        "flows_scored": int(len(edges)),
        "flows_flagged": int(len(flagged)),
        "repeated_violations": int((flagged["status"] == STATUS_REPEATED).sum()),
        "new_flows_observed": int(len(new_flows)),
        "new_flow_bursts": int(len(bursts)),
        "routers_flagged": int(nodes["is_anomalous"].sum()),
        "app_events": events["kind"].value_counts().to_dict(),
        "patterns": patterns.loc[(patterns["scope"] == "all windows") & patterns["is_pattern"],
                                 "statement"].tolist(),
        "missing_windows": [str(t) for t in fileset.missing],
        "thresholds": lm.thresholds,
        "out_dir": str(out_dir),
    }
    if labels_dir:
        summary["metrics"] = evaluate(edges, windows, labels_dir)
        (out_dir / "metrics.json").write_text(json.dumps(summary["metrics"], indent=2))
    if write_readable_report:
        report = write_report(out_dir / "anomaly_report.txt", edges, nodes, windows, bursts,
                              summary, rules, lm.meta, max_flows_per_window=report_max_flows,
                              events=events, patterns=patterns)
        summary["report"] = str(report)
    return summary
