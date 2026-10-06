"""Inference: score a new time window (or explicit files) with a trained model.

Outputs (written to ``out_dir``):
    edge_anomalies.csv  flagged flows with expected vs. actual values and a reason
    node_scores.csv     per-router scores per window
    window_scores.csv   per-window summary and verdict
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

from netflow_prototype.data import load_graphs
from netflow_prototype.graph import PORT_CLASSES, FeatureStats, WindowGraph
from netflow_prototype.model import FlowGNN, ModelConfig, predict
from netflow_prototype.schema import FLOW_KEY
from netflow_prototype.train import ARTIFACT_NAME, SEEN_FLOWS_NAME
from netflow_prototype.windows import DEFAULT_INTERVAL, FileSet

logger = logging.getLogger(__name__)

# A router is flagged when enough of its incident flows (or bytes) are anomalous.
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


def load_model(model_dir: str | Path, device: str = "cpu") -> LoadedModel:
    model_dir = Path(model_dir)
    artifact = torch.load(model_dir / ARTIFACT_NAME, map_location=device, weights_only=True)
    cfg = ModelConfig(**artifact["model_config"])
    model = FlowGNN(cfg).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    seen_path = model_dir / SEEN_FLOWS_NAME
    seen = np.load(seen_path) if seen_path.exists() else np.zeros(0, dtype=np.uint64)
    return LoadedModel(model, cfg, FeatureStats(**artifact["stats"]),
                       artifact["thresholds"], artifact["meta"], seen)


def _reason(row: pd.Series, thr: float) -> str:
    parts = []
    if abs(row.z_bytes) >= thr:
        ratio = row.bytes / max(row.expected_bytes, 1.0)
        if ratio >= 1:
            parts.append(f"bytes {ratio:.1f}x above expected")
        else:
            parts.append(f"bytes {100 * (1 - ratio):.0f}% below expected")
    if abs(row.z_miles) >= thr:
        parts.append(f"route_miles {row.route_miles:.1f} vs expected "
                     f"{row.expected_route_miles:.1f}")
    if row.is_new_flow:
        parts.append("flow not seen in training")
    return "; ".join(parts)


def score_window(lm: LoadedModel, g: WindowGraph, device: str = "cpu") -> pd.DataFrame:
    """Return one row per flow with predictions, z-scores, and verdicts."""
    pred = predict(lm.model, g, lm.stats, device)
    df = g.flows.reset_index(drop=True).copy()
    df.insert(0, "window", g.timestamp)
    df.insert(6, "flow_type", np.asarray(PORT_CLASSES)[g.port_class])
    for key, values in pred.items():
        df[key] = values
    df["is_new_flow"] = ~np.isin(g.flow_hash, lm.seen_flows)
    df["is_anomalous"] = df["score"] >= lm.thresholds["edge_score"]
    return df


def _node_scores(edges: pd.DataFrame) -> pd.DataFrame:
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
    nodes["is_anomalous"] = (nodes["flagged_flows"] >= NODE_MIN_FLAGGED_FLOWS) & (
        (nodes["flagged_fraction"] >= NODE_FLAGGED_FRACTION)
        | (nodes["flagged_bytes_fraction"] >= NODE_FLAGGED_BYTES_FRACTION)
    )
    return (nodes.drop(columns=["flagged_bytes"]).reset_index()
            .sort_values(["window", "node_score"], ascending=[True, False]))


def _window_scores(edges: pd.DataFrame, thr: float) -> pd.DataFrame:
    edges = edges.assign(flagged_bytes=edges["bytes"].where(edges["is_anomalous"], 0.0))
    win = edges.groupby("window").agg(
        flows=("score", "size"),
        flagged_flows=("is_anomalous", "sum"),
        new_flows=("is_new_flow", "sum"),
        max_score=("score", "max"),
        bytes=("bytes", "sum"),
        flagged_bytes=("flagged_bytes", "sum"),
    )
    win["flagged_fraction"] = win["flagged_flows"] / win["flows"]
    win["flagged_bytes_fraction"] = win["flagged_bytes"] / win["bytes"].clip(lower=1.0)
    win["is_anomalous"] = win["flagged_fraction"] >= thr
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
) -> dict:
    """Score every window in ``fileset`` and write CSV reports."""
    if not fileset.files:
        raise ValueError("No NetFlow files found in the requested time window.")
    lm = load_model(model_dir, device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_thr = lm.thresholds["edge_score"]

    graphs = load_graphs(fileset.files, lm.config.hashes, context=context, keep_flows=True)
    scored: list[pd.DataFrame] = []
    prev_ts, prev_flagged = None, np.zeros(0, dtype=np.uint64)
    for g in graphs:
        # Anomaly-masked lag: a flow flagged in the previous window must not
        # serve as its own baseline, otherwise persistent anomalies are absorbed
        # and the recovery window is flagged instead.
        if prev_ts is not None and g.timestamp - prev_ts == DEFAULT_INTERVAL:
            g.prev[np.isin(g.flow_hash, prev_flagged)] = 0.0
        df = score_window(lm, g, device)
        scored.append(df)
        prev_ts, prev_flagged = g.timestamp, g.flow_hash[df["is_anomalous"].to_numpy()]
    edges = pd.concat(scored, ignore_index=True)
    nodes = _node_scores(edges)
    windows = _window_scores(edges, lm.thresholds["window_flagged_fraction"])

    flagged = edges[edges["is_anomalous"]].sort_values("score", ascending=False).copy()
    flagged["reason"] = [_reason(r, edge_thr) for r in flagged.itertuples()]
    flagged.to_csv(out_dir / "edge_anomalies.csv", index=False, float_format="%.4f")
    nodes.to_csv(out_dir / "node_scores.csv", index=False, float_format="%.4f")
    windows.to_csv(out_dir / "window_scores.csv", index=False, float_format="%.6f")
    if write_all_edges:
        edges.to_csv(out_dir / "edge_scores.csv", index=False, float_format="%.4f")

    summary = {
        "windows": len(windows),
        "anomalous_windows": int(windows["is_anomalous"].sum()),
        "flows_scored": int(len(edges)),
        "flows_flagged": int(len(flagged)),
        "routers_flagged": int(nodes["is_anomalous"].sum()),
        "missing_windows": [str(t) for t in fileset.missing],
        "thresholds": lm.thresholds,
        "out_dir": str(out_dir),
    }
    if labels_dir:
        summary["metrics"] = evaluate(edges, windows, labels_dir)
        (out_dir / "metrics.json").write_text(json.dumps(summary["metrics"], indent=2))
    return summary
