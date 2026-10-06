"""Training: fit the FlowGNN on a time window and calibrate thresholds."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from netflow_prototype.data import load_graphs
from netflow_prototype.graph import FeatureStats, WindowGraph
from netflow_prototype.model import FlowGNN, ModelConfig, gaussian_nll, predict, to_batch
from netflow_prototype.windows import FileSet

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "model.pt"
SEEN_FLOWS_NAME = "seen_flows.npy"
SUMMARY_NAME = "train_summary.json"


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    patience: int = 5
    lag_dropout: float = 0.3          # randomly hide previous-window values
    max_loss_edges: int = 200_000     # edges sampled for the loss per window
    max_flows_per_window: int | None = None
    edge_quantile: float = 0.999      # validation quantile for the edge threshold
    min_edge_threshold: float = 4.0   # floor on the edge z-score threshold
    seed: int = 7
    device: str = "cpu"


def _split(graphs: list[WindowGraph], val_fraction: float
           ) -> tuple[list[WindowGraph], list[WindowGraph]]:
    """Chronological split: the last windows are held out for validation."""
    if len(graphs) < 4:
        logger.warning("Only %d windows; validating on the training windows.", len(graphs))
        return graphs, graphs
    n_val = max(1, round(len(graphs) * val_fraction))
    return graphs[:-n_val], graphs[-n_val:]


def calibrate(model: FlowGNN, graphs: list[WindowGraph], stats: FeatureStats,
              cfg: TrainConfig) -> dict[str, float]:
    """Derive edge and window thresholds from (assumed normal) validation windows."""
    scores = [predict(model, g, stats, cfg.device)["score"] for g in graphs]
    all_scores = np.concatenate(scores)
    edge_thr = max(cfg.min_edge_threshold, float(np.quantile(all_scores, cfg.edge_quantile)))
    fracs = np.array([(s >= edge_thr).mean() for s in scores])
    window_thr = max(float(fracs.mean() + 3 * fracs.std()), 2 * float(fracs.mean()), 0.002)
    return {
        "edge_score": edge_thr,
        "window_flagged_fraction": window_thr,
        "val_score_p50": float(np.quantile(all_scores, 0.5)),
        "val_score_p99": float(np.quantile(all_scores, 0.99)),
        "val_flagged_fraction_mean": float(fracs.mean()),
    }


def train(
    fileset: FileSet,
    model_dir: str | Path,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    context: tuple | None = None,
) -> dict:
    """Train on every file in ``fileset`` and write the model artifact."""
    model_cfg = model_cfg or ModelConfig()
    cfg = train_cfg or TrainConfig()
    if not fileset.files:
        raise ValueError("No NetFlow files found in the requested time window.")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %d windows (%s -> %s)", len(fileset.files), fileset.start, fileset.end)
    if fileset.missing:
        logger.warning("%d expected 10-minute files are missing in the window",
                       len(fileset.missing))
    graphs = load_graphs(fileset.files, model_cfg.hashes, context=context,
                         max_flows_per_window=cfg.max_flows_per_window, seed=cfg.seed)
    train_graphs, val_graphs = _split(graphs, cfg.val_fraction)
    stats = FeatureStats.fit(train_graphs)
    train_batches = [to_batch(g, stats, cfg.device) for g in train_graphs]
    val_batches = [to_batch(g, stats, cfg.device) for g in val_graphs]

    model = FlowGNN(model_cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger.info("Training on %d windows, validating on %d; %d parameters",
                len(train_batches), len(val_batches),
                sum(p.numel() for p in model.parameters()))

    history: list[dict[str, float]] = []
    best_val, best_state, stale = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for i in rng.permutation(len(train_batches)):
            batch = dict(train_batches[i])
            num_edges = batch["y"].size(0)
            keep = torch.rand(num_edges, 1, device=cfg.device) >= cfg.lag_dropout
            batch["prev"] = batch["prev"] * keep
            mu, logvar = model(batch)
            idx = slice(None)
            if num_edges > cfg.max_loss_edges:
                idx = torch.randperm(num_edges, device=cfg.device)[: cfg.max_loss_edges]
            loss = gaussian_nll(mu[idx], logvar[idx], batch["y"][idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([
                gaussian_nll(*model(b), b["y"]).item() for b in val_batches
            ]))
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss})
        logger.info("epoch %3d  train_nll=%.4f  val_nll=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val - 1e-4:
            best_val, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    thresholds = calibrate(model, val_graphs, stats, cfg)
    logger.info("Edge z-score threshold=%.2f, window flagged-fraction threshold=%.4f",
                thresholds["edge_score"], thresholds["window_flagged_fraction"])

    seen = np.unique(np.concatenate([g.flow_hash for g in graphs]))
    np.save(model_dir / SEEN_FLOWS_NAME, seen)

    meta = {
        "train_start": str(fileset.start),
        "train_end": str(fileset.end),
        "num_windows": len(graphs),
        "num_train_windows": len(train_graphs),
        "num_val_windows": len(val_graphs),
        "missing_windows": [str(t) for t in fileset.missing],
        "files": [p.name for p in fileset.paths],
        "unique_flows": int(len(seen)),
        "train_seconds": round(time.time() - t0, 1),
        "best_val_nll": best_val,
    }
    torch.save({
        "model_config": model_cfg.to_dict(),
        "stats": stats.to_dict(),
        "thresholds": thresholds,
        "meta": meta,
        "state_dict": model.state_dict(),
    }, model_dir / ARTIFACT_NAME)

    summary = {"meta": meta, "thresholds": thresholds, "train_config": asdict(cfg),
               "model_config": model_cfg.to_dict(), "history": history}
    (model_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2))
    logger.info("Model written to %s", model_dir / ARTIFACT_NAME)
    return summary
