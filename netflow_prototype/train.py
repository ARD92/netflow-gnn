"""Training: fit the FlowGNN on a time window and calibrate thresholds."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from netflow_prototype.baselines import WindowBaselines
from netflow_prototype.enrich import load_port_map, save_port_map
from netflow_prototype.data import load_graphs
from netflow_prototype.graph import FeatureStats, WindowGraph
from netflow_prototype.model import (
    DEFAULT_CHUNK,
    FlowGNN,
    ModelConfig,
    gaussian_nll,
    limit_gpu_memory,
    predict,
    to_batch,
    to_device,
)
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
    chunk_size: int = DEFAULT_CHUNK    # flows per chunk (checkpointed); bounds GPU memory
    max_gpu_mem_gb: float | None = 16.0  # hard cap on PyTorch CUDA memory (None = no cap)
    max_flows_per_window: int | None = None
    edge_quantile: float = 0.999      # validation quantile for the edge threshold
    min_edge_threshold: float = 4.0   # floor on the edge z-score threshold
    seed: int = 7
    device: str = "cpu"


def _back_off(size: int, what: str, minimum: int = 5_000) -> int:
    """Halve a size after a CUDA out-of-memory error (raise below ``minimum``)."""
    torch.cuda.empty_cache()
    new = size // 2
    if new < minimum:
        raise RuntimeError(
            f"Out of GPU memory even with {what}={size}. Raise --max-gpu-mem-gb, "
            "lower --hidden-dim, or use --device cpu."
        )
    logger.warning("CUDA out of memory; reducing %s from %d to %d", what, size, new)
    return new


def _split(graphs: list[WindowGraph], val_fraction: float
           ) -> tuple[list[WindowGraph], list[WindowGraph]]:
    """Chronological split: the last windows are held out for validation."""
    if len(graphs) < 4:
        logger.warning("Only %d windows; validating on the training windows.", len(graphs))
        return graphs, graphs
    n_val = max(1, round(len(graphs) * val_fraction))
    return graphs[:-n_val], graphs[-n_val:]


def predict_with_backoff(model: FlowGNN, g: WindowGraph, stats: FeatureStats,
                         cfg: TrainConfig) -> dict[str, np.ndarray]:
    """``predict`` that halves ``cfg.chunk_size`` on CUDA out-of-memory."""
    while True:
        try:
            return predict(model, g, stats, cfg.device, cfg.chunk_size)
        except torch.cuda.OutOfMemoryError:
            cfg.chunk_size = _back_off(cfg.chunk_size, "chunk size")


def calibrate(model: FlowGNN, graphs: list[WindowGraph], stats: FeatureStats,
              cfg: TrainConfig) -> dict[str, float]:
    """Derive edge and window thresholds from (assumed normal) validation windows."""
    scores = [predict_with_backoff(model, g, stats, cfg)["score"] for g in graphs]
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
    cache_dir: str | Path | None = None,
    workers: int = 1,
    port_map: str | Path | None = None,
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
    window_baselines = WindowBaselines(load_port_map(port_map))
    graphs = load_graphs(fileset.files, model_cfg.hashes, context=context,
                         window_observer=window_baselines, cache_dir=cache_dir,
                         workers=workers,
                         max_flows_per_window=cfg.max_flows_per_window, seed=cfg.seed)
    train_graphs, val_graphs = _split(graphs, cfg.val_fraction)
    stats = FeatureStats.fit(train_graphs)
    # Batches stay in host memory; one window at a time is moved to the device.
    train_batches = [to_batch(g, stats) for g in train_graphs]
    val_batches = [to_batch(g, stats) for g in val_graphs]
    limit_gpu_memory(cfg.device, cfg.max_gpu_mem_gb)
    cuda = str(cfg.device).startswith("cuda")

    model = FlowGNN(model_cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger.info("Training on %d windows, validating on %d; %d parameters; "
                "chunks of %d flows", len(train_batches), len(val_batches),
                sum(p.numel() for p in model.parameters()), cfg.chunk_size)

    device_type = "cuda" if str(cfg.device).startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def train_step(cpu_batch: dict) -> float:
        """One optimizer step on a full window, processed in checkpointed chunks."""
        batch = to_device(cpu_batch, cfg.device)
        keep = torch.rand(batch["y"].size(0), 1, device=cfg.device) >= cfg.lag_dropout
        batch["prev"] = batch["prev"] * keep
        
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            mu, logvar = model(batch, cfg.chunk_size)
            loss = gaussian_nll(mu, logvar, batch["y"])
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return loss.item()

    @torch.no_grad()
    def val_step(cpu_batch: dict) -> float:
        batch = to_device(cpu_batch, cfg.device)
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            return gaussian_nll(*model(batch, cfg.chunk_size), batch["y"]).item()

    def with_backoff(step, cpu_batch: dict) -> float:
        while True:
            try:
                return step(cpu_batch)
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                cfg.chunk_size = _back_off(cfg.chunk_size, "chunk size")

    history: list[dict[str, float]] = []
    best_val, best_state, stale = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        model.train()
        losses = []
        for i in rng.permutation(len(train_batches)):
            losses.append(with_backoff(train_step, train_batches[i]))

        model.eval()
        val_losses = [with_backoff(val_step, b) for b in val_batches]
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss})
        peak = (f"  peak_gpu={torch.cuda.max_memory_allocated() / 1024**3:.1f}GiB"
                if cuda else "")
        logger.info("epoch %3d  train_nll=%.4f  val_nll=%.4f%s",
                    epoch, train_loss, val_loss, peak)

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
    save_port_map(load_port_map(port_map), model_dir)
    logger.info("Baselines: %s", window_baselines.save(model_dir))

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
