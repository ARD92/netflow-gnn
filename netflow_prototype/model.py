"""Edge-attribute GNN for NetFlow anomaly detection (PyTorch).

Architecture
------------
    router features + router id embedding ──► node encoder ─┐
    flow type, dstPort, src/dst prefix embeddings,          │
    previous-window bytes/route_miles ──────► edge encoder ─┤
    time of day / day of week ──────────────► time encoder ─┤
                                                            ▼
                     L x bidirectional edge-conditioned message passing
                     (messages flow ingress->egress and egress->ingress,
                      mean-aggregated per router, residual + LayerNorm)
                                                            ▼
              per-edge decoder [h_ingress, h_egress, edge, time]
                                                            ▼
           Gaussian heads: (mu, log sigma^2) for log(bytes) and route_miles

Training minimizes the Gaussian negative log-likelihood on normal traffic.
At inference, each flow's anomaly score is its largest absolute z-score
``|y - mu| / sigma`` across the two edge properties.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from netflow_prototype.graph import (
    NODE_FEATURE_NAMES,
    NUM_PORT_CLASSES,
    TIME_FEATURE_NAMES,
    FeatureStats,
    HashConfig,
    WindowGraph,
)

logger = logging.getLogger(__name__)

LOGVAR_MIN, LOGVAR_MAX = -7.0, 5.0
DEFAULT_CHUNK = 250_000  # edges per chunk; bounds GPU memory in training and scoring


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    id_dim: int = 16
    num_layers: int = 2
    dropout: float = 0.0  # 0.1 made early stopping noisy and cut F1 from ~0.99 to ~0.94
    router_buckets: int = 1024
    prefix_buckets: int = 16384
    port_buckets: int = 4096

    @property
    def hashes(self) -> HashConfig:
        return HashConfig(self.router_buckets, self.prefix_buckets, self.port_buckets)

    def to_dict(self) -> dict:
        return asdict(self)


def _mlp(in_dim: int, out_dim: int, hidden: int | None = None, dropout: float = 0.0) -> nn.Module:
    hidden = hidden or out_dim
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
    )


def _chunks(num_edges: int, chunk_size: int | None) -> list[slice]:
    """Edge slices of at most ``chunk_size`` (one slice when None)."""
    step = num_edges if not chunk_size else max(1, chunk_size)
    return [slice(i, i + step) for i in range(0, max(num_edges, 1), step)]


def _maybe_checkpoint(enabled: bool, fn, *args):
    """Run ``fn`` with activation checkpointing when ``enabled``.

    Only the inputs and outputs of ``fn`` are kept; its intermediate activations
    are recomputed during backward. Dropout RNG state is preserved, so the
    result and gradients are identical to running ``fn`` directly.
    """
    if enabled:
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


class EdgeConditionedLayer(nn.Module):
    """Bidirectional message passing where messages depend on edge features."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.msg_fwd = _mlp(2 * hidden, hidden, dropout=dropout)  # ingress -> egress
        self.msg_bwd = _mlp(2 * hidden, hidden, dropout=dropout)  # egress -> ingress
        self.update = _mlp(3 * hidden, hidden, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)

    def _partial_sums(self, h: torch.Tensor, e: torch.Tensor, s: torch.Tensor,
                      d: torch.Tensor, w: torch.Tensor | None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-router message sums for one chunk of edges (weighted by ``w``)."""
        zeros = h.new_zeros(h.size(0), h.size(1))
        msg_in = self.msg_fwd(torch.cat([h[s], e], dim=1))
        msg_out = self.msg_bwd(torch.cat([h[d], e], dim=1))
        if w is not None:
            msg_in, msg_out = msg_in * w.unsqueeze(1), msg_out * w.unsqueeze(1)
        part_in = zeros.index_add(0, d, msg_in)
        part_out = zeros.index_add(0, s, msg_out)
        return part_in, part_out

    def forward(self, h: torch.Tensor, e_chunks: list[torch.Tensor],
                edge_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
                src: torch.Tensor, dst: torch.Tensor, ckpt: bool = False,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-aggregate messages per router, one edge chunk at a time.

        Per-edge messages only live inside a chunk; with ``ckpt`` they are
        recomputed in backward, so peak memory is bounded by the chunk size
        rather than the window size, for training as well as inference.
        """
        n = h.size(0)
        sum_in = h.new_zeros(n, h.size(1))
        sum_out = h.new_zeros(n, h.size(1))
        for e, (s, d, w) in zip(e_chunks, edge_chunks, strict=True):
            part_in, part_out = _maybe_checkpoint(ckpt, self._partial_sums, h, e, s, d, w)
            sum_in = sum_in + part_in
            sum_out = sum_out + part_out
        cnt_in = torch.bincount(dst, weights=weights, minlength=n)
        cnt_out = torch.bincount(src, weights=weights, minlength=n)
        cnt_in = cnt_in.clamp(min=1).unsqueeze(1).to(h.dtype)
        cnt_out = cnt_out.clamp(min=1).unsqueeze(1).to(h.dtype)
        m_in, m_out = sum_in / cnt_in, sum_out / cnt_out
        return self.norm(h + self.update(torch.cat([h, m_in, m_out], dim=1)))


class FlowGNN(nn.Module):
    """Predicts a Gaussian over (log bytes, route_miles) for every flow edge."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, d = cfg.hidden_dim, cfg.id_dim
        self.router_emb = nn.Embedding(cfg.router_buckets, d)
        self.src_prefix_emb = nn.Embedding(cfg.prefix_buckets, d)
        self.dst_prefix_emb = nn.Embedding(cfg.prefix_buckets, d)
        self.port_emb = nn.Embedding(cfg.port_buckets, d)
        self.type_emb = nn.Embedding(NUM_PORT_CLASSES, d)

        self.node_enc = _mlp(len(NODE_FEATURE_NAMES) + d, h)
        self.edge_enc = _mlp(4 * d + 3, h)
        self.time_enc = _mlp(len(TIME_FEATURE_NAMES), h)
        self.layers = nn.ModuleList(
            EdgeConditionedLayer(h, cfg.dropout) for _ in range(cfg.num_layers)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4 * h, 2 * h), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 4),
        )

    def _encode_edges(self, port_class: torch.Tensor, port_hash: torch.Tensor,
                      src_prefix: torch.Tensor, dst_prefix: torch.Tensor,
                      prev: torch.Tensor) -> torch.Tensor:
        return self.edge_enc(torch.cat([
            self.type_emb(port_class), self.port_emb(port_hash),
            self.src_prefix_emb(src_prefix), self.dst_prefix_emb(dst_prefix), prev,
        ], dim=1))

    def _decode(self, h: torch.Tensor, e: torch.Tensor, s: torch.Tensor,
                d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([h[s], h[d], e, t.expand(e.size(0), -1)], dim=1))

    def forward(self, batch: dict[str, torch.Tensor], chunk_size: int | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict (mu, logvar) for every edge of a full window.

        ``chunk_size`` bounds peak memory without changing the result: edges are
        processed in chunks, and when gradients are recorded each chunk is
        checkpointed (recomputed in backward). Memory then scales with
        ``num_edges x hidden_dim`` plus one chunk's activations.
        """
        src, dst = batch["src"], batch["dst"]
        slices = _chunks(src.size(0), chunk_size)
        ckpt = torch.is_grad_enabled() and len(slices) > 1
        # Optional per-edge context weight: 0 = scored but excluded from messages.
        weights = batch.get("context")
        edge_chunks = [(src[sl], dst[sl], None if weights is None else weights[sl])
                       for sl in slices]
        t = self.time_enc(batch["time_x"]).unsqueeze(0)

        h = self.node_enc(torch.cat([batch["node_x"], self.router_emb(batch["router_hash"])], 1))
        h = h + t
        e_chunks = [
            _maybe_checkpoint(ckpt, self._encode_edges, batch["port_class"][sl],
                              batch["port_hash"][sl], batch["src_prefix_hash"][sl],
                              batch["dst_prefix_hash"][sl], batch["prev"][sl])
            for sl in slices
        ]

        for layer in self.layers:
            h = layer(h, e_chunks, edge_chunks, src, dst, ckpt, weights)

        out = torch.cat([
            _maybe_checkpoint(ckpt, self._decode, h, e, s, d, t)
            for e, (s, d, _) in zip(e_chunks, edge_chunks, strict=True)
        ])
        mu = out[:, :2]
        logvar = out[:, 2:].clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar)).mean()


EDGE_KEYS = ("src", "dst", "port_class", "port_hash", "src_prefix_hash",
             "dst_prefix_hash", "prev", "y", "context")


def to_device(batch: dict[str, torch.Tensor], device: torch.device | str
              ) -> dict[str, torch.Tensor]:
    """Move a (compact, CPU) batch to ``device``, widening int32 indices to int64."""
    return {k: (v.to(device, non_blocking=True).long() if not v.is_floating_point()
                else v.to(device, non_blocking=True))
            for k, v in batch.items()}


def limit_gpu_memory(device: str, max_gb: float | None) -> None:
    """Cap this process's PyTorch CUDA allocations at ``max_gb`` GiB.

    Allocations past the cap raise ``torch.cuda.OutOfMemoryError`` instead of
    consuming the rest of the GPU.
    """
    if not max_gb or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    fraction = min(1.0, max_gb * 1024**3 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    logger.info("GPU %d memory capped at %.1f GiB of %.1f GiB",
                index, fraction * total / 1024**3, total / 1024**3)


def to_batch(g: WindowGraph, stats: FeatureStats, device: torch.device | str = "cpu"
             ) -> dict[str, torch.Tensor]:
    """Normalize a WindowGraph and convert it to tensors.

    Index tensors are int32 to halve host memory; ``to_device`` widens them.
    """
    t_mean, t_std = stats.target_mean, stats.target_std
    node_x = (g.node_x - np.asarray(stats.node_mean, np.float32)) / np.asarray(
        stats.node_std, np.float32)
    y = (g.y - t_mean) / t_std
    prev = g.prev.copy()
    present = prev[:, 2] > 0
    prev[present, :2] = (prev[present, :2] - t_mean) / t_std
    prev[~present, :2] = 0.0

    def long(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(device)

    def flt(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

    batch = {
        "src": long(g.src), "dst": long(g.dst),
        "node_x": flt(node_x), "router_hash": long(g.router_hash),
        "port_class": long(g.port_class), "port_hash": long(g.port_hash),
        "src_prefix_hash": long(g.src_prefix_hash),
        "dst_prefix_hash": long(g.dst_prefix_hash),
        "prev": flt(prev), "y": flt(y), "time_x": flt(g.time_x),
    }
    if g.context is not None:
        batch["context"] = flt(g.context.astype(np.float32))
    return batch


@torch.no_grad()
def predict(model: FlowGNN, g: WindowGraph, stats: FeatureStats,
            device: torch.device | str = "cpu", chunk_size: int | None = DEFAULT_CHUNK
            ) -> dict[str, np.ndarray]:
    """Score every flow in a window.

    Returns z-scores, the combined anomaly score, and expected values in
    original units (bytes, route_miles).
    """
    model.eval()
    batch = to_device(to_batch(g, stats), device)
    mu, logvar = model(batch, chunk_size)
    sigma = torch.exp(0.5 * logvar)
    z = ((batch["y"] - mu) / sigma).cpu().numpy()
    mu = mu.cpu().numpy() * stats.target_std + stats.target_mean
    return {
        "z_bytes": z[:, 0],
        "z_miles": z[:, 1],
        "score": np.abs(z).max(axis=1),
        "expected_bytes": np.expm1(mu[:, 0]),
        "expected_route_miles": mu[:, 1],
    }
