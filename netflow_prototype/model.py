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

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from netflow_prototype.graph import (
    NODE_FEATURE_NAMES,
    NUM_PORT_CLASSES,
    TIME_FEATURE_NAMES,
    FeatureStats,
    HashConfig,
    WindowGraph,
)

LOGVAR_MIN, LOGVAR_MAX = -7.0, 5.0


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    id_dim: int = 16
    num_layers: int = 2
    dropout: float = 0.1
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


def scatter_mean(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean of ``values`` rows grouped by ``index`` (pure torch, no PyG needed)."""
    out = values.new_zeros(num_nodes, values.size(1)).index_add_(0, index, values)
    count = values.new_zeros(num_nodes).index_add_(0, index, values.new_ones(index.size(0)))
    return out / count.clamp(min=1.0).unsqueeze(1)


class EdgeConditionedLayer(nn.Module):
    """Bidirectional message passing where messages depend on edge features."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.msg_fwd = _mlp(2 * hidden, hidden, dropout=dropout)  # ingress -> egress
        self.msg_bwd = _mlp(2 * hidden, hidden, dropout=dropout)  # egress -> ingress
        self.update = _mlp(3 * hidden, hidden, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, e: torch.Tensor,
                src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        n = h.size(0)
        m_in = scatter_mean(self.msg_fwd(torch.cat([h[src], e], dim=1)), dst, n)
        m_out = scatter_mean(self.msg_bwd(torch.cat([h[dst], e], dim=1)), src, n)
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

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = batch["src"], batch["dst"]
        t = self.time_enc(batch["time_x"]).unsqueeze(0)

        h = self.node_enc(torch.cat([batch["node_x"], self.router_emb(batch["router_hash"])], 1))
        h = h + t
        e = self.edge_enc(torch.cat([
            self.type_emb(batch["port_class"]),
            self.port_emb(batch["port_hash"]),
            self.src_prefix_emb(batch["src_prefix_hash"]),
            self.dst_prefix_emb(batch["dst_prefix_hash"]),
            batch["prev"],
        ], dim=1))

        for layer in self.layers:
            h = layer(h, e, src, dst)

        out = self.decoder(torch.cat([h[src], h[dst], e, t.expand(e.size(0), -1)], dim=1))
        mu = out[:, :2]
        logvar = out[:, 2:].clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar)).mean()


def to_batch(g: WindowGraph, stats: FeatureStats, device: torch.device | str = "cpu"
             ) -> dict[str, torch.Tensor]:
    """Normalize a WindowGraph and convert it to tensors."""
    t_mean, t_std = stats.target_mean, stats.target_std
    node_x = (g.node_x - np.asarray(stats.node_mean, np.float32)) / np.asarray(
        stats.node_std, np.float32)
    y = (g.y - t_mean) / t_std
    prev = g.prev.copy()
    present = prev[:, 2] > 0
    prev[present, :2] = (prev[present, :2] - t_mean) / t_std
    prev[~present, :2] = 0.0

    def long(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)).to(device)

    def flt(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

    return {
        "src": long(g.src), "dst": long(g.dst),
        "node_x": flt(node_x), "router_hash": long(g.router_hash),
        "port_class": long(g.port_class), "port_hash": long(g.port_hash),
        "src_prefix_hash": long(g.src_prefix_hash),
        "dst_prefix_hash": long(g.dst_prefix_hash),
        "prev": flt(prev), "y": flt(y), "time_x": flt(g.time_x),
    }


@torch.no_grad()
def predict(model: FlowGNN, g: WindowGraph, stats: FeatureStats,
            device: torch.device | str = "cpu") -> dict[str, np.ndarray]:
    """Score every flow in a window.

    Returns z-scores, the combined anomaly score, and expected values in
    original units (bytes, route_miles).
    """
    model.eval()
    batch = to_batch(g, stats, device)
    mu, logvar = model(batch)
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
