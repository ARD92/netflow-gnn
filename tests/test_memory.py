"""GPU-memory safeguards: checkpointed chunking, compact batches, OOM back-off."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch

from netflow_prototype import train as train_mod
from netflow_prototype.graph import FeatureStats, build_window_graph
from netflow_prototype.model import (
    FlowGNN,
    ModelConfig,
    gaussian_nll,
    to_batch,
    to_device,
)


SMALL = ModelConfig(hidden_dim=16, id_dim=4, prefix_buckets=256)


def _graph(num_flows: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    routers = [f"R{i}" for i in range(12)]
    flows = pd.DataFrame({
        "ingress": rng.choice(routers, num_flows),
        "egress": rng.choice(routers, num_flows),
        "srcIpPrefix": [f"10.{i % 50}.0.0/24" for i in range(num_flows)],
        "dstIpPrefix": [f"192.0.{i % 40}.0/24" for i in range(num_flows)],
        "dstPort": rng.choice(["443", "53", "*", "8080"], num_flows),
        "bytes": rng.lognormal(15, 1, num_flows),
        "packets": rng.lognormal(8, 1, num_flows),
        "route_miles": rng.uniform(10, 900, num_flows),
    })
    return build_window_graph(flows, datetime(2026, 10, 6, 12), hashes=SMALL.hashes)


def _model_and_batch():
    torch.manual_seed(0)
    g = _graph()
    stats = FeatureStats.fit([g])
    model = FlowGNN(SMALL).eval()
    return model, to_device(to_batch(g, stats), "cpu")


def test_chunked_forward_matches_full_forward():
    model, batch = _model_and_batch()
    with torch.no_grad():
        mu_full, lv_full = model(batch)
        mu_chunk, lv_chunk = model(batch, chunk_size=37)
    assert torch.allclose(mu_full, mu_chunk, atol=1e-5)
    assert torch.allclose(lv_full, lv_chunk, atol=1e-5)


def test_chunked_training_gradients_match_full_window():
    """Checkpointed chunks must give the same loss and gradients as one pass."""
    model, batch = _model_and_batch()  # eval(): dropout off, gradients still recorded

    def grads(chunk_size):
        model.zero_grad()
        loss = gaussian_nll(*model(batch, chunk_size), batch["y"])
        loss.backward()
        return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()
                             if p.grad is not None}

    loss_full, g_full = grads(None)
    loss_chunk, g_chunk = grads(53)
    assert loss_chunk == pytest.approx(loss_full, rel=1e-5)
    assert g_full.keys() == g_chunk.keys()
    for name in g_full:
        assert torch.allclose(g_full[name], g_chunk[name], atol=1e-5), name


def test_compact_batch_is_int32_and_widened_on_move():
    g = _graph()
    batch = to_batch(g, FeatureStats.fit([g]))
    assert batch["src"].dtype == torch.int32
    assert to_device(batch, "cpu")["src"].dtype == torch.int64


def test_back_off_halves_then_gives_up():
    assert train_mod._back_off(200_000, "x") == 100_000
    with pytest.raises(RuntimeError, match="max-gpu-mem-gb"):
        train_mod._back_off(8_000, "x")


def test_predict_with_backoff_retries_on_oom(monkeypatch):
    model, _ = _model_and_batch()
    g = _graph()
    stats = FeatureStats.fit([g])
    real_predict = train_mod.predict
    calls = []

    def fake_predict(model, g, stats, device, chunk_size):
        calls.append(chunk_size)
        if chunk_size > 100_000:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_predict(model, g, stats, device, chunk_size)

    monkeypatch.setattr(train_mod, "predict", fake_predict)
    cfg = train_mod.TrainConfig(chunk_size=400_000)
    out = train_mod.predict_with_backoff(model, g, stats, cfg)
    assert calls == [400_000, 200_000, 100_000]
    assert cfg.chunk_size == 100_000
    assert len(out["score"]) == g.num_edges
