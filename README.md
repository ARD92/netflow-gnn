# NetFlow GNN Prototype

GNN-based anomaly detection for 10-minute NetFlow exports in the production
schema. It is modeled on `../gnn-netflow-anomaly`, with two differences: the
model trains for real (PyTorch, no PyG required), and it works on real files
selected by a time window.

## Input schema

Pipe-delimited files named `netflow.YYYYMMDD.HH.MM.txt[.gz]`, one every 10 minutes:

```
time|protocol|ingress|srcIpPrefix|srcPort|srcAs|egress|dstIpPrefix|dstPort|dstAs|packetsamplingfactor|vpn_label|src_snrc|dst_snrc|route_miles|rawbytes|rawpackets|bytes|packets
```

The model uses `ingress`, `egress`, `srcIpPrefix`, `dstIpPrefix`, `dstPort`,
`route_miles`, `bytes` (and `packets` for reporting). `dstPort` may be `*`.

## Graph model

| Element | Definition |
|---------|------------|
| Graph | One 10-minute file |
| Node | A router seen in `ingress` or `egress` |
| Edge | A flow `srcIpPrefix -> dstIpPrefix : dstPort`, directed ingress -> egress (records with the same key are aggregated) |
| Edge properties | `bytes` (summed) and `route_miles` (bytes-weighted mean) |
| Flow type | `dstPort` classified as any (`*`), web, dns, ntp, mail, remote_access, voip, database, well_known_other, registered, dynamic |

**Node features** (structural only): flow fan-out/fan-in, neighbor routers,
distinct src/dst prefixes, flow-type mix, and dstPort entropy, plus a hashed
router-identity embedding. Current byte volumes are excluded on purpose, so an
anomalous flow cannot leak its own value into the context used to predict it.

**Edge input features**: flow type, hashed dstPort, src prefix and dst prefix
embeddings, and the same flow's `bytes`/`route_miles` from the previous window
(lag).

**Network**: two layers of bidirectional edge-conditioned message passing
(ingress->egress and egress->ingress, mean aggregation, residual + LayerNorm),
conditioned on time of day and day of week. A per-edge decoder outputs a
Gaussian (mean, variance) for `log(bytes)` and `route_miles`.

**Training**: Gaussian negative log-likelihood on the training window, with a
chronological validation split (last 15%) and early stopping. Thresholds are
calibrated on validation windows.

**Scoring**:

| Level | Score | Flagged when |
|-------|-------|--------------|
| Flow (edge) | max of \|z\| for bytes and route_miles, where z = (actual - expected) / sigma | score >= edge threshold (validation 99.9th percentile, minimum 4.0) |
| Router (node) | mean of its top-3 incident flow scores | >= 2 flagged flows and (>= 5% of its flows or >= 10% of its bytes) |
| Window (graph) | fraction of flows flagged | above validation mean + 3 sigma |

During inference, a flow flagged in window *t* has its lag hidden in window
*t+1*. Without this, a persistent anomaly becomes its own baseline, and the
recovery window is flagged instead. Flows not seen during training are marked
`is_new_flow`.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. (Optional) synthetic data in the same schema: 48h, anomalies in the final 12h
python -m netflow_prototype generate -o data --start "2026-10-04 00:00" --hours 48 --anomaly-hours 12

# 2. Check which files a time window resolves to ([start, end), 10-minute slots, gaps reported)
python -m netflow_prototype files -d data -s "2026-10-05 11:00" -e "2026-10-05 12:00"

# 3. Train on a time window (--end or --duration)
python -m netflow_prototype train -d data -s "2026-10-04 00:00" --duration 36h -m models/demo

# 4. Infer on a new time window
python -m netflow_prototype infer -m models/demo -d data -s "2026-10-05 12:00" -e "2026-10-06 00:00" -o results/demo

# 4b. ...or on explicit files from a new dataset
python -m netflow_prototype infer -m models/demo -f /path/netflow.20261006.11.50.txt.gz -f /path/netflow.20261006.12.00.txt.gz -o results/new

# Optional: evaluate against label sidecars (synthetic data)
python -m netflow_prototype infer ... --labels-dir data/labels
```

Times accept `YYYY-MM-DD HH:MM`, ISO 8601, or the file style `YYYYMMDD.HH.MM`.
Durations accept `90m`, `6h`, or `2d`. When it exists, the file immediately before the
window is loaded only to provide lag features. On real servers, point
`--data-dir` at the export directory (for example `/data1/netflow-data-for-ai`).

Useful training options: `--epochs`, `--hidden-dim`, `--num-layers`, `--dropout`,
`--max-flows-per-window` (random sample for very large files), and `--device cuda|mps`.

### GPU memory

Production windows can hold millions of flows, and whole-window training on 2.4M
flows needed more than 44 GB of VRAM. Three settings keep memory bounded:

| Option (train and infer) | Default | Effect |
|---|---|---|
| `--max-gpu-mem-gb` | 16 | Hard cap on this process's PyTorch CUDA memory (`0` = no cap). Exceeding it raises an error instead of taking the whole GPU. |
| `--chunk-size` | 250000 | Flows processed per chunk. Training still uses the full window: each chunk is checkpointed (recomputed during backward), so loss and gradients are identical to a single pass. Lower it to use less memory. |
| (automatic) | | On a CUDA out-of-memory error the chunk size is halved and the step is retried. |

Windows stay in host memory and move to the GPU one at a time. Peak VRAM is
about `num_flows x hidden_dim` floats plus one chunk's activations. The training
log reports `peak_gpu=` per epoch so you can tune `--chunk-size` and the cap.

### Outputs

| File | Contents |
|------|----------|
| `models/<name>/model.pt` | Weights, model config, normalization stats, thresholds, training metadata |
| `models/<name>/seen_flows.npy` | Hashes of flows seen in training (for `is_new_flow`) |
| `models/<name>/train_summary.json` | Loss history, thresholds, file list |
| `results/<name>/edge_anomalies.csv` | Flagged flows: actual vs expected bytes and route_miles, z-scores, reason |
| `results/<name>/node_scores.csv` | Per-router, per-window scores |
| `results/<name>/window_scores.csv` | Per-window flagged fraction and verdict |
| `results/<name>/edge_scores.csv` | Every scored flow (with `--all-edges`) |
| `results/<name>/metrics.json` | Evaluation (with `--labels-dir`) |

Example `reason` values: `bytes 18.2x above expected`, `bytes 96% below expected`,
`route_miles 1810.7 vs expected 851.2`, `flow not seen in training`.

## Results on synthetic data

The model was trained on 36 hours (216 windows, about 2,700 flows per window, 24 routers),
then scored on the following 12 hours (72 windows) with injected anomalies.
Results are for chunked training (`--chunk-size 500`) across 5 random seeds.
CPU training took 2-4 minutes per run; inference took about 5 seconds.

| Setting | Flow F1 (5 seeds) | Flow AUC |
|---------|-------------------|----------|
| `dropout=0.0` (default) | 0.978-0.991, mean 0.986 | 0.998-1.000 |
| `dropout=0.1` (earlier default) | 0.68-0.96, mean 0.87 | about 0.985 |

With dropout off, chunked and whole-window training give the same result
(F1 0.991 and 0.990 on the same seed). Dropout made early stopping noisy, so
it was turned off.

Synthetic results show that the pipeline works. They do not predict accuracy on
production traffic.

## Limitations and next steps

- **Training data is assumed mostly normal.** Exclude known incident periods from the training window.
- **Disappearing flows are not scored.** A flow that is absent in a window has no edge. A future step would add expected-but-missing edge detection per router pair.
- **Drift.** Retrain periodically, for example weekly on a rolling 7-14 day window. At least one full day is needed to learn diurnal shape, and a week is better.
- **Hashed identities.** Prefixes, ports, and routers use hash buckets. Increase `prefix_buckets` in `ModelConfig` for very large prefix counts.
- **Very large files.** Each window is one full-batch graph. Use `--max-flows-per-window` or a GPU when windows exceed a few hundred thousand flows.

## Project structure

```
netflow-prototype/
  netflow_prototype/
    schema.py      # file schema, parsing, flow aggregation
    windows.py     # time window -> 10-minute files
    graph.py       # router/flow graph, dstPort flow types, normalization stats
    data.py        # file sequence -> graphs with lag features
    model.py       # FlowGNN (edge-conditioned message passing + Gaussian heads)
    train.py       # training, early stopping, threshold calibration
    infer.py       # scoring, router/window aggregation, evaluation
    synthetic.py   # synthetic data in the production schema, with labels
    cli.py         # generate / files / train / infer
  tests/           # pytest suite (python -m pytest)
  requirements.txt
```
