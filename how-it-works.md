# How It Works: NetFlow Edge-Conditioned GNN

This document details the architectural design, mathematical foundations, data processing pipeline, and inference rules powering the **NetFlow Graph Neural Network (`netflow-prototype`)**.

---

## 1. Core Problem & Concept

Standard NetFlow anomaly detection often relies on rigid thresholding, per-flow static baselines, or supervised classification. However, network traffic exhibits high non-stationarity, complex spatial dependencies across routers, and heavy-tailed traffic distributions. Real-world labels for rare network anomalies are seldom available.

To address these challenges, this implementation treats anomaly detection as **unsupervised conditional density estimation** on dynamic multigraphs:
- Network topology and traffic flows are represented as a directed multigraph for every 10-minute export window.
- The GNN predicts a bivariate Gaussian distribution $(\mu, \sigma^2)$ over expected flow metrics: **volume** (`log1p(bytes)`) and **path length** (`route_miles`).
- Conditioning factors include:
  1. Router structural topology (fan-in/fan-out, neighbor counts, protocol mix).
  2. Flow identity (source/destination IP prefix hashes, destination port classes).
  3. Temporal context (cyclical time of day and day of week).
  4. Historical flow lag ($t - 1$ window values).
- Anomaly scoring measures the statistical deviation ($|z|$-score) between observed and predicted values.

---

## 2. Graph Construction & Data Schema

Each 10-minute NetFlow export is converted into a directed multigraph $G = (V, E)$.

### Graph Elements

| Element | Definition | Description |
| :--- | :--- | :--- |
| **Graph ($G$)** | Single 10-minute window | A complete snapshot of network flows across all observed routers. |
| **Node ($v \in V$)** | Router | Any router appearing as an `ingress` or `egress` point. |
| **Edge ($e \in E$)** | Aggregated Flow | Directed edge from `ingress` to `egress` identified by `[ingress, egress, srcIpPrefix, dstIpPrefix, dstPort]`. |
| **Edge Targets ($y$)** | `[log1p(bytes), route_miles]` | Continuous target properties predicted by the GNN decoder. |

### Port Classification & Flow Typing

Destination ports are mapped into 11 semantic classes (`netflow_prototype/graph.py`):
- `any` (`*`, `0`, or blank)
- `web` (80, 443, 8080, 8443)
- `dns` (53, 853)
- `ntp` (123)
- `mail` (25, 110, 143, 465, 587, 993, 995)
- `remote_access` (22, 23, 3389, 5900)
- `voip` (5060, 5061)
- `database` (1433, 1521, 3306, 5432, 6379, 27017)
- `well_known_other` (ports < 1024)
- `registered` (1024 to 49151)
- `dynamic` (49152 to 65535)

Support for 3GPP and 5G telecommunication ports is integrated via `--port-map`:
- 4G/5G interfaces: GTP-U (2152), GTP-C (2123), PFCP (8805), NGAP (38412), XnAP (38422), S1AP (36412), Diameter (3868).

### Feature Engineering & Anti-Leakage Design

```text
               Node Features (Structural Only)
  ┌─────────────────────────────────────────────────────────┐
  │ • log1p(in_flows), log1p(out_flows)                     │
  │ • log1p(in_neighbors), log1p(out_neighbors)             │
  │ • log1p(src_prefixes), log1p(dst_prefixes)              │
  │ • type_frac (11-class port distribution)                │
  │ • dstPort Shannon Entropy                               │
  │ • Hashed router identity embedding                      │
  └─────────────────────────────────────────────────────────┘
        ▲
        │  CURRENT BYTE VOLUMES STRICTLY EXCLUDED
        │  (Prevents anomalous edges from leaking volume into context)
```

1. **Node Features (Strictly Structural):**
   - Outgoing / Incoming flow counts (fan-out / fan-in degree).
   - Unique neighbor counts.
   - Unique source and destination IP prefix counts.
   - Distribution across the 11 port classes (`type_frac_*`).
   - Shannon entropy of `dstPort` usage across all incident flows.
   - **Anti-Leakage Guarantee:** Current-window byte volumes are intentionally excluded from node features so an anomalous flow cannot distort the router context used to judge it.

2. **Edge Features:**
   - One-hot / dense embedding of categorical `port_class`.
   - Hashed identity embeddings for `port_hash` (4,096 buckets), `src_prefix_hash` (16,384 buckets), and `dst_prefix_hash` (16,384 buckets).
   - Lag features from previous window $t-1$: `[log1p(prev_bytes), prev_route_miles, is_present]`.

3. **Time Context Features:**
   - Continuous cyclical sine and cosine embeddings of minute-of-day ($[0, 1440)$) and day-of-week ($[0, 7)$).

---

## 3. GNN Model Architecture (`FlowGNN`)

The network is implemented in native PyTorch (`netflow_prototype/model.py`) without external graph framework dependencies (such as PyTorch Geometric or DGL).

```text
  Router Features + Router ID Emb ──► Node Encoder (MLP) ──┐
                                                           │
  Port class, hashes, previous lag ──► Edge Encoder (MLP) ─┼──► L x Bidirectional
                                                           │    Edge-Conditioned Layers
  Time of day / Day of week ────────► Time Encoder (MLP) ──┘              │
                                                                          ▼
  [h_ingress, h_egress, edge_repr, time_repr] ──────────────► Edge Decoder (MLP)
                                                                          ▼
                                                      Gaussian heads: (mu, log_sigma^2)
                                                      for [log(bytes), route_miles]
```

### Encoders
- **Node Encoder:** 2-layer MLP with GELU activations mapping `[node_features; router_embedding]` to `hidden_dim` ($h$).
- **Edge Encoder:** 2-layer MLP mapping `[type_emb; port_emb; src_emb; dst_emb; lag_features]` to `hidden_dim` ($h$).
- **Time Encoder:** 2-layer MLP mapping 4 temporal cyclical features to `hidden_dim` ($h$), which is added as an inductive bias to node representations.

### Bidirectional Edge-Conditioned Message Passing (`EdgeConditionedLayer`)

Each message passing layer propagates information along graph edges in both directions:

1. **Forward message (ingress $\to$ egress):**
   $$m_{u \to v} = \text{GELU}\left( W_{\text{node\_fwd}} h_u + W_{\text{edge\_fwd}} e_{uv} \right)$$
2. **Backward message (egress $\to$ ingress):**
   $$m_{v \to u} = \text{GELU}\left( W_{\text{node\_bwd}} h_v + W_{\text{edge\_bwd}} e_{uv} \right)$$
3. **Chunked Accumulation:**
   To process millions of edges efficiently, partial message sums are aggregated into node accumulators via `torch.Tensor.index_add`:
   $$\text{sum\_in}_v = \sum_{u \in \mathcal{N}_{\text{in}}(v)} m_{u \to v}, \qquad \text{sum\_out}_u = \sum_{v \in \mathcal{N}_{\text{out}}(u)} m_{v \to u}$$
4. **Mean Aggregation & Node Update:**
   Mean messages $\bar{m}_{\text{in}} = \frac{\text{sum\_in}}{\text{deg}_{\text{in}}}$ and $\bar{m}_{\text{out}} = \frac{\text{sum\_out}}{\text{deg}_{\text{out}}}$ are concatenated with the router's prior state:
   $$h_v^{(l+1)} = \text{LayerNorm}\left( h_v^{(l)} + \text{MLP}([h_v^{(l)}, \bar{m}_{\text{in}}, \bar{m}_{\text{out}}]) \right)$$

### Decoder
For every edge $(u, v)$, the decoder concatenates the final representations of the endpoints, the edge feature, and the global time vector:
$$z_{uv} = [h_u^{(L)}, h_v^{(L)}, e_{uv}, t]$$
A 3-layer MLP decodes $z_{uv}$ into 4 parameters:
$$[\mu_{\text{bytes}}, \mu_{\text{miles}}, \log \sigma^2_{\text{bytes}}, \log \sigma^2_{\text{miles}}]$$
Log-variance is clamped to $[-7.0, 5.0]$ to prevent numerical instability.

---

## 4. Understanding the GNN Model and Weight

With the default configuration (`hidden_dim=64`, `id_dim=16`, `num_layers=2`), `FlowGNN` contains **864,820 total parameters** (~3.46 MB in FP32).

Over **83% of the weights reside in compact embedding tables** (hashed router IDs, IP subnets, and port lookups), while the message-passing and decoding networks comprise a lean, fast computation graph of ~143,000 parameters.

### High-Level Weight Distribution

| Component | Modules | Parameter Shapes | Parameter Count | % of Model |
| :--- | :--- | :--- | :--- | :--- |
| **Identity Embeddings** | `router_emb`, `src_prefix_emb`, `dst_prefix_emb`, `port_emb`, `type_emb` | `[8192, 16]`, `[16384, 16]`, `[4096, 16]`, `[11, 16]` | **721,072** | 83.4% |
| **GNN Message Passing (x2)** | `layers.0`, `layers.1` (`EdgeConditionedLayer`) | Node/edge projections, message MLPs, update MLPs, LayerNorms | **80,128** | 9.3% |
| **Per-Edge Decoder** | `decoder` (3-layer MLP) | `[128, 256]`, `[64, 128]`, `[4, 64]` + biases | **41,412** | 4.8% |
| **Feature Encoders** | `node_enc`, `edge_enc`, `time_enc` | 2-layer MLPs projecting raw inputs to `hidden_dim=64` | **19,392** | 2.2% |
| **Layer Normalization** | `layers.*.norm` | Scale $\gamma$ `[64]` and shift $\beta$ `[64]` | **256** | < 0.1% |
| **Total** | | | **864,820** | **100%** |

### Detailed Layer & Weight Breakdown

#### 1. Identity & Attribute Embeddings (721,072 parameters)
All categorical entities and hashed identity keys are mapped to dense $d=16$ dimensional latent vectors:
- `router_emb.weight` $\to$ `[8192, 16]` ($131,072$ params): Hashed router ID space (8,192 buckets).
- `src_prefix_emb.weight` $\to$ `[16384, 16]` ($262,144$ params): Source subnet hash space (16,384 buckets).
- `dst_prefix_emb.weight` $\to$ `[16384, 16]` ($262,144$ params): Destination subnet hash space (16,384 buckets).
- `port_emb.weight` $\to$ `[4096, 16]` ($65,536$ params): Destination port hash space (4,096 buckets).
- `type_emb.weight` $\to$ `[11, 16]` ($176$ params): 11 semantic port classes (web, dns, telco, etc.).

#### 2. Feature Encoders (19,392 parameters)
Each input domain is projected to `hidden_dim = 64` ($h$) via a 2-layer MLP (`Linear -> GELU -> Dropout -> Linear`):

- **Node Encoder (`node_enc` — 6,400 params):**
  - **Input ($34$ dims):** $18$ structural metrics (degrees, prefix diversity, 11 port fractions, port entropy) $+$ $16$-dim `router_emb`.
  - `node_enc.0`: `Linear(34, 64)` $\to$ `weight: [64, 34]`, `bias: [64]` ($2,240$ params)
  - `node_enc.3`: `Linear(64, 64)` $\to$ `weight: [64, 64]`, `bias: [64]` ($4,160$ params)
- **Edge Encoder (`edge_enc` — 8,512 params):**
  - **Input ($67$ dims):** $4 \times 16$-dim embeddings (`type_emb`, `port_emb`, `src_prefix_emb`, `dst_prefix_emb`) $+$ $3$ lag features (`[log1p(prev_bytes), prev_route_miles, is_present]`).
  - `edge_enc.0`: `Linear(67, 64)` $\to$ `weight: [64, 67]`, `bias: [64]` ($4,352$ params)
  - `edge_enc.3`: `Linear(64, 64)` $\to$ `weight: [64, 64]`, `bias: [64]` ($4,160$ params)
- **Time Encoder (`time_enc` — 4,480 params):**
  - **Input ($4$ dims):** Cyclical $[\sin, \cos]$ of time-of-day and day-of-week.
  - `time_enc.0`: `Linear(4, 64)` $\to$ `weight: [64, 4]`, `bias: [64]` ($320$ params)
  - `time_enc.3`: `Linear(64, 64)` $\to$ `weight: [64, 64]`, `bias: [64]` ($4,160$ params)
  - *Usage:* Injected directly into initial node states: $h_0 = h_{\text{node}} + t$.

#### 3. Message Passing Layers: `EdgeConditionedLayer` (40,064 params per layer $\times 2 = 80,128$ params)
Each layer propagates information bidirectionally along flows using edge features as filters:

```text
Forward Message (Ingress -> Egress):
  h_src [64]  ──► proj_node_fwd [64, 64] ──┐
                                           ├── (+) ──► GELU ──► msg_rest_fwd [64, 64] ──► msg_in
  e_flow [64] ──► proj_edge_fwd [64, 64] ──┘

Backward Message (Egress -> Ingress):
  h_dst [64]  ──► proj_node_bwd [64, 64] ──┐
                                           ├── (+) ──► GELU ──► msg_rest_bwd [64, 64] ──► msg_out
  e_flow [64] ──► proj_edge_bwd [64, 64] ──┘
```

- **Forward Projections:**
  - `proj_node_fwd.weight` $\to$ `[64, 64]` ($4,096$ params, no bias)
  - `proj_edge_fwd.weight` $\to$ `[64, 64]`, `bias: [64]` ($4,160$ params)
  - `msg_rest_fwd.2.weight` $\to$ `[64, 64]`, `bias: [64]` ($4,160$ params)
- **Backward Projections:**
  - `proj_node_bwd.weight` $\to$ `[64, 64]` ($4,096$ params, no bias)
  - `proj_edge_bwd.weight` $\to$ `[64, 64]`, `bias: [64]` ($4,160$ params)
  - `msg_rest_bwd.2.weight` $\to$ `[64, 64]`, `bias: [64]` ($4,160$ params)
- **Router Update MLP (`update` — 16,512 params):**
  - Takes concatenation of router state $h$ ($64$), aggregated incoming messages $m_{\text{in}}$ ($64$), and outgoing messages $m_{\text{out}}$ ($64$) $\implies \mathbf{192}$ dims.
  - `update.0`: `Linear(192, 64)` $\to$ `weight: [64, 192]`, `bias: [64]` ($12,352$ params)
  - `update.3`: `Linear(64, 64)` $\to$ `weight: [64, 64]`, `bias: [64]` ($4,160$ params)
- **LayerNorm (`norm` — 128 params):**
  - `norm.weight` $\to$ `[64]`, `norm.bias` $\to$ `[64]` ($128$ params)
  - Residual update: $h^{(l+1)} = \text{LayerNorm}(h^{(l)} + \text{update}([h^{(l)}, m_{\text{in}}, m_{\text{out}}]))$

#### 4. Per-Edge Gaussian Decoder (41,412 parameters)
Takes the concatenated state $[h_{\text{ingress}}, h_{\text{egress}}, e_{\text{flow}}, t]$ of size $4 \times 64 = \mathbf{256}$ dimensions and decodes it through a 3-layer MLP:

- `decoder.0`: `Linear(256, 128)` $\to$ `weight: [128, 256]`, `bias: [128]` ($32,896$ params)
- `decoder.3`: `Linear(128, 64)` $\to$ `weight: [64, 128]`, `bias: [64]` ($8,192$ params)
- `decoder.5`: `Linear(64, 4)` $\to$ `weight: [4, 64]`, `bias: [4]` ($260$ params)

The **4 scalar outputs** represent:
1. $\mu_{\text{bytes}}$: Predicted mean of $\log(1 + \text{bytes})$
2. $\mu_{\text{miles}}$: Predicted mean of $\text{route\_miles}$
3. $\log \sigma^2_{\text{bytes}}$: Predicted log-variance of flow volume (clamped to $[-7.0, 5.0]$)
4. $\log \sigma^2_{\text{miles}}$: Predicted log-variance of path distance (clamped to $[-7.0, 5.0]$)

---

## 5. Scalability & GPU Memory Management

To scale across millions of flows per window on standard GPUs or Apple Silicon, the implementation uses several memory techniques:

1. **Edge Chunking (`chunk_size=250,000`):**
   Edges are partitioned into contiguous slices. Message projection, aggregation, and edge decoding execute chunk-by-chunk. Peak memory is bounded by the chunk size rather than the total edge count.
2. **Activation Checkpointing:**
   During training, intermediate activations of message-passing chunks are discarded during forward passes and recomputed during backward passes (`torch.utils.checkpoint`). This bounds peak gradient memory to $O(|E| \cdot d_{\text{hidden}} + \text{chunk\_size})$.
3. **Dynamic OOM Backoff (`_back_off`):**
   If a `torch.cuda.OutOfMemoryError` is encountered, the chunk size is halved automatically (e.g., $250k \to 125k \to 62.5k$) and the batch is retried without crashing.
4. **Host-Side Int32 Storage:**
   Node and edge indices are stored in RAM as compact `int32` tensors (halving host RAM requirements) and cast to `int64` via `to_device()` when loaded onto GPU.

---

## 6. Training & Calibration

### Objective Function: Gaussian Negative Log-Likelihood (NLL)
$$\mathcal{L}_{\text{NLL}} = \frac{1}{2} \sum_{i \in \{\text{bytes}, \text{miles}\}} \left( \log \sigma_i^2 + \frac{(y_i - \mu_i)^2}{\sigma_i^2} \right)$$
Optimized using **AdamW** with cosine or plateau learning rate scheduling and gradient clipping ($\|\mathbf{g}\| \le 1.0$).

### Lag Dropout (`lag_dropout=0.3`)
If historical flow values were always available, the network might degenerate into an autoregressive copy mechanism ($y_t \approx y_{t-1}$). To force the GNN to learn spatial topology and routing correlations, lag features are zeroed out with 30% probability during training.

### Chronological Validation & Threshold Calibration
1. **Validation Split:** The final 15% of chronological time windows are held out for validation (no random shuffling, preserving temporal ordering).
2. **Edge Threshold ($\tau_{\text{edge}}$):** Calibrated as the 99.9th percentile ($P_{99.9}$) of all validation $z$-scores, with a minimum floor of $4.0\sigma$.
3. **Window Threshold ($\tau_{\text{win}}$):** Calibrated from validation window anomaly rates as $\mu_{\text{val}} + 3\sigma_{\text{val}}$ (minimum floor 0.2%).

---

## 7. Multi-Tier Anomaly Scoring & Domain Rules

Anomaly detection occurs across multiple tiers: edge, node, and graph window.

```text
               ┌───────────────────────────────┐
               │    Window Graph Evaluation    │
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │  Edge Z-Score & Pair Baseline │
               └───────────────┬───────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                                           ▼
┌─────────────────┐                         ┌─────────────────┐
│ Router Flapping │                         │ New-Flow Bursts │
│   (Reversals)   │                         │  (Ingress Scan) │
└────────┬────────┘                         └────────┬────────┘
         │                                           │
         └─────────────────────┬─────────────────────┘
                               │
               ┌───────────────▼───────────────┐
               │  Multi-Tier Scoring Hierarchy │
               │     (Edge -> Node -> Window)  │
               └───────────────────────────────┘
```

### 1. Router-Pair Empirical Baseline for `route_miles`
Physical distances between router pairs are invariant under normal routing conditions. Hash collisions in identity embeddings can cause minor variance in GNN distance predictions.
- For router pairs observed in $\ge 3$ training windows, `pair_baseline.csv` records the median, 1st percentile, and 99th percentile of observed distances.
- If an observed flow deviates from this empirical range by $\ge 10\%$ of median and $\ge 40$ miles, it is flagged directly on the pair baseline.
- Unestablished pairs fall back to the GNN model's predicted Gaussian distribution.

### 2. Anomaly-Masked Lag
If an anomalous burst occurs at window $t$, using its value as lag at $t+1$ would:
1. Normalize the anomaly at $t+1$ (causing a false negative).
2. Flag the return to normal traffic at $t+2$ as an anomaly (causing a false positive).

**Solution:** When a flow is flagged at $t$, its lag at $t+1$ is substituted with its **last known normal observation** (from an LRU cache valid up to 1 hour). If no normal baseline exists within 1 hour, lag is zeroed out.

### 3. Hierarchical Scoring Rules

| Scope | Metric | Threshold / Condition |
| :--- | :--- | :--- |
| **Flow (Edge)** | $z = \frac{y - \mu}{\sigma}$ | $\max(|z_{\text{bytes}}|, |z_{\text{miles}}|) \ge \tau_{\text{edge}}$ |
| **Router (Node)** | Mean of top-3 incident flow scores | Flagged if $\ge 2$ anomalous flows and ($\ge 5\%$ of its flows or $\ge 10\%$ of its bytes), OR if it initiates a new-flow burst |
| **Window (Graph)** | Flagged flow fraction | Flagged if fraction $\ge \tau_{\text{win}}$, OR any route flapping detected, OR any new-flow burst |

### 4. Domain Rules (`netflow_prototype/rules.py`)
- **Route Flapping (Repeated Violations):** Identifies flows whose `route_miles` alternates directions (e.g. low $\to$ high $\to$ low) across consecutive windows within a rolling 20-minute horizon.
- **New-Flow Bursts:** Identifies unestablished flows (flows absent from the training set). If $\ge 1,000$ new flows originate from a single ingress router in one window, the ingress router is flagged for a traffic burst / scanning activity.
- **Context Preservation:** New flows are scored by the model, but their messages are excluded from updating router embeddings (`context_flows`), preventing unknown traffic bursts from distorting baseline router representations.

---

## 8. Pipeline Execution Summary

```text
Raw NetFlow Files (.txt.gz / .parquet)
                 │
                 ▼
       schema.aggregate_flows()
                 │
                 ▼
       graph.build_window_graph()  ◄─── Masked Lag & Port Maps
                 │
                 ▼
       model.FlowGNN.forward()     ◄─── Chunked & Checkpointed
                 │
                 ▼
       infer.score_window()        ◄─── Pair Baseline & Z-Scores
                 │
                 ▼
       rules.apply_rules()         ◄─── Flapping & New-Flow Bursts
                 │
                 ▼
    Outputs: edge_anomalies.csv, node_scores.csv,
             window_scores.csv, anomaly_report.html
```
