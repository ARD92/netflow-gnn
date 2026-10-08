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
| Router (node) | mean of its top-3 incident flow scores | >= 2 flagged flows and (>= 5% of its flows or >= 10% of its bytes), or a new-flow burst |
| Window (graph) | fraction of flows flagged | above validation mean + 3 sigma, or any repeated violation or new-flow burst |

**Anomaly-masked lag**: when a flow was flagged in window *t*, its lag in window
*t+1* is its last normal value (up to 1 hour old) instead of the anomalous one.
A persistent anomaly therefore stays flagged, and the return to normal is not
reported as a new anomaly. The file just before the requested window is scored as
an unreported warm-up, so the first window gets the same treatment.

### route_miles: router-pair baseline

route_miles belongs to the path between two routers, so it is not predicted by
the model when a **router-pair baseline** exists. `pair_baseline.csv` (in the
model folder) holds each ingress/egress pair's usual route_miles: the median,
a low-high range (1st percentile of per-window minimums to 99th percentile of
maximums, so multiple normal paths are allowed) and the number of windows seen.
A flow is flagged on route_miles only when it is outside that range by at least
10% of the median and 40 miles (`--miles-min-change`, `--miles-min-change-abs`).
Reasons read `route_miles 896.3 vs usual 311.8 for this router pair (range 310.2-313.4)`.
Pairs seen in fewer than 3 windows fall back to the model; `--no-pair-baseline`
turns the baseline off.

`train` writes the baseline automatically. For an existing model, build it (and
the application baselines below) from files without retraining:

```bash
python -m netflow_prototype baselines -d /data1/netflow-data-for-ai -s "2026-10-04 00:00" -e "2026-10-06 00:00" -m models/2day --workers 8
```

Without it, the model predicted route_miles from hashed router IDs; with about
2,750 routers in 1,024 hash slots, distinct pairs blurred together (a pair
always at 460.6 miles was "expected" at 136). New models use 8,192 router slots
(`--router-buckets`).

### Enrichment, application baselines and patterns

**Applications from ports.** Every flow gets an `application` from its dstPort
(22 `ssh`, 53 `dns`, 443 `https`, 3389 `rdp`, about 90 well-known ports; `any`
for `*`; `other-well-known` / `other-registered` / `other-dynamic` otherwise).
`--port-map ports.csv` (columns `port,application`) adds or overrides names; pass
it to `train`/`baselines` and it is saved with the model so inference matches.

**Customers and services from prefixes.** `infer --enrich-prefixes prefixes.txt`
reads a pipe-delimited file with the header `PREFIX|ASN|ASN_CUSTOMER|SERVICE|IP_MODE`
and matches each flow's source and destination prefix to the most specific file
prefix containing it (IPv4 and IPv6; several SERVICE rows for one prefix are joined
with `;`). Flows get `src_*`/`dst_*` ASN, customer and service, plus a flow-level
`customer` and `service` taken from the side with a known customer (source first).

**Application baselines** (`app_baseline.csv`, `app_pair_baseline.csv`, written by
`train` or the `baselines` command) record each application's usual bytes per
window, overall and by hour of day, and on each router pair. Rules are asymmetric:

| Event | Rule (defaults) |
|---|---|
| `application_disappeared` / `application_drop` | An application normally present (90% of windows) vanishes or falls below 20% of usual (`--app-drop-fraction`). Example: DNS disappearing. |
| `application_surge` | Above 5x usual and its 99th percentile (`--app-surge-factor`), **except** burst-tolerant applications (`--burst-tolerant-apps`, default https, http, http-alt, https-alt), whose bursts are expected. |
| `app_pair_surge` | Any application, https included, on one router pair above 20x its usual bytes and 4 standard deviations (`--app-pair-surge-factor`). |
| `app_pair_disappeared` / `app_pair_drop` | An application normally on a router pair (95% of windows, 5+ flows on average) vanishes or falls below 10%. |

**Patterns.** Anomalous flows and application-pair events are clustered per window
and across the run by router pair, ingress, egress, application, service and
customer. Each dimension's top value is listed with its share of the anomalies, its
share of all traffic and the ratio of the two (lift); it is a pattern when the share
is at least 50% and the lift at least 2 (`--pattern-min-share`, `--pattern-min-lift`),
so "most anomalies are https" is not reported when most traffic is https. Example:
`94% of 18 anomalies leave at egress router SITEA401CR1 (0.6% of flows)`.

On the synthetic benchmark, 39 of the 41 windows with patterns contained an injected
anomaly, and the pattern named the affected egress router or router pair; 26 of 27
application events fell in injected windows.

```bash
python -m netflow_prototype baselines -d /data1/netflow-data-for-ai -s "2026-10-04 00:00" -e "2026-10-06 00:00" -m models/2day --workers 8
python -m netflow_prototype infer -m models/2day -f /data1/netflow-data-for-ai/netflow.20261007.16.00.txt.gz -o results/1007 --enrich-prefixes prefixes.txt --report --device cuda
```

### Rules on top of the model

| Rule | Behavior | Options (`infer`) |
|---|---|---|
| **Repeated violation** | A flow whose route_miles flip low -> high -> low (or high -> low -> high) within the flap window on the same ingress/egress pair. A step counts when it changes route_miles by more than 25% of the lower value and at least 50 miles. Status `repeated_violation`, with the history in `miles_history` (for example `18:40 1363.0 -> 18:50 2862.1 -> 19:00 1353.4`). | `--flap-window 20m` (three 10-minute exports), `--flap-min-change 0.25`, `--flap-min-miles 50`, `--flap-min-reversals 1` |
| **New flow** | A flow with no training baseline is status `new_flow`: reported in `new_flows.csv`, **not** an anomaly. New flows are scored but excluded from router context, so they cannot distort how established flows on the same routers are judged. | |
| **New-flow burst** | At least N flows first seen in one window from the same ingress router: a router-level anomaly (`node_scores.csv`, report). | `--new-flow-burst 1000` |
| **Original vs detected values** | Every anomalous flow lists `baseline_time`, `baseline_bytes`, `baseline_route_miles` (its last normal observation; `baseline_source` is `observed`, or `model` when none exists and the expected value is used), `detected_time` with the current values, and `anomaly_start_time`. | |

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

# Optional: readable report (anomaly_report.txt) and evaluation against labels
python -m netflow_prototype infer ... --report --labels-dir data/labels
```

Times accept `YYYY-MM-DD HH:MM`, ISO 8601, or the file style `YYYYMMDD.HH.MM`.
Durations accept `90m`, `6h`, or `2d`. When it exists, the file immediately before the
window is loaded only to provide lag features. On real servers, point
`--data-dir` at the export directory (for example `/data1/netflow-data-for-ai`).

Useful training options: `--epochs`, `--hidden-dim`, `--num-layers`, `--dropout`,
`--max-flows-per-window` (random sample for very large files), and `--device cuda|mps`.

### Faster runs: parse exports once

Parsing a gzipped export takes several seconds per window (about 7 s for 2
million flows). `prepare` parses each export once, in parallel, into a flow
cache; `train`, `infer` and `pair-baseline` then load windows from it in about
0.2 s each (two 2-million-flow windows: 11.3 s from raw files, 0.5 s from the cache).

```bash
# Once, then e.g. hourly or daily: only new or re-delivered exports are parsed
python -m netflow_prototype prepare -d /data1/netflow-data-for-ai -s "2026-09-22 00:00" -e "2026-10-08 00:00" --cache-dir /data1/netflow-cache --workers 8

# Every command that reads exports accepts the same cache (or set NETFLOW_CACHE_DIR once)
export NETFLOW_CACHE_DIR=/data1/netflow-cache
python -m netflow_prototype train -d /data1/netflow-data-for-ai -s "2026-09-22 00:00" --duration 14d -m models/v2 --device cuda
```

- Commands also read in parallel (`--workers`, default 4) and add any missing
  cache entries as they go, so `prepare` is optional; it just does the work up front.
- An entry is reused only while it is newer than its export, so a re-delivered
  file is parsed again. Damaged exports are reported as `skipped_unreadable`.
- Size: about half of the raw export (about 40 MB per 2-million-flow window, so
  about 80 GB for two weeks). Delete old entries freely; they are rebuilt on demand.
- The cache uses Python pickle: keep the cache directory writable only by trusted users.

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

### Draw the graph as an image

Each 10-minute file is one graph: routers are nodes, and flows are edges from
the ingress router to the egress router. `visualize` draws a window with router
pairs collapsed into one arrow each. Arrow width scales with log(bytes); nodes are
grouped and colored by site (`src_snrc`/`dst_snrc`) and sized by bytes handled.
Pass an inference output directory to highlight pairs with flagged flows (red,
darker means a higher score) and anomalous routers (red outline).

```bash
# Whole network for one window, with anomalies from an inference run
python -m netflow_prototype visualize -f /data1/netflow-data-for-ai/netflow.20261006.11.50.txt.gz -r results/20261006_1150 -o results/20261006_1150/graph.png

# Only the pairs that include one router
python -m netflow_prototype visualize -f /data1/netflow-data-for-ai/netflow.20261006.11.50.txt.gz -r results/20261006_1150 --router <ROUTER> -o router.png
```

- The extension sets the format: `.png`, `.svg` (zoomable; best for large networks), or `.pdf`.
- `--max-pairs` (default 300) draws the heaviest pairs; flagged pairs are always drawn.
- `--labels auto|all|none`: `auto` labels every router up to 80, otherwise the busiest 20 and anomalous ones.
- A table of the drawn pairs (flows, bytes, route_miles, flagged flows, top reason)
  is written next to the image as `<name>.pairs.csv`.
- `--results` must point to an `infer` output that covers the same window.

### Outputs

| File | Contents |
|------|----------|
| `models/<name>/model.pt` | Weights, model config, normalization stats, thresholds, training metadata |
| `models/<name>/seen_flows.npy` | Hashes of flows seen in training (for `is_new_flow`) |
| `models/<name>/train_summary.json` | Loss history, thresholds, file list |
| `results/<name>/edge_anomalies.csv` | Anomalous flows (status `anomaly` or `repeated_violation`): original time and values, detected time and values, expected values, z-scores, anomaly start, reason; `miles_baseline` (`pair` or `model`) with the pair's `usual_miles_low`/`usual_miles_high` |
| `models/<name>/pair_baseline.csv` | Usual route_miles per ingress/egress pair (median, low, high, windows) |
| `results/<name>/new_flows.csv` | Flows with no training baseline at first observation (not anomalies), and whether they belong to a burst |
| `results/<name>/node_scores.csv` | Per-router, per-window scores, including new-flow bursts |
| `results/<name>/window_scores.csv` | Per-window flagged fraction, repeated violations, bursts and verdict |
| `results/<name>/graph_edges.csv` | The scored graph: one row per window, `a_node` (ingress), `z_node` (egress), with flows, bytes, route_miles, anomalous, repeated and new flows |
| `results/<name>/app_anomalies.csv` | Application and application-on-router-pair events: kind, bytes vs usual, reason |
| `results/<name>/anomaly_patterns.csv` | Per window and overall: top value per dimension, share, traffic share, lift, `is_pattern`, statement |
| `models/<name>/app_baseline.csv`, `app_pair_baseline.csv` | Usual bytes per application (overall and by hour) and per application on each router pair |
| `results/<name>/anomaly_report.txt` | Readable report grouped by window (with `--report`; `--report-max-flows` per window, default 25) |
| `results/<name>/edge_scores.csv` | Every scored flow (with `--all-edges`) |
| `results/<name>/metrics.json` | Evaluation (with `--labels-dir`) |

Example `reason` values: `bytes 18.2x above expected`, `bytes 96% below expected`,
`route_miles 1810.7 vs expected 851.2`,
`route_miles flapping 18:50 2862.1 -> 19:00 1353.4 -> 19:10 2853.8 (repeated violation)`.

Example report entry:

```
[ANOMALY] SITEA401CR1 -> SITEB401CR1 | 150.250.155.128/25 -> 217.240.0.0/16 : 443 (web)   score 7.9
    bytes:       539,400,614 at 2026-10-05 18:40  ->  15,050,594,090 at 2026-10-05 19:00  (expected 739,293,440, z=+7.9)
    anomaly started: 2026-10-05 18:50
```

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
- **New flows are not anomalies by design.** Small scans or egress shifts onto new
  flow keys below the burst threshold are only listed in `new_flows.csv`. Lower
  `--new-flow-burst` to catch smaller bursts.
- **Single excursions count as flapping.** With the defaults, a route_miles change
  that lasts one window (low -> high -> low) is a repeated violation. Use
  `--flap-window 30m --flap-min-reversals 2` to require repeated flips.
- **Disappearing flows are not scored.** A flow that is absent in a window has no edge. A future step would add expected-but-missing edge detection per router pair.
- **Drift.** Retrain periodically, for example weekly on a rolling 7-14 day window. At least one full day is needed to learn diurnal shape, and a week is better.
- **Hashed identities.** Prefixes, ports, and routers use hash buckets. Keep `--router-buckets` well above the router count (default 8,192) and raise `--prefix-buckets` (for example 131072) for millions of prefixes; both need retraining.
- **Pair baseline reflects the training window.** A reroute that lasted through much of the training window becomes part of the usual range. Rebuild the baseline from a clean period if that happens.
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
    rules.py       # repeated violations, new flows and bursts, baselines, graph table
    baselines.py   # router-pair route_miles baseline
    cache.py       # parsed-export cache (prepare command)
    enrich.py      # dstPort -> application; prefix -> ASN, customer, service
    appbaseline.py # application byte baselines and application rules
    patterns.py    # cluster anomalies into patterns
    report.py      # readable anomaly_report.txt
    synthetic.py   # synthetic data in the production schema, with labels
    visualize.py   # draw a window's router graph as an image
    cli.py         # generate / files / prepare / train / baselines / infer / visualize
  tests/           # pytest suite (python -m pytest)
  requirements.txt
```
