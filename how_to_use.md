# How to Use NetFlow GNN Prototype

This guide provides end-to-end instructions and exact commands for **training**, **rolling retraining**, **inference**, and **HTML reporting** using the NetFlow Graph Neural Network (`netflow-prototype`).

---

## 1. Environment Setup

Activate the configured Conda environment:

```bash
conda activate gemini_cli
```

Verify your Python environment and PyTorch MPS/CUDA support:

```bash
python -c "import torch; print('PyTorch Version:', torch.__version__, '| MPS available:', torch.backends.mps.is_available(), '| CUDA available:', torch.cuda.is_available())"
```

---

## 2. Model Training

Training runs on time windows of 10-minute snapshot files (both `.parquet` bucket files and `.txt.gz` NetFlow exports).

### A. Mac M-Series GPU (`mps`)
Train on all 144 snapshot windows (24h period) using Apple Silicon hardware acceleration:

```bash
python -m netflow_prototype train \
  --data-dir /private/var/tmp/netflow_mock/data_24h_10min \
  --start "2026-10-07 10:30" \
  --duration 24h \
  --model-dir models/gnn_m5_24h \
  --epochs 30 \
  --device mps \
  --workers 4
```

### B. Linux GPU Server (`cuda`)
On an NVIDIA A100/H100/V100/RTX server:

```bash
python -m netflow_prototype train \
  --data-dir /path/to/netflow_data \
  --start "2026-10-07 10:30" \
  --duration 24h \
  --model-dir models/gnn_cuda_24h \
  --epochs 30 \
  --device cuda \
  --chunk-size 250000 \
  --workers 8
```

### C. Quick Sanity / CPU Smoke Test
To test changes quickly with a bounded sample of flows per window:

```bash
python -m netflow_prototype train \
  --data-dir /private/var/tmp/netflow_mock/data_24h_10min \
  --start "2026-10-07 10:30" \
  --duration 1h \
  --model-dir models/gnn_test \
  --max-flows-per-window 5000 \
  --epochs 2 \
  --device cpu
```

### Generated Artifacts in `--model-dir`
- `model.pt`: FlowGNN weights, architecture configuration, and feature normalization stats.
- `seen_flows.npy`: Hash inventory of normal flows used to detect new-flow bursts.
- `pair_baseline.csv`: Empirical `route_miles` statistical envelope per router pair.
- `app_baseline.csv` & `app_pair_baseline.csv`: Network-wide and router-pair application volume baselines.
- `train_summary.json`: Epoch-by-epoch loss, metrics, and calibrated thresholds.

---

## 3. Rolling Retraining

Rolling retraining updates the model on newly arrived data without forgetting previously established normal flows or path baselines.

### A. Automatic Sliding Window Retraining (Recommended)
Automatically detects the newest files in `--data-dir`, warm-starts from `--model-dir`, unions the flow inventory (`seen_flows.npy`), merges router-pair baselines, and writes the updated model:

```bash
python -m netflow_prototype retrain \
  --data-dir /private/var/tmp/netflow_mock/data_24h_10min \
  --model-dir models/gnn_m5_24h \
  --out-dir models/gnn_m5_updated \
  --duration 24h \
  --epochs 10 \
  --lr 1e-4 \
  --device mps
```
*(Note: If `--out-dir` is omitted, it safely updates `--model-dir` in-place.)*

### B. Retrain Over a Specific Historical Window

```bash
python -m netflow_prototype retrain \
  --data-dir /private/var/tmp/netflow_mock/data_24h_10min \
  --model-dir models/gnn_m5_24h \
  --start "2026-10-08 00:00" \
  --duration 12h \
  --epochs 10 \
  --lr 1e-4 \
  --device mps
```

### C. Using Standard `train` Command with `--init-model`

```bash
python -m netflow_prototype train \
  --data-dir /private/var/tmp/netflow_mock/data_24h_10min \
  --start "2026-10-08 00:00" \
  --duration 12h \
  --init-model models/gnn_m5_24h \
  --model-dir models/gnn_m5_updated \
  --epochs 10 \
  --lr 1e-4 \
  --device mps
```

---

## 4. Inference & Anomaly Detection

### A. Time Window Mode
Run inference over a continuous time range of snapshot files:

```bash
python -m netflow_prototype infer \
  -m models/gnn_m5_24h \
  -d /private/var/tmp/netflow_mock/test_anomalies_data \
  -s "2026-10-08 10:20" \
  --duration 1h \
  -o results/test_anomalies_eval \
  --device mps \
  --report
```

### B. Explicit Files Mode (`-f`)
Evaluate specific files directly (e.g. ad-hoc anomaly investigation):

```bash
python -m netflow_prototype infer \
  -m models/gnn_m5_24h \
  -f /private/var/tmp/netflow_mock/test_anomalies_data/test_bucket_0001_20261008_1030.parquet \
  -f /private/var/tmp/netflow_mock/test_anomalies_data/test_bucket_0003_20261008_1050.parquet \
  -o results/test_selected \
  --device mps \
  --report
```

### Output Files Generated in `-o / --out-dir`
- **`anomaly_report.txt`**: Human-readable incident summary (anomalous windows, routers, top flagged flows).
- **`window_scores.csv`**: Summary scores per window (flow count, flagged flow count, max score, anomalous flag).
- **`node_scores.csv`**: Router-level scores (incident flagged flows, flagged bytes fraction, new-flow bursts).
- **`edge_anomalies.csv`**: Granular flow-level records (z-score, expected vs actual bytes, reason, customer, service).
- **`anomaly_patterns.csv`**: Synthesized incident patterns detected across routers.

---

## 5. Interactive HTML Report & Model Weight Inspector

Generate a self-contained, interactive HTML dashboard combining the inference results and an internal GNN weight inspector:

```bash
python -m netflow_prototype report \
  -m models/gnn_m5_24h \
  -r results/test_anomalies_eval \
  -o results/final_inference_report.html
```

To view the generated report on macOS:

```bash
open results/final_inference_report.html
```

### What the Report Includes:
1. **Anomaly Analytics Dashboard**:
   - **Strongest Pattern Banner**: Highlights dominant multi-router incidents (e.g. DDoS floods, reconnaissance scans).
   - **Where Anomalies Flow (Chord Diagram)**: Interactive circular ribbon graphic showing flow paths between cities and routers.
   - **Top Applications in Anomalies**: Horizontal bar chart comparing tracked applications (`https`, `dns`, `gtp-u`, etc.).
   - **Anomalies Over Time**: Stacked bar chart with rolling average trendline.
   - **Route-Miles Violations**: Scatter plot with $\pm 10\%$ ($\ge 40$ mi) tolerance envelope.
   - **Largest Route-Miles Jumps**: Dumbbell chart visualizing path reroute distances.
2. **Internal GNN Model & Weight Inspector**:
   - **FlowGNN Pipeline Diagram**: Visual message-passing architecture flow.
   - **55-Tensor Parameter Catalog**: Inspect shapes, norms, parameter counts, and statistics for all layers.
   - **Interactive Weight Matrix Heatmap**: Click any tensor to see a color-coded numerical 2D heatmap with cell-level inspection.
   - **Weight Distribution Histogram**: Visualizes weight variance and distribution around zero.
3. **Anomalies Explorer**:
   - Filterable, searchable table of all flagged edge anomalies with sorting and customer/service tags.

---

## 6. Utilities & Dry-Run Tools

### Check Window Files Resolution (`files`)
Verify which files match a time range before training or inferring:

```bash
python -m netflow_prototype files \
  -d /private/var/tmp/netflow_mock/data_24h_10min \
  -s "2026-10-07 10:30" \
  --duration 24h
```

### Recalculate Baselines Without Retraining (`baselines`)
Rebuild `pair_baseline.csv` and `app_baseline.csv` from historical data without modifying GNN neural weights:

```bash
python -m netflow_prototype baselines \
  -d /private/var/tmp/netflow_mock/data_24h_10min \
  -s "2026-10-07 10:30" \
  --duration 24h \
  -m models/gnn_m5_24h \
  --workers 4
```

### Visualize Graph as an Image (`visualize`)
Render an image (`.png`, `.svg`, or `.pdf`) of a single window's router graph with flagged anomalies highlighted:

```bash
python -m netflow_prototype visualize \
  /private/var/tmp/netflow_mock/test_anomalies_data/test_bucket_0001_20261008_1030.parquet \
  -r results/test_anomalies_eval \
  -o results/graph_window.png
```
