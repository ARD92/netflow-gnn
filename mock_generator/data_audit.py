#!/usr/bin/env python3
import glob
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def run_data_quality_audit(snapshot_dir, num_expected_nodes=2000, sample_size=10):
    files = sorted(glob.glob(os.path.join(snapshot_dir, "*.parquet")))
    if not files:
        print(f"[ERROR] No parquet files found in {snapshot_dir}")
        return

    print(f"==================================================")
    print(f"      GNN NETFLOW DATA QUALITY AUDIT REPORT       ")
    print(f"==================================================")
    print(f"Found {len(files)} snapshot files.")
    
    # 1. TEMPORAL CONTINUITY CHECK
    print("\n[1/5] Checking Temporal Bucket Continuity...")
    timestamps = []
    for f in files:
        meta = pq.read_metadata(f)
        # Read just timestamp column from first row
        df_t = pq.read_table(f, columns=["timestamp"]).slice(0, 1).to_pandas()
        timestamps.append(df_t["timestamp"].iloc[0])
    
    time_series = pd.Series(timestamps).sort_values()
    diffs = time_series.diff().dropna()
    expected_diff = pd.Timedelta(minutes=10)
    gaps = diffs[diffs != expected_diff]
    
    if len(gaps) == 0:
        print("  ✔ PASS: Continuous 10-minute intervals without missing snapshots.")
    else:
        print(f"  ❌ FAIL: Found {len(gaps)} missing/irregular time gaps!")

    # Sample snapshots for deep structural and numerical checks
    sample_indices = np.linspace(0, len(files) - 1, min(sample_size, len(files)), dtype=int)
    sampled_files = [files[i] for i in sample_indices]

    node_map_reference = {}
    null_count = 0
    physics_violations = 0
    degrees_list = []
    anomaly_counts = 0
    total_records = 0
    byte_ranges = []

    print(f"\n[2/5] Running Structural & Numerical Checks on {len(sampled_files)} sampled snapshots...")

    for idx, f in enumerate(sampled_files):
        df = pd.read_parquet(f)
        total_records += len(df)

        # Check Nulls
        null_count += df.isnull().sum().sum()

        # Check Physical Constraints
        # - Packet size: bytes / packets must be between 40 bytes and 9000 bytes (typical MTU)
        avg_pkt_size = df["bytes"] / np.maximum(df["packets"], 1)
        invalid_packets = (df["packets"] <= 0) | (df["bytes"] <= 0)
        invalid_size = (avg_pkt_size < 40) | (avg_pkt_size > 9200)
        physics_violations += (invalid_packets | invalid_size).sum()

        # Check Node ID Consistency
        router_col = "src_router" if "src_router" in df.columns else ("ingress" if "ingress" in df.columns else None)
        if router_col is not None:
            sub_map = dict(zip(df["src_id"], df[router_col]))
            if idx == 0:
                node_map_reference = sub_map
            else:
                for n_id, name in sub_map.items():
                    if n_id in node_map_reference and node_map_reference[n_id] != name:
                        print(f"  ❌ CRITICAL: Node ID Drift! ID {n_id} mapped to '{name}' vs '{node_map_reference[n_id]}'")
                        break

        # Check Node Degrees (Graph Sparsity)
        unique_nodes_in_bucket = len(set(df["src_id"]).union(set(df["dst_id"])))
        avg_out_degree = len(df) / num_expected_nodes
        degrees_list.append((unique_nodes_in_bucket, avg_out_degree))

        # Check Feature Dynamic Range
        byte_ranges.append((df["bytes"].min(), df["bytes"].quantile(0.5), df["bytes"].max()))

        # Check Labels
        if "is_anomaly" in df.columns:
            anomaly_counts += df["is_anomaly"].sum()

    # Evaluation Output
    print("\n--- Integrity Results ---")
    if null_count == 0:
        print("  ✔ Null / NaN count: 0")
    else:
        print(f"  ❌ Found {null_count} NaNs! GNN backprop will fail.")

    if physics_violations == 0:
        print("  ✔ Physical sanity: All packet/byte ratios conform to MTU constraints.")
    else:
        print(f"  ⚠ Warning: {physics_violations} rows violated network MTU/packet physics.")

    print("\n[3/5] Graph Density & Topology Stability:")
    avg_active_nodes = np.mean([d[0] for d in degrees_list])
    avg_degree = np.mean([d[1] for d in degrees_list])
    print(f"  - Active nodes per bucket: {avg_active_nodes:.1f} / {num_expected_nodes} ({(avg_active_nodes/num_expected_nodes)*100:.1f}%)")
    print(f"  - Average Node Degree: {avg_degree:.2f} edges/node")
    if avg_active_nodes < 0.5 * num_expected_nodes:
        print("  ⚠ Low Graph Density: Over 50% of nodes are isolated per snapshot. Consider temporal aggregation or adding virtual edge mechanisms.")

    print("\n[4/5] Dynamic Range (Raw Traffic Features):")
    min_b = np.min([b[0] for b in byte_ranges])
    med_b = np.median([b[1] for b in byte_ranges])
    max_b = np.max([b[2] for b in byte_ranges])
    print(f"  - Min Bytes: {min_b:,.0f} | Median: {med_b:,.0f} | Max: {max_b:,.0f}")
    if (max_b / (med_b + 1e-5)) > 1e4:
        print("  ⚠ Extreme Feature Skew: Log-transform `log1p(bytes)` is mandatory before feeding to GNN.")

    print("\n[5/5] Target Label Quality & Imbalance:")
    if anomaly_counts > 0:
        imbalance = anomaly_counts / total_records
        print(f"  - Total Anomalies in Sample: {anomaly_counts:,} / {total_records:,} ({imbalance * 100:.3f}%)")
        if imbalance < 0.001:
            print("  ⚠ Severe Class Imbalance (<0.1%): Use Focal Loss or Weighted Cross-Entropy.")
        elif imbalance > 0.15:
            print("  ⚠ Unrealistic Anomaly Rate (>15%): Anomaly detection models may overfit.")
        else:
            print("  ✔ Class balance is within healthy anomaly detection range (0.1% - 10%).")
    else:
        print("  ⚠ No anomalies found or target column missing.")

    print("\n==================================================")
    print("                AUDIT COMPLETE                    ")
    print("==================================================")

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "./data_24h_10min"
    run_data_quality_audit(target, num_expected_nodes=2000)