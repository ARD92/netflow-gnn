#!/usr/bin/env python3
import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def generate_router_names(num_nodes=2000):
    prefixes = ["BRHAL", "ATTGA", "CGCIL", "DLLTX", "NYCNY", "SNTCA", "STLAM", "CHCGX"]
    tiers = ["ME", "IGX", "CR", "PE"]
    nodes = []
    for i in range(num_nodes):
        pref = prefixes[i % len(prefixes)]
        num = 400 + (i // len(prefixes)) % 50
        tier = tiers[(i // 200) % len(tiers)]
        sub = i % 10
        nodes.append(f"{pref}{num}{tier}{sub}")
    return np.array(nodes)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate per-10min bucket Parquet files.")
    parser.add_argument("--mode", choices=["24h", "7d"], default="24h",
                        help="Choose '24h' (144 files) or '7d' (1008 files).")
    parser.add_argument("--num_buckets", type=int, default=None,
                        help="Explicit number of 10-min files to generate (overrides --mode).")
    parser.add_argument("--nodes", type=int, default=2000, help="Number of nodes (default: 2000).")
    parser.add_argument("--active_ratio", type=float, default=0.015,
                        help="Ratio of active pairs per bucket (default: 0.015 -> ~60k edges/file).")
    parser.add_argument("--inject_anomalies", action="store_true", help="Inject anomalous traffic bursts.")
    parser.add_argument("--output_dir", type=str, default="./snapshots_10min",
                        help="Target directory to store bucket files.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    num_nodes = args.nodes
    if args.num_buckets is not None:
        total_buckets = args.num_buckets
    else:
        total_buckets = 144 if args.mode == "24h" else 1008
    edges_per_bucket = int(num_nodes * (num_nodes - 1) * args.active_ratio)

    print(f"[*] Configuration:")
    print(f"    - Nodes: {num_nodes:,}")
    print(f"    - Mode: {args.mode} ({total_buckets} files, 1 per 10 minutes)")
    print(f"    - Target Directory: {args.output_dir}")
    print(f"    - Edges per file: ~{edges_per_bucket:,}")

    # Generate Node Hostnames & Zipf distribution weights
    router_names = generate_router_names(num_nodes)
    router_weights = 1.0 / np.power(np.arange(1, num_nodes + 1), 0.75)
    router_probs = router_weights / router_weights.sum()

    # Time setup
    end_time = datetime(2026, 10, 8, 10, 20)
    start_time = end_time - timedelta(minutes=10 * total_buckets)

    # Pre-defined PyArrow schema for consistency
    schema = pa.schema([
        ("timestamp", pa.timestamp("ms")),
        ("src_id", pa.int32()),
        ("dst_id", pa.int32()),
        ("src_router", pa.string()),
        ("dst_router", pa.string()),
        ("router_pair", pa.string()),
        ("bytes", pa.float64()),
        ("packets", pa.int64()),
        ("flow_count", pa.int32()),
        ("is_anomaly", pa.bool_())
    ])

    for b_idx in range(total_buckets):
        current_time = start_time + timedelta(minutes=10 * (b_idx + 1))
        hour = current_time.hour
        time_str = current_time.strftime("%Y%m%d_%H%M")
        
        # 1. Diurnal traffic factor
        diurnal_multiplier = 1.0 + 0.45 * np.sin(2 * np.pi * (hour - 8) / 24)

        # 2. Sample edges without self-loops
        src_ids = np.random.choice(num_nodes, size=edges_per_bucket, p=router_probs)
        dst_ids = np.random.choice(num_nodes, size=edges_per_bucket, p=router_probs)
        valid = src_ids != dst_ids
        src_ids, dst_ids = src_ids[valid], dst_ids[valid]
        k = len(src_ids)

        # 3. Features
        traffic_bytes = np.random.lognormal(mean=18.0, sigma=1.5, size=k) * diurnal_multiplier
        packets = (traffic_bytes / np.random.uniform(600, 1400, size=k)).astype(np.int64)
        flows = np.random.poisson(lam=15, size=k).astype(np.int32) + 1
        is_anomaly = np.zeros(k, dtype=bool)

        # 4. Optional Anomaly Injection
        if args.inject_anomalies and (b_idx % 36 == 0):
            num_anom = np.random.randint(5, 15)
            idx_anom = np.random.choice(k, size=num_anom, replace=False)
            traffic_bytes[idx_anom] *= np.random.uniform(10.0, 50.0, size=num_anom)
            packets[idx_anom] *= np.random.uniform(8.0, 40.0, size=num_anom).astype(np.int64)
            is_anomaly[idx_anom] = True

        src_names = router_names[src_ids]
        dst_names = router_names[dst_ids]
        router_pairs = [f"{s} -> {d}" for s, d in zip(src_names, dst_names)]

        # 5. Write individual file
        table = pa.Table.from_arrays([
            pa.array([current_time] * k, type=pa.timestamp("ms")),
            pa.array(src_ids, type=pa.int32()),
            pa.array(dst_ids, type=pa.int32()),
            pa.array(src_names, type=pa.string()),
            pa.array(dst_names, type=pa.string()),
            pa.array(router_pairs, type=pa.string()),
            pa.array(traffic_bytes, type=pa.float64()),
            pa.array(packets, type=pa.int64()),
            pa.array(flows, type=pa.int32()),
            pa.array(is_anomaly, type=pa.bool_())
        ], schema=schema)

        # Standard zero-padded filename: bucket_0001_20261007_1030.parquet
        filename = f"bucket_{b_idx:04d}_{time_str}.parquet"
        filepath = os.path.join(args.output_dir, filename)
        pq.write_table(table, filepath, compression="snappy")

        if (b_idx + 1) % 36 == 0 or (b_idx + 1) == total_buckets:
            print(f"    [Progress] Generated {b_idx + 1}/{total_buckets} -> {filename}")

    print(f"\n[✔] Completed. {total_buckets} files saved in: {args.output_dir}")

if __name__ == "__main__":
    main()