#!/usr/bin/env python3
import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def generate_router_names(num_nodes=800):
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
    parser = argparse.ArgumentParser(description="Generate NetFlow test data that triggers route_miles anomalies.")
    parser.add_argument("--num_buckets", type=int, default=3, help="Number of 10-min files (3 files = 30 min)")
    parser.add_argument("--nodes", type=int, default=800)
    parser.add_argument("--rows_per_file", type=int, default=352000, help="Matches your current 351k flow count")
    parser.add_argument("--output_dir", type=str, default="./route_miles_test_data")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    num_nodes = args.nodes
    k = args.rows_per_file
    router_names = generate_router_names(num_nodes)

    # Core router popularity
    router_weights = 1.0 / np.power(np.arange(1, num_nodes + 1), 0.75)
    router_probs = router_weights / router_weights.sum()

    schema = pa.schema([
        ("timestamp", pa.timestamp("ms")),
        ("src_id", pa.int32()),
        ("dst_id", pa.int32()),
        ("ingress", pa.string()),
        ("egress", pa.string()),
        ("router_pair", pa.string()),
        ("application", pa.string()),
        ("service", pa.string()),
        ("customer", pa.string()),
        ("bytes", pa.float64()),
        ("packets", pa.int64()),
        ("flow_count", pa.int32()),
        ("is_anomaly", pa.bool_()),
        ("anomaly_type", pa.string())
    ])

    start_time = datetime(2026, 10, 8, 10, 20)

    # TARGET ROUTE MILE FOR ANOMALY INJECTION:
    # A major transit pair between BRHAL400ME0 -> ATTGA400ME1
    target_src, target_dst = 0, 1
    target_pair_str = f"{router_names[target_src]} -> {router_names[target_dst]}"

    print("=" * 65)
    print(f"[*] Generating {args.num_buckets} buckets targeting route_miles anomalies")
    print(f"[*] Target Route Mile: {target_pair_str}")
    print("=" * 65)

    for b_idx in range(args.num_buckets):
        current_time = start_time + timedelta(minutes=10 * b_idx)
        time_str = current_time.strftime("%Y%m%d_%H%M")

        # 1. Background traffic WITH INTRA-ROUTER (self-loops) PRESERVED
        # ~10% of flows in production are intra-router (ingress == egress)
        src_ids = np.random.choice(num_nodes, size=k, p=router_probs)
        dst_ids = np.random.choice(num_nodes, size=k, p=router_probs)

        # Baseline traffic generation
        traffic_bytes = np.random.lognormal(mean=18.0, sigma=1.0, size=k)
        packets = (traffic_bytes / np.random.uniform(800, 1300, size=k)).astype(np.int64)
        flows = np.random.poisson(lam=12, size=k).astype(np.int32) + 1
        
        is_anomaly = np.zeros(k, dtype=bool)
        anomaly_types = np.array(["normal"] * k, dtype=object)

        # 2. INJECT PERSISTENT ROUTE_MILE ANOMALY ACROSS CONSECUTIVE WINDOWS
        # Inject 1,200 flows (exceeding window_flagged_fraction 0.002 = 700 flows)
        num_anomaly_flows = 1200
        
        # Mild +45% traffic deviation on this specific route_mile pair
        mild_surge_bytes = np.random.lognormal(mean=18.0, sigma=0.4, size=num_anomaly_flows) * 1.45
        mild_packets = (mild_surge_bytes / 1000).astype(np.int64)

        injected_src = np.array([target_src] * num_anomaly_flows, dtype=np.int32)
        injected_dst = np.array([target_dst] * num_anomaly_flows, dtype=np.int32)
        injected_anomaly = np.ones(num_anomaly_flows, dtype=bool)
        injected_types = np.array(["route_mile_surge_+45%"] * num_anomaly_flows, dtype=object)

        # Append anomaly records
        src_ids = np.append(src_ids, injected_src)
        dst_ids = np.append(dst_ids, injected_dst)
        traffic_bytes = np.append(traffic_bytes, mild_surge_bytes)
        packets = np.append(packets, mild_packets)
        flows = np.append(flows, np.array([25] * num_anomaly_flows, dtype=np.int32))
        is_anomaly = np.append(is_anomaly, injected_anomaly)
        anomaly_types = np.append(anomaly_types, injected_types)

        total_rows = len(src_ids)
        src_names = router_names[src_ids]
        dst_names = router_names[dst_ids]
        router_pairs = [f"{s} -> {d}" for s, d in zip(src_names, dst_names)]

        table = pa.Table.from_arrays([
            pa.array([current_time] * total_rows, type=pa.timestamp("ms")),
            pa.array(src_ids, type=pa.int32()),
            pa.array(dst_ids, type=pa.int32()),
            pa.array(src_names, type=pa.string()),
            pa.array(dst_names, type=pa.string()),
            pa.array(router_pairs, type=pa.string()),
            pa.array(["any"] * total_rows, type=pa.string()),
            pa.array(["ADI;HSIA"] * total_rows, type=pa.string()),
            pa.array(["UNKNOWN"] * total_rows, type=pa.string()),
            pa.array(traffic_bytes, type=pa.float64()),
            pa.array(packets, type=pa.int64()),
            pa.array(flows, type=pa.int32()),
            pa.array(is_anomaly, type=pa.bool_()),
            pa.array(anomaly_types, type=pa.string())
        ], schema=schema)

        filename = f"mild_test_bucket_{b_idx:04d}_{time_str}.parquet"
        filepath = os.path.join(args.output_dir, filename)
        pq.write_table(table, filepath, compression="snappy")
        print(f"  [✔] Generated {filename} with 1,200 flows on route_mile: {target_pair_str}")

if __name__ == "__main__":
    main()
