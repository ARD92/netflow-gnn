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
    parser = argparse.ArgumentParser(description="Generate test NetFlow files with structured GNN anomalies.")
    parser.add_argument("--num_buckets", type=int, default=6, help="Number of 10-min files to generate (e.g. 6 = 1 hour test window).")
    parser.add_argument("--nodes", type=int, default=2000)
    parser.add_argument("--rows_per_file", type=int, default=600000, help="~20 MB per file")
    parser.add_argument("--output_dir", type=str, default="./test_anomalies_data")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    num_nodes = args.nodes
    k = args.rows_per_file
    router_names = generate_router_names(num_nodes)
    
    # Power-law background router traffic
    router_weights = 1.0 / np.power(np.arange(1, num_nodes + 1), 0.75)
    router_probs = router_weights / router_weights.sum()

    applications = np.array(["any", "HTTPS", "DNS", "SSH", "BGP", "SNMP", "SIP"])
    app_probs = np.array([0.75, 0.12, 0.05, 0.03, 0.02, 0.02, 0.01])
    services = np.array(["ADI;HSIA", "BROADBAND", "VPN", "DIRECT_CONNECT", "UNKNOWN"])
    svc_probs = np.array([0.55, 0.20, 0.15, 0.07, 0.03])
    customers = np.array(["UNKNOWN", "ENTERPRISE_A", "ENTERPRISE_B", "RETAIL_1", "CARRIER_X"])
    cust_probs = np.array([0.45, 0.25, 0.15, 0.10, 0.05])

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
        ("anomaly_type", pa.string())  # Ground truth label for evaluation
    ])

    start_time = datetime(2026, 10, 8, 10, 20)
    print("=" * 60)
    print(f"[*] Generating {args.num_buckets} test buckets in '{args.output_dir}'")
    print(f"[*] Total Nodes: {num_nodes:,} | Baseline Rows/File: {k:,}")
    print("=" * 60)

    for b_idx in range(args.num_buckets):
        current_time = start_time + timedelta(minutes=10 * b_idx)
        time_str = current_time.strftime("%Y%m%d_%H%M")
        
        # 1. Normal Background Traffic
        src_ids = np.random.choice(num_nodes, size=k, p=router_probs)
        dst_ids = np.random.choice(num_nodes, size=k, p=router_probs)
        self_loop = src_ids == dst_ids
        dst_ids[self_loop] = (dst_ids[self_loop] + 1) % num_nodes

        traffic_bytes = np.random.lognormal(mean=18.0, sigma=1.5, size=k)
        packets = (traffic_bytes / np.random.uniform(600, 1400, size=k)).astype(np.int64)
        flows = np.random.poisson(lam=12, size=k).astype(np.int32) + 1
        
        is_anomaly = np.zeros(k, dtype=bool)
        anomaly_types = np.array(["normal"] * k, dtype=object)

        # -------------------------------------------------------------
        # INJECT ANOMALIES (Only in specific buckets to test temporal changes)
        # -------------------------------------------------------------
        injected_src = []
        injected_dst = []
        injected_bytes = []
        injected_packets = []
        injected_flows = []
        injected_types = []

        # ANOMALY 1: DDoS Attack (Many-to-One) in Bucket 1 & 2
        if b_idx in [1, 2]:
            target_node = 42
            num_attackers = 500
            attacker_nodes = np.random.choice(num_nodes, size=num_attackers, replace=False)
            attacker_nodes = attacker_nodes[attacker_nodes != target_node]

            injected_src.extend(attacker_nodes)
            injected_dst.extend([target_node] * len(attacker_nodes))
            # Flood: tiny packets, high packet count, moderate bytes
            injected_bytes.extend(np.random.uniform(5e7, 2e8, size=len(attacker_nodes)))
            injected_packets.extend(np.random.uniform(50000, 200000, size=len(attacker_nodes)).astype(np.int64))
            injected_flows.extend([100] * len(attacker_nodes))
            injected_types.extend(["ddos_flood"] * len(attacker_nodes))
            print(f"  [!] Bucket {b_idx}: Injected DDoS Flood -> Target: Node {target_node} ({router_names[target_node]}) from {len(attacker_nodes)} sources")

        # ANOMALY 2: Recon / Port-Scan Fan-Out (One-to-Many) in Bucket 3
        if b_idx == 3:
            scanner_node = 150
            num_targets = 400
            targets = np.random.choice(num_nodes, size=num_targets, replace=False)
            targets = targets[targets != scanner_node]

            injected_src.extend([scanner_node] * len(targets))
            injected_dst.extend(targets)
            # Recon: minimal bytes, very few packets per target (tests graph structure over scalars)
            injected_bytes.extend(np.random.uniform(200, 800, size=len(targets)))
            injected_packets.extend(np.random.randint(1, 4, size=len(targets)))
            injected_flows.extend([1] * len(targets))
            injected_types.extend(["recon_scan"] * len(targets))
            print(f"  [!] Bucket {b_idx}: Injected Fan-Out Scan -> Source: Node {scanner_node} ({router_names[scanner_node]}) scanning {len(targets)} nodes")

        # ANOMALY 3: Data Exfiltration (Volumetric Burst on single pair) in Bucket 4
        if b_idx == 4:
            src_victim = 88
            dst_dest = 1900
            injected_src.extend([src_victim])
            injected_dst.extend([dst_dest])
            # Exfiltration: 100x normal volume burst
            injected_bytes.extend([5e11])  # 500 GB in 10 mins
            injected_packets.extend([400_000_000])
            injected_flows.extend([5000])
            injected_types.extend(["exfiltration_burst"])
            print(f"  [!] Bucket {b_idx}: Injected Exfiltration Burst -> {router_names[src_victim]} -> {router_names[dst_dest]}")

        # Combine background and injected records
        if injected_src:
            m = len(injected_src)
            src_ids = np.append(src_ids, injected_src)
            dst_ids = np.append(dst_ids, injected_dst)
            traffic_bytes = np.append(traffic_bytes, injected_bytes)
            packets = np.append(packets, injected_packets)
            flows = np.append(flows, injected_flows)
            is_anomaly = np.append(is_anomaly, [True] * m)
            anomaly_types = np.append(anomaly_types, injected_types)

        total_rows = len(src_ids)
        src_names = router_names[src_ids]
        dst_names = router_names[dst_ids]
        router_pairs = [f"{s} -> {d}" for s, d in zip(src_names, dst_names)]
        apps = np.random.choice(applications, size=total_rows, p=app_probs)
        svcs = np.random.choice(services, size=total_rows, p=svc_probs)
        custs = np.random.choice(customers, size=total_rows, p=cust_probs)

        # Write Parquet
        table = pa.Table.from_arrays([
            pa.array([current_time] * total_rows, type=pa.timestamp("ms")),
            pa.array(src_ids, type=pa.int32()),
            pa.array(dst_ids, type=pa.int32()),
            pa.array(src_names, type=pa.string()),
            pa.array(dst_names, type=pa.string()),
            pa.array(router_pairs, type=pa.string()),
            pa.array(apps, type=pa.string()),
            pa.array(svcs, type=pa.string()),
            pa.array(custs, type=pa.string()),
            pa.array(traffic_bytes, type=pa.float64()),
            pa.array(packets, type=pa.int64()),
            pa.array(flows, type=pa.int32()),
            pa.array(is_anomaly, type=pa.bool_()),
            pa.array(anomaly_types, type=pa.string())
        ], schema=schema)

        filename = f"test_bucket_{b_idx:04d}_{time_str}.parquet"
        filepath = os.path.join(args.output_dir, filename)
        pq.write_table(table, filepath, compression="snappy")
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [✔] Wrote {filename} ({file_size_mb:.2f} MB, {is_anomaly.sum()} anomalies)")

if __name__ == "__main__":
    main()
