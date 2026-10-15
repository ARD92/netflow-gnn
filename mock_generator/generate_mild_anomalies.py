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
    parser = argparse.ArgumentParser(description="Generate realistic, mild NetFlow anomalies.")
    parser.add_argument("--num_buckets", type=int, default=6, help="Number of 10-min files (6 = 1 hour window)")
    parser.add_argument("--nodes", type=int, default=2000)
    parser.add_argument("--rows_per_file", type=int, default=600000, help="~20 MB parquet per file")
    parser.add_argument("--output_dir", type=str, default="./mild_anomalies_data")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    num_nodes = args.nodes
    k = args.rows_per_file
    router_names = generate_router_names(num_nodes)

    # Core/Middle-mile router popularity distribution
    router_weights = 1.0 / np.power(np.arange(1, num_nodes + 1), 0.75)
    router_probs = router_weights / router_weights.sum()

    applications = np.array(["any", "HTTPS", "DNS", "SSH", "BGP", "SNMP", "SIP"])
    app_probs = np.array([0.70, 0.15, 0.05, 0.04, 0.03, 0.02, 0.01])
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
        ("anomaly_type", pa.string())
    ])

    start_time = datetime(2026, 10, 8, 10, 20)
    print("=" * 65)
    print(f"[*] Generating {args.num_buckets} buckets with MILD anomalies (25% - 75% deviations)")
    print(f"[*] Output directory: {args.output_dir}")
    print("=" * 65)

    for b_idx in range(args.num_buckets):
        current_time = start_time + timedelta(minutes=10 * b_idx)
        time_str = current_time.strftime("%Y%m%d_%H%M")
        hour = current_time.hour
        diurnal = 1.0 + 0.35 * np.sin(2 * np.pi * (hour - 8) / 24)

        # 1. Background Traffic (strictly no self-loops)
        src_ids = np.random.choice(num_nodes, size=k, p=router_probs)
        dst_ids = np.random.choice(num_nodes, size=k, p=router_probs)
        self_loops = src_ids == dst_ids
        dst_ids[self_loops] = (dst_ids[self_loops] + 1) % num_nodes

        traffic_bytes = np.random.lognormal(mean=18.0, sigma=1.2, size=k) * diurnal
        packets = (traffic_bytes / np.random.uniform(700, 1300, size=k)).astype(np.int64)
        flows = np.random.poisson(lam=12, size=k).astype(np.int32) + 1
        
        is_anomaly = np.zeros(k, dtype=bool)
        anomaly_types = np.array(["normal"] * k, dtype=object)

        # -------------------------------------------------------------
        # INJECT REALISTIC, MILD / MIDDLE-MILE ANOMALIES
        # -------------------------------------------------------------
        injected_src = []
        injected_dst = []
        injected_bytes = []
        injected_packets = []
        injected_flows = []
        injected_types = []

        # PATTERN 1: Middle-Mile Route Shift / BGP Flap (Bucket 1 & 2)
        # Primary path experiences a mild -35% drop; Secondary path absorbs a +45% surge
        if b_idx in [1, 2]:
            num_rerouted = 120
            # Primary transit link: BRHAL400ME0 -> ATTGA400ME1
            p_src, p_dst = 0, 1
            # Secondary backup link: BRHAL400ME0 -> CGCIL400ME2
            s_src, s_dst = 0, 2

            # Mild drop on primary path (-35%)
            drop_bytes = np.random.lognormal(mean=18.0, sigma=0.5, size=num_rerouted) * 0.65
            injected_src.extend([p_src] * num_rerouted)
            injected_dst.extend([p_dst] * num_rerouted)
            injected_bytes.extend(drop_bytes)
            injected_packets.extend((drop_bytes / 1000).astype(np.int64))
            injected_flows.extend([8] * num_rerouted)
            injected_types.extend(["mild_route_drop_-35%"] * num_rerouted)

            # Mild surge on secondary path (+45%)
            surge_bytes = np.random.lognormal(mean=18.0, sigma=0.5, size=num_rerouted) * 1.45
            injected_src.extend([s_src] * num_rerouted)
            injected_dst.extend([s_dst] * num_rerouted)
            injected_bytes.extend(surge_bytes)
            injected_packets.extend((surge_bytes / 1000).astype(np.int64))
            injected_flows.extend([16] * num_rerouted)
            injected_types.extend(["mild_reroute_surge_+45%"] * num_rerouted)

            print(f"  [!] Bucket {b_idx}: Injected Middle-Mile Route Shift (BRHAL400ME0: Primary -35%, Backup +45%)")

        # PATTERN 2: Mild Video/Streaming Flash Congestion (Bucket 3)
        # Moderate +60% traffic increase across 80 pairs on service ADI;HSIA
        if b_idx == 3:
            num_congested = 80
            c_src = np.random.choice(num_nodes // 4, size=num_congested) # Aggregation nodes
            c_dst = np.random.choice(num_nodes // 4, size=num_congested)
            c_dst[c_src == c_dst] = (c_dst[c_src == c_dst] + 1) % num_nodes

            cong_bytes = np.random.lognormal(mean=18.0, sigma=0.6, size=num_congested) * 1.60  # +60%
            injected_src.extend(c_src)
            injected_dst.extend(c_dst)
            injected_bytes.extend(cong_bytes)
            injected_packets.extend((cong_bytes / 1200).astype(np.int64))
            injected_flows.extend([20] * num_congested)
            injected_types.extend(["mild_congestion_surge_+60%"] * num_congested)

            print(f"  [!] Bucket {b_idx}: Injected Flash Congestion (+60% above baseline on 80 pairs)")

        # PATTERN 3: Stealthy Low-and-Slow Exfiltration / Elevated Flow (Bucket 4)
        # Bytes only increase by +30%, but flow count doubles (elevated sessions)
        if b_idx == 4:
            num_stealth = 40
            s_src = np.random.choice(range(50, 150), size=num_stealth)
            s_dst = np.random.choice(range(500, 600), size=num_stealth)

            stealth_bytes = np.random.lognormal(mean=18.0, sigma=0.5, size=num_stealth) * 1.30  # +30%
            injected_src.extend(s_src)
            injected_dst.extend(s_dst)
            injected_bytes.extend(stealth_bytes)
            injected_packets.extend((stealth_bytes / 800).astype(np.int64))
            injected_flows.extend([30] * num_stealth) # High flow sessions
            injected_types.extend(["stealth_flow_surge_+30%"] * num_stealth)

            print(f"  [!] Bucket {b_idx}: Injected Stealthy Activity (+30% bytes, 2x flow sessions)")

        # Append injected records
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

        filename = f"mild_test_bucket_{b_idx:04d}_{time_str}.parquet"
        filepath = os.path.join(args.output_dir, filename)
        pq.write_table(table, filepath, compression="snappy")
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [✔] Wrote {filename} ({file_size_mb:.2f} MB, {is_anomaly.sum()} mild anomalies)")

    print(f"\n[✔] Generation complete. Files saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
