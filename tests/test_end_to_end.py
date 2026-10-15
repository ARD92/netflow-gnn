"""Generate synthetic data, train briefly, and run inference through the CLI."""

import json

import pandas as pd
from click.testing import CliRunner

from netflow_prototype.cli import main


def test_generate_train_infer(tmp_path):
    data, model, out = tmp_path / "data", tmp_path / "model", tmp_path / "out"
    runner = CliRunner()

    res = runner.invoke(main, ["generate", "-o", str(data), "--start", "2026-10-04 00:00",
                               "--hours", "4", "--anomaly-hours", "1", "--num-flows", "300"])
    assert res.exit_code == 0, res.output
    assert len(list(data.glob("netflow.*.txt.gz"))) == 24

    res = runner.invoke(main, ["train", "-d", str(data), "-s", "2026-10-04 00:00",
                               "--duration", "3h", "-m", str(model), "--epochs", "3",
                               "--cache-dir", str(tmp_path / "cache"), "--workers", "2"])
    assert res.exit_code == 0, res.output
    assert (model / "model.pt").exists()
    assert (model / "pair_baseline.csv").exists()
    summary = json.loads((model / "train_summary.json").read_text())
    assert summary["meta"]["num_windows"] == 18

    res = runner.invoke(main, ["infer", "-m", str(model), "-d", str(data),
                               "-s", "2026-10-04 03:00", "-e", "2026-10-04 04:00",
                               "-o", str(out), "--labels-dir", str(data / "labels"),
                               "--cache-dir", str(tmp_path / "cache")])
    assert res.exit_code == 0, res.output
    windows = pd.read_csv(out / "window_scores.csv")
    assert len(windows) == 6
    assert (out / "edge_anomalies.csv").exists()
    assert (out / "metrics.json").exists()

    # Explicit files (a "new dataset") instead of a time window.
    files = sorted(data.glob("netflow.20261004.03.*.txt.gz"))[:2]
    args = ["infer", "-m", str(model), "-o", str(tmp_path / "out2")]
    for f in files:
        args += ["-f", str(f)]
    res = runner.invoke(main, args)
    assert res.exit_code == 0, res.output
    assert len(pd.read_csv(tmp_path / "out2" / "window_scores.csv")) == 2


def test_parquet_bucket_train_infer(tmp_path):
    import numpy as np
    from datetime import datetime, timedelta

    data, model, out = tmp_path / "parquet_data", tmp_path / "parquet_model", tmp_path / "parquet_out"
    data.mkdir()
    runner = CliRunner()

    base_time = datetime(2026, 10, 7, 10, 30)
    for b_idx in range(4):
        ts = base_time + timedelta(minutes=10 * b_idx)
        df = pd.DataFrame({
            "timestamp": [ts] * 20,
            "src_id": list(range(20)),
            "dst_id": list(range(1, 21)),
            "src_router": [f"R_{i}" for i in range(20)],
            "dst_router": [f"R_{i+1}" for i in range(20)],
            "router_pair": [f"R_{i} -> R_{i+1}" for i in range(20)],
            "bytes": np.random.uniform(1e6, 1e8, size=20),
            "packets": np.random.randint(1000, 50000, size=20),
            "flow_count": [5] * 20,
            "is_anomaly": [False] * 20,
        })
        time_str = ts.strftime("%Y%m%d_%H%M")
        df.to_parquet(data / f"bucket_{b_idx:04d}_{time_str}.parquet")

    res = runner.invoke(main, ["train", "-d", str(data), "-s", "2026-10-07 10:30",
                               "--duration", "40m", "-m", str(model), "--epochs", "2",
                               "--device", "cpu"])
    assert res.exit_code == 0, res.output
    assert (model / "model.pt").exists()
    assert (model / "pair_baseline.csv").exists()
    summary = json.loads((model / "train_summary.json").read_text())
    assert summary["meta"]["num_windows"] == 4

    res = runner.invoke(main, ["infer", "-m", str(model), "-d", str(data),
                               "-s", "2026-10-07 10:30", "--duration", "20m",
                               "-o", str(out), "--device", "cpu"])
    assert res.exit_code == 0, res.output
    windows = pd.read_csv(out / "window_scores.csv")
    assert len(windows) == 2


def test_rolling_retrain(tmp_path):
    import numpy as np
    from datetime import datetime, timedelta

    data = tmp_path / "data"
    model1 = tmp_path / "model1"
    model2 = tmp_path / "model2"
    data.mkdir()
    runner = CliRunner()

    base_time = datetime(2026, 10, 7, 10, 30)
    # Generate 4 initial windows with flows R_0..R_19 -> R_1..R_20
    for b_idx in range(4):
        ts = base_time + timedelta(minutes=10 * b_idx)
        df = pd.DataFrame({
            "timestamp": [ts] * 20,
            "src_id": list(range(20)),
            "dst_id": list(range(1, 21)),
            "src_router": [f"R_{i}" for i in range(20)],
            "dst_router": [f"R_{i+1}" for i in range(20)],
            "router_pair": [f"R_{i} -> R_{i+1}" for i in range(20)],
            "bytes": np.random.uniform(1e6, 1e8, size=20),
            "packets": np.random.randint(1000, 50000, size=20),
            "flow_count": [5] * 20,
            "is_anomaly": [False] * 20,
        })
        time_str = ts.strftime("%Y%m%d_%H%M")
        df.to_parquet(data / f"bucket_{b_idx:04d}_{time_str}.parquet")

    # Initial training
    res = runner.invoke(main, ["train", "-d", str(data), "-s", "2026-10-07 10:30",
                               "--duration", "40m", "-m", str(model1), "--epochs", "2",
                               "--device", "cpu"])
    assert res.exit_code == 0, res.output
    seen1 = np.load(model1 / "seen_flows.npy")
    assert len(seen1) == 20

    # Generate 2 additional windows with brand new flows (R_20..R_29 -> R_21..R_30)
    for b_idx in range(4, 6):
        ts = base_time + timedelta(minutes=10 * b_idx)
        df = pd.DataFrame({
            "timestamp": [ts] * 10,
            "src_id": list(range(20, 30)),
            "dst_id": list(range(21, 31)),
            "src_router": [f"R_{i}" for i in range(20, 30)],
            "dst_router": [f"R_{i+1}" for i in range(20, 30)],
            "router_pair": [f"R_{i} -> R_{i+1}" for i in range(20, 30)],
            "bytes": np.random.uniform(1e6, 1e8, size=10),
            "packets": np.random.randint(1000, 50000, size=10),
            "flow_count": [5] * 10,
            "is_anomaly": [False] * 10,
        })
        time_str = ts.strftime("%Y%m%d_%H%M")
        df.to_parquet(data / f"bucket_{b_idx:04d}_{time_str}.parquet")

    # Retrain on the rolling window (automatically slides to the latest window)
    res = runner.invoke(main, ["retrain", "-d", str(data), "-m", str(model1),
                               "-o", str(model2), "--duration", "20m", "--epochs", "2",
                               "--device", "cpu"])
    assert res.exit_code == 0, res.output
    seen2 = np.load(model2 / "seen_flows.npy")
    # All 20 previous flows + 10 new flows are remembered in flow inventory
    assert len(seen2) == 30
    assert (model2 / "model.pt").exists()
    assert (model2 / "pair_baseline.csv").exists()

    # Router pairs in pair_baseline.csv should be merged
    pair_table = pd.read_csv(model2 / "pair_baseline.csv")
    assert len(pair_table) == 30

    summary = json.loads((model2 / "train_summary.json").read_text())
    assert summary["meta"]["init_model"] == str(model1)


