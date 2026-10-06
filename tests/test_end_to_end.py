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
                               "--duration", "3h", "-m", str(model), "--epochs", "3"])
    assert res.exit_code == 0, res.output
    assert (model / "model.pt").exists()
    summary = json.loads((model / "train_summary.json").read_text())
    assert summary["meta"]["num_windows"] == 18

    res = runner.invoke(main, ["infer", "-m", str(model), "-d", str(data),
                               "-s", "2026-10-04 03:00", "-e", "2026-10-04 04:00",
                               "-o", str(out), "--labels-dir", str(data / "labels")])
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
