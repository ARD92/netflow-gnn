"""Router-pair route_miles baseline."""

import json

import numpy as np
import pandas as pd
from click.testing import CliRunner

from netflow_prototype.baselines import (
    PairBaselineBuilder,
    PairMilesConfig,
    apply_pair_baseline,
    pair_summary,
)
from netflow_prototype.cli import main


def _flows(rows):
    return pd.DataFrame(rows, columns=["ingress", "egress", "route_miles"])


def _baseline():
    builder = PairBaselineBuilder()
    for _ in range(5):  # five windows
        builder.add(pair_summary(_flows([
            ("SFF", "LAA", 460.6), ("SFF", "LAA", 460.6),
            ("ATL", "NYC", 800.0), ("ATL", "NYC", 1200.0),  # two normal paths
        ])))
    builder.add(pair_summary(_flows([("DAL", "CHI", 700.0)])))  # seen once
    return builder.build()


def test_builder_summarizes_pairs():
    table = _baseline().set_index(["ingress", "egress"])
    sff = table.loc[("SFF", "LAA")]
    assert sff["median_miles"] == 460.6 and sff["low_miles"] == 460.6
    assert sff["high_miles"] == 460.6 and sff["windows"] == 5
    atl = table.loc[("ATL", "NYC")]
    assert atl["low_miles"] == 800.0 and atl["high_miles"] == 1200.0
    assert table.loc[("DAL", "CHI"), "windows"] == 1


def test_pair_baseline_replaces_model_expectation():
    pairs = _baseline().set_index(["ingress", "egress"])
    df = pd.DataFrame({
        "ingress": ["SFF", "SFF", "SFF", "ATL", "DAL", "NEW"],
        "egress": ["LAA", "LAA", "LAA", "NYC", "CHI", "PAIR"],
        "route_miles": [460.6, 480.0, 1665.0, 1000.0, 2000.0, 999.0],
        # The model wrongly expected ~136 for SFF->LAA (the reported false positive).
        "expected_route_miles": np.float32([136.3, 136.3, 136.3, 900.0, 700.0, 100.0]),
        "z_miles": np.float32([23.2, 24.0, 90.0, 3.0, 50.0, 40.0]),
        "z_bytes": np.float32([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        "score": np.float32([23.2, 24.0, 90.0, 3.0, 50.0, 40.0]),
    })
    out = apply_pair_baseline(df, pairs, PairMilesConfig(), edge_threshold=4.0)
    assert list(out["miles_baseline"]) == ["pair", "pair", "pair", "pair", "model", "model"]
    assert out.loc[0, "expected_route_miles"] == 460.6
    assert out.loc[0, "z_miles"] == 0.0 and out.loc[0, "score"] == 0.5   # usual distance
    assert out.loc[1, "z_miles"] < 4.0                                   # small drift, not flagged
    assert out.loc[2, "z_miles"] >= 4.0                                  # real path change
    assert out.loc[3, "z_miles"] == 0.0                                  # within multi-path range
    assert out.loc[4, "z_miles"] == 50.0 and out.loc[5, "z_miles"] == 40.0  # model fallback


def test_pair_baseline_command(tmp_path):
    data, model = tmp_path / "data", tmp_path / "model"
    model.mkdir()
    runner = CliRunner()
    res = runner.invoke(main, ["generate", "-o", str(data), "--start", "2026-10-04 00:00",
                               "--hours", "1", "--anomaly-hours", "0", "--num-flows", "200"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(main, ["pair-baseline", "-d", str(data), "-s", "2026-10-04 00:00",
                               "--duration", "1h", "-m", str(model), "--workers", "2"])
    assert res.exit_code == 0, res.output
    summary = json.loads(res.output)
    table = pd.read_csv(model / "pair_baseline.csv")
    assert summary["router_pairs"] == len(table) > 0
    assert (table["windows"] == 6).mean() > 0.5
    apps = pd.read_csv(model / "app_baseline.csv")
    assert {"https", "dns"} <= set(apps["application"]) and summary["applications"] > 0
    assert (model / "app_pair_baseline.csv").exists()
