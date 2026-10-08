"""Flow cache and parallel reading."""

import json
import os
import time
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from netflow_prototype import cache
from netflow_prototype.cli import main
from netflow_prototype.data import iter_flows
from netflow_prototype.schema import flow_hash, load_window
from tests.conftest import write_file


def test_cache_round_trip_and_reuse(sample_file, tmp_path, monkeypatch):
    cdir = tmp_path / "cache"
    first = cache.load_window_cached(sample_file, cdir)
    assert cache.cache_path(cdir, sample_file).exists()
    assert cache.cache_path(cdir, sample_file).name == "netflow.20261006.11.50.flows.v1.pkl.gz"

    def no_parse(path):
        raise AssertionError("should have used the cache")

    monkeypatch.setattr(cache, "load_window", no_parse)
    second = cache.load_window_cached(sample_file, cdir)
    assert (flow_hash(first) == flow_hash(second)).all()
    assert second["bytes"].equals(first["bytes"])
    assert str(second["ingress"].dtype) == str(load_window(sample_file)["ingress"].dtype)


def test_newer_source_is_parsed_again(sample_file, tmp_path, monkeypatch):
    cdir = tmp_path / "cache"
    cache.load_window_cached(sample_file, cdir)
    future = time.time() + 60
    os.utime(sample_file, (future, future))  # re-delivered export
    assert not cache.is_fresh(cdir, sample_file)
    calls = []
    real = cache.load_window
    monkeypatch.setattr(cache, "load_window", lambda p: calls.append(p) or real(p))
    cache.load_window_cached(sample_file, cdir)
    assert calls and cache.is_fresh(cdir, sample_file)


def test_parallel_reading_keeps_order_and_skips_bad_files(tmp_path):
    good = [write_file(tmp_path / f"netflow.20261006.11.{m:02d}.txt.gz") for m in (0, 10, 30)]
    bad = tmp_path / "netflow.20261006.11.20.txt.gz"
    bad.write_bytes(good[0].read_bytes()[:-20])  # truncated gzip
    paths = [good[0], good[1], bad, good[2]]
    out = list(iter_flows(paths, cache_dir=tmp_path / "cache", workers=2))
    assert [p for p, _, _ in out] == paths
    assert [f is None for _, f, _ in out] == [False, False, True, False]
    assert all(len(f) == 4 for _, f, _ in out if f is not None)


def test_prepare_command_is_incremental(tmp_path):
    data, cdir = tmp_path / "data", tmp_path / "cache"
    runner = CliRunner()
    res = runner.invoke(main, ["generate", "-o", str(data), "--start", "2026-10-04 00:00",
                               "--hours", "1", "--anomaly-hours", "0", "--num-flows", "200"])
    assert res.exit_code == 0, res.output
    args = ["prepare", "-d", str(data), "-s", "2026-10-04 00:00", "--duration", "1h",
            "--cache-dir", str(cdir), "--workers", "2"]
    first = json.loads(runner.invoke(main, args).output)
    second = json.loads(runner.invoke(main, args).output)
    assert first["newly_cached"] == 6 and first["already_cached"] == 0
    assert second["newly_cached"] == 0 and second["already_cached"] == 6
    assert len(list(Path(cdir).glob("*.flows.v1.pkl.gz"))) == 6
    np.testing.assert_array_equal(
        flow_hash(cache.read(cdir, next(data.glob("netflow.*.txt.gz")))).shape[0] > 0, True)
