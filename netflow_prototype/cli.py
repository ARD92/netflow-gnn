"""Command-line interface.

    python -m netflow_prototype generate  -- write synthetic files in the production schema
    python -m netflow_prototype files     -- show which files a time window resolves to
    python -m netflow_prototype train     -- train a model on a time window
    python -m netflow_prototype infer     -- score a time window (or explicit files)
    python -m netflow_prototype visualize -- draw one window's router graph as an image
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from netflow_prototype.data import context_file
from netflow_prototype.windows import (
    DEFAULT_INTERVAL,
    FileSet,
    files_from_paths,
    parse_duration,
    parse_time,
    resolve_files,
    resolve_range,
)

logger = logging.getLogger("netflow_prototype")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def _window_options(func):
    """Shared options that turn a time window into a list of files."""
    options = [
        click.option("--data-dir", "-d", type=click.Path(file_okay=False, path_type=Path),
                     help="Directory containing netflow.YYYYMMDD.HH.MM.txt.gz files."),
        click.option("--start", "-s", help="Window start, e.g. '2026-10-06 00:00'."),
        click.option("--end", "-e", help="Window end (exclusive), e.g. '2026-10-06 12:00'."),
        click.option("--duration", help="Window length instead of --end, e.g. 6h, 90m, 2d."),
        click.option("--verbose", "-v", is_flag=True, help="Enable debug logging."),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def _resolve(data_dir: Path | None, start: str | None, end: str | None,
             duration: str | None) -> FileSet:
    if data_dir is None or start is None:
        raise click.UsageError("--data-dir and --start are required.")
    try:
        t0, t1 = resolve_range(start, end, duration)
        fileset = resolve_files(data_dir, t0, t1)
    except (ValueError, FileNotFoundError) as exc:
        raise click.UsageError(str(exc)) from exc
    logger.info("Window %s -> %s resolved to %d files (%d missing)",
                t0, t1, len(fileset.files), len(fileset.missing))
    if not fileset.files:
        raise click.UsageError(f"No files in {data_dir} between {t0} and {t1}.")
    return fileset


@click.group()
def main() -> None:
    """GNN-based anomaly detection for 10-minute NetFlow exports."""


@main.command()
@click.option("--out-dir", "-o", required=True, type=click.Path(path_type=Path))
@click.option("--start", "-s", default="2026-10-04 00:00", show_default=True)
@click.option("--hours", default=48.0, show_default=True, help="Total hours to generate.")
@click.option("--anomaly-hours", default=12.0, show_default=True,
              help="Inject anomalies only in the final N hours.")
@click.option("--num-flows", default=3000, show_default=True)
@click.option("--seed", default=7, show_default=True)
@click.option("--verbose", "-v", is_flag=True)
def generate(out_dir: Path, start: str, hours: float, anomaly_hours: float,
             num_flows: int, seed: int, verbose: bool) -> None:
    """Generate synthetic NetFlow files (same schema) with labeled anomalies."""
    from netflow_prototype.synthetic import SyntheticConfig
    from netflow_prototype.synthetic import generate as run_generate

    _setup_logging(verbose)
    cfg = SyntheticConfig(start=parse_time(start), hours=hours, anomaly_hours=anomaly_hours,
                          num_flows=num_flows, seed=seed)
    click.echo(json.dumps(run_generate(cfg, out_dir), indent=2))


@main.command()
@_window_options
def files(data_dir: Path, start: str, end: str, duration: str, verbose: bool) -> None:
    """List the files a time window resolves to (dry run)."""
    _setup_logging(verbose)
    fileset = _resolve(data_dir, start, end, duration)
    for ts, path in fileset.files:
        click.echo(f"{ts:%Y-%m-%d %H:%M}  {path}")
    for ts in fileset.missing:
        click.echo(f"{ts:%Y-%m-%d %H:%M}  MISSING")
    click.echo(f"{len(fileset.files)} files, {len(fileset.missing)} missing")


@main.command()
@_window_options
@click.option("--model-dir", "-m", required=True, type=click.Path(path_type=Path),
              help="Where to write model.pt, seen_flows.npy and train_summary.json.")
@click.option("--epochs", default=30, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--hidden-dim", default=64, show_default=True)
@click.option("--num-layers", default=2, show_default=True)
@click.option("--dropout", default=0.0, show_default=True)
@click.option("--max-flows-per-window", type=int, default=None,
              help="Randomly sample at most N flows per window (large files).")
@click.option("--chunk-size", default=250_000, show_default=True,
              help="Flows processed per GPU chunk; lower it to use less memory.")
@click.option("--device", default="cpu", show_default=True, help="cpu, cuda, or mps.")
@click.option("--max-gpu-mem-gb", default=16.0, show_default=True,
              help="Hard cap on GPU memory used by this process (0 = no cap).")
@click.option("--seed", default=7, show_default=True)
def train(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
          model_dir: Path, epochs: int, lr: float, hidden_dim: int, num_layers: int,
          dropout: float, max_flows_per_window: int | None, chunk_size: int, device: str,
          max_gpu_mem_gb: float, seed: int) -> None:
    """Train the GNN on all files in a time window."""
    from netflow_prototype.model import ModelConfig
    from netflow_prototype.train import TrainConfig
    from netflow_prototype.train import train as run_train

    _setup_logging(verbose)
    fileset = _resolve(data_dir, start, end, duration)
    summary = run_train(
        fileset,
        model_dir,
        ModelConfig(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout),
        TrainConfig(epochs=epochs, lr=lr, max_flows_per_window=max_flows_per_window,
                    chunk_size=chunk_size, device=device,
                    max_gpu_mem_gb=max_gpu_mem_gb or None, seed=seed),
        context=context_file(data_dir, fileset.files[0][0], DEFAULT_INTERVAL),
    )
    meta = {k: v for k, v in summary["meta"].items() if k != "files"}
    click.echo(json.dumps({"meta": meta, "thresholds": summary["thresholds"]}, indent=2))


@main.command()
@_window_options
@click.option("--model-dir", "-m", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--file", "-f", "file_paths", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Score explicit files instead of a time window (repeatable).")
@click.option("--out-dir", "-o", default="results", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--labels-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Optional label sidecars for evaluation (synthetic data).")
@click.option("--all-edges", is_flag=True, help="Also write every scored flow.")
@click.option("--top", default=15, show_default=True, help="Flagged flows to print.")
@click.option("--device", default="cpu", show_default=True)
@click.option("--max-gpu-mem-gb", default=16.0, show_default=True,
              help="Hard cap on GPU memory used by this process (0 = no cap).")
@click.option("--chunk-size", default=250_000, show_default=True,
              help="Flows scored per GPU chunk; lower it to use less memory.")
@click.option("--report", "write_report", is_flag=True,
              help="Also write anomaly_report.txt, a readable summary of the anomalies.")
@click.option("--report-max-flows", default=25, show_default=True,
              help="Anomalous flows listed per window in the report.")
@click.option("--flap-window", default="20m", show_default=True,
              help="route_miles reversing within this period is a repeated violation "
                   "(minimum 20m: three 10-minute exports).")
@click.option("--flap-min-change", default=0.25, show_default=True,
              help="A route_miles step counts when it exceeds this fraction of the lower value...")
@click.option("--flap-min-miles", default=50.0, show_default=True,
              help="...and at least this many miles.")
@click.option("--flap-min-reversals", default=1, show_default=True,
              help="Direction reversals needed within --flap-window (1: low-high-low).")
@click.option("--new-flow-burst", default=1000, show_default=True,
              help="First-seen new flows from one ingress in one window that make an anomaly.")
def infer(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
          model_dir: Path, file_paths: tuple[Path, ...], out_dir: Path,
          labels_dir: Path | None, all_edges: bool, top: int, device: str,
          max_gpu_mem_gb: float, chunk_size: int, write_report: bool, report_max_flows: int,
          flap_window: str, flap_min_change: float, flap_min_miles: float,
          flap_min_reversals: int, new_flow_burst: int) -> None:
    """Score a new time window (or explicit files) with a trained model."""
    import pandas as pd

    from netflow_prototype.infer import infer as run_infer
    from netflow_prototype.rules import RuleConfig

    _setup_logging(verbose)
    try:
        rules = RuleConfig(flap_window=parse_duration(flap_window),
                           flap_min_change=flap_min_change, flap_min_miles=flap_min_miles,
                           flap_min_reversals=flap_min_reversals,
                           new_flow_burst=new_flow_burst)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if file_paths:
        fileset = files_from_paths(list(file_paths))
        context = context_file(fileset.paths[0].parent, fileset.files[0][0])
    else:
        fileset = _resolve(data_dir, start, end, duration)
        context = context_file(data_dir, fileset.files[0][0])

    summary = run_infer(model_dir, fileset, out_dir, context=context,
                        labels_dir=labels_dir, write_all_edges=all_edges, device=device,
                        max_gpu_mem_gb=max_gpu_mem_gb or None, chunk_size=chunk_size,
                        rules=rules, write_readable_report=write_report,
                        report_max_flows=report_max_flows)

    windows = pd.read_csv(out_dir / "window_scores.csv")
    click.echo("\nWindows flagged as anomalous:")
    flagged_windows = windows[windows["is_anomalous"]]
    click.echo(flagged_windows[["window", "flows", "flagged_flows", "repeated_violations",
                                "new_flow_bursts", "max_score"]].to_string(index=False)
               if len(flagged_windows) else "  none")

    edges = pd.read_csv(out_dir / "edge_anomalies.csv")
    click.echo(f"\nTop {min(top, len(edges))} anomalous flows:")
    if len(edges):
        cols = ["detected_time", "status", "ingress", "egress", "srcIpPrefix",
                "dstIpPrefix", "dstPort", "score", "reason"]
        with pd.option_context("display.max_colwidth", 60, "display.width", 250):
            click.echo(edges[cols].head(top).to_string(index=False))
    click.echo("\n" + json.dumps(summary, indent=2, default=str))


@main.command()
@click.option("--file", "-f", "file_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="NetFlow file (one 10-minute window) to draw.")
@click.option("--results", "-r", "results_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Inference output directory; colors flagged flows and routers.")
@click.option("--out", "-o", default="graph.png", show_default=True,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Image path; the extension sets the format (.png, .svg, .pdf).")
@click.option("--router", help="Only draw router pairs that include this router.")
@click.option("--max-pairs", default=300, show_default=True,
              help="Draw the heaviest N router pairs (flagged pairs are always drawn).")
@click.option("--labels", type=click.Choice(["auto", "all", "none"]), default="auto",
              show_default=True, help="auto labels all routers up to 80, else the busiest.")
@click.option("--verbose", "-v", is_flag=True)
def visualize(file_path: Path, results_dir: Path | None, out: Path, router: str | None,
              max_pairs: int, labels: str, verbose: bool) -> None:
    """Draw one window's router graph as an image."""
    from netflow_prototype.visualize import render_window
    from netflow_prototype.windows import file_timestamp

    _setup_logging(verbose)
    try:
        summary = render_window(file_path, out, results_dir=results_dir,
                                timestamp=file_timestamp(file_path), router=router,
                                max_pairs=max_pairs, labels=labels)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
