"""Command-line interface.

    python -m netflow_prototype generate  -- write synthetic files in the production schema
    python -m netflow_prototype files     -- show which files a time window resolves to
    python -m netflow_prototype train     -- train a model on a time window
    python -m netflow_prototype infer     -- score a time window (or explicit files)
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
@click.option("--max-flows-per-window", type=int, default=None,
              help="Randomly sample at most N flows per window (large files).")
@click.option("--device", default="cpu", show_default=True, help="cpu, cuda, or mps.")
@click.option("--seed", default=7, show_default=True)
def train(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
          model_dir: Path, epochs: int, lr: float, hidden_dim: int, num_layers: int,
          max_flows_per_window: int | None, device: str, seed: int) -> None:
    """Train the GNN on all files in a time window."""
    from netflow_prototype.model import ModelConfig
    from netflow_prototype.train import TrainConfig
    from netflow_prototype.train import train as run_train

    _setup_logging(verbose)
    fileset = _resolve(data_dir, start, end, duration)
    summary = run_train(
        fileset,
        model_dir,
        ModelConfig(hidden_dim=hidden_dim, num_layers=num_layers),
        TrainConfig(epochs=epochs, lr=lr, max_flows_per_window=max_flows_per_window,
                    device=device, seed=seed),
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
def infer(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
          model_dir: Path, file_paths: tuple[Path, ...], out_dir: Path,
          labels_dir: Path | None, all_edges: bool, top: int, device: str) -> None:
    """Score a new time window (or explicit files) with a trained model."""
    import pandas as pd

    from netflow_prototype.infer import infer as run_infer

    _setup_logging(verbose)
    if file_paths:
        fileset = files_from_paths(list(file_paths))
        context = context_file(fileset.paths[0].parent, fileset.files[0][0])
    else:
        fileset = _resolve(data_dir, start, end, duration)
        context = context_file(data_dir, fileset.files[0][0])

    summary = run_infer(model_dir, fileset, out_dir, context=context,
                        labels_dir=labels_dir, write_all_edges=all_edges, device=device)

    windows = pd.read_csv(out_dir / "window_scores.csv")
    click.echo("\nWindows flagged as anomalous:")
    flagged_windows = windows[windows["is_anomalous"]]
    click.echo(flagged_windows[["window", "flows", "flagged_flows", "flagged_fraction",
                                "max_score"]].to_string(index=False)
               if len(flagged_windows) else "  none")

    edges = pd.read_csv(out_dir / "edge_anomalies.csv")
    click.echo(f"\nTop {min(top, len(edges))} anomalous flows:")
    if len(edges):
        cols = ["window", "ingress", "egress", "srcIpPrefix", "dstIpPrefix", "dstPort",
                "flow_type", "score", "reason"]
        with pd.option_context("display.max_colwidth", 60, "display.width", 250):
            click.echo(edges[cols].head(top).to_string(index=False))
    click.echo("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
