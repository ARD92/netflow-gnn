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


def _io_options(func):
    """Flow cache and parallel reading, shared by commands that read exports."""
    options = [
        click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path),
                     envvar="NETFLOW_CACHE_DIR", show_envvar=True,
                     help="Reuse parsed exports from this cache (see the prepare command); "
                          "missing entries are parsed and added."),
        click.option("--workers", default=4, show_default=True,
                     help="Exports parsed in parallel."),
    ]
    for option in reversed(options):
        func = option(func)
    return func


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
@click.option("--router-buckets", default=8192, show_default=True,
              help="Router ID hash slots; keep well above the number of routers.")
@click.option("--prefix-buckets", default=16384, show_default=True,
              help="Prefix hash slots; raise (e.g. 131072) for millions of prefixes.")
@click.option("--max-flows-per-window", type=int, default=None,
              help="Randomly sample at most N flows per window (large files).")
@click.option("--chunk-size", default=250_000, show_default=True,
              help="Flows processed per GPU chunk; lower it to use less memory.")
@click.option("--device", default="cpu", show_default=True, help="cpu, cuda, or mps.")
@click.option("--max-gpu-mem-gb", default=16.0, show_default=True,
              help="Hard cap on GPU memory used by this process (0 = no cap).")
@click.option("--seed", default=7, show_default=True)
@_io_options
@click.option("--port-map", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="port,application file adding to or overriding the built-in port names.")
def train(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
          model_dir: Path, epochs: int, lr: float, hidden_dim: int, num_layers: int,
          dropout: float, router_buckets: int, prefix_buckets: int,
          max_flows_per_window: int | None, chunk_size: int, device: str,
          max_gpu_mem_gb: float, seed: int, cache_dir: Path | None, workers: int,
          port_map: Path | None) -> None:
    """Train the GNN on all files in a time window."""
    from netflow_prototype.model import ModelConfig
    from netflow_prototype.train import TrainConfig
    from netflow_prototype.train import train as run_train

    _setup_logging(verbose)
    fileset = _resolve(data_dir, start, end, duration)
    summary = run_train(
        fileset,
        model_dir,
        ModelConfig(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
                    router_buckets=router_buckets, prefix_buckets=prefix_buckets),
        TrainConfig(epochs=epochs, lr=lr, max_flows_per_window=max_flows_per_window,
                    chunk_size=chunk_size, device=device,
                    max_gpu_mem_gb=max_gpu_mem_gb or None, seed=seed),
        context=context_file(data_dir, fileset.files[0][0], DEFAULT_INTERVAL),
        cache_dir=cache_dir,
        workers=workers,
        port_map=port_map,
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
@click.option("--miles-min-change", default=0.10, show_default=True,
              help="Flag route_miles outside the router pair's usual range by this fraction "
                   "of its median...")
@click.option("--miles-min-change-abs", default=40.0, show_default=True,
              help="...and by at least this many miles.")
@click.option("--no-pair-baseline", is_flag=True,
              help="Judge route_miles with the model instead of the router-pair baseline.")
@click.option("--new-flow-burst", default=1000, show_default=True,
              help="First-seen new flows from one ingress in one window that make an anomaly.")
@_io_options
@click.option("--port-map", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="port,application file adding to or overriding the built-in port names.")
@click.option("--enrich-prefixes", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="PREFIX|ASN|ASN_CUSTOMER|SERVICE|IP_MODE file mapping prefixes to "
                   "customers and services.")
@click.option("--tracked-apps", default=None,
              help="Comma-separated applications whose drop or disappearance is an anomaly "
                   "(default: DNS, NTP, DHCP, RADIUS, Diameter, LDAP, Kerberos, BGP, SIP, "
                   "GTP-C/U, PFCP, NGAP, XnAP, E1AP, F1AP, S1AP, X2AP).")
@click.option("--burst-tolerant-apps", default="https,http,http-alt,https-alt,gtp-u",
              show_default=True, help="Applications whose network-wide surges are expected.")
@click.option("--app-drop-fraction", default=0.2, show_default=True,
              help="Application below this fraction of its usual bytes is a drop.")
@click.option("--app-surge-factor", default=5.0, show_default=True,
              help="Application above this multiple of usual bytes is a surge.")
@click.option("--app-pair-surge-factor", default=20.0, show_default=True,
              help="Application on one router pair above this multiple of usual is a surge.")
@click.option("--pattern-min-share", default=0.5, show_default=True,
              help="Share of anomalies a value needs to be reported as a pattern...")
@click.option("--pattern-min-lift", default=2.0, show_default=True,
              help="...and how much more concentrated than traffic it must be.")
def infer(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
          model_dir: Path, file_paths: tuple[Path, ...], out_dir: Path,
          labels_dir: Path | None, all_edges: bool, top: int, device: str,
          max_gpu_mem_gb: float, chunk_size: int, write_report: bool, report_max_flows: int,
          flap_window: str, flap_min_change: float, flap_min_miles: float,
          flap_min_reversals: int, miles_min_change: float, miles_min_change_abs: float,
          no_pair_baseline: bool, new_flow_burst: int, cache_dir: Path | None,
          workers: int, port_map: Path | None, enrich_prefixes: Path | None,
          tracked_apps: str | None, burst_tolerant_apps: str, app_drop_fraction: float, app_surge_factor: float,
          app_pair_surge_factor: float, pattern_min_share: float,
          pattern_min_lift: float) -> None:
    """Score a new time window (or explicit files) with a trained model."""
    import pandas as pd

    from netflow_prototype.infer import infer as run_infer
    from netflow_prototype.appbaseline import AppRuleConfig
    from netflow_prototype.baselines import PairMilesConfig
    from netflow_prototype.patterns import PatternConfig
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
                        pair_miles=PairMilesConfig(min_change=miles_min_change,
                                                   min_change_miles=miles_min_change_abs),
                        use_pair_baseline=not no_pair_baseline,
                        cache_dir=cache_dir, workers=workers,
                        report_max_flows=report_max_flows, port_map=port_map,
                        enrich_prefixes_file=enrich_prefixes,
                        app_rules=AppRuleConfig(
                            **({"tracked": tuple(a.strip() for a in tracked_apps.split(",")
                                                 if a.strip())} if tracked_apps else {}),
                            burst_tolerant=tuple(a.strip() for a in
                                                 burst_tolerant_apps.split(",") if a.strip()),
                            drop_fraction=app_drop_fraction, surge_factor=app_surge_factor,
                            pair_surge_factor=app_pair_surge_factor),
                        patterns_cfg=PatternConfig(min_share=pattern_min_share,
                                                   min_lift=pattern_min_lift))

    windows = pd.read_csv(out_dir / "window_scores.csv")
    click.echo("\nWindows flagged as anomalous:")
    flagged_windows = windows[windows["is_anomalous"]]
    click.echo(flagged_windows[["window", "flows", "flagged_flows", "repeated_violations",
                                "new_flow_bursts", "app_events", "max_score"]]
               .to_string(index=False) if len(flagged_windows) else "  none")

    patterns = pd.read_csv(out_dir / "anomaly_patterns.csv")
    found = patterns[(patterns["scope"] == "all windows") & patterns["is_pattern"]]
    click.echo("\nPatterns across all windows:")
    click.echo("\n".join(f"  - {s}" for s in found["statement"]) if len(found) else "  none")

    edges = pd.read_csv(out_dir / "edge_anomalies.csv")
    click.echo(f"\nTop {min(top, len(edges))} anomalous flows:")
    if len(edges):
        cols = ["detected_time", "status", "ingress", "egress", "srcIpPrefix",
                "dstIpPrefix", "dstPort", "score", "reason"]
        with pd.option_context("display.max_colwidth", 60, "display.width", 250):
            click.echo(edges[cols].head(top).to_string(index=False))
    click.echo("\n" + json.dumps(summary, indent=2, default=str))


@main.command()
@_window_options
@click.option("--cache-dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              envvar="NETFLOW_CACHE_DIR", show_envvar=True,
              help="Where parsed exports are stored.")
@click.option("--workers", default=8, show_default=True, help="Exports parsed in parallel.")
def prepare(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
            cache_dir: Path, workers: int) -> None:
    """Parse exports once into the flow cache (only new or changed files)."""
    from netflow_prototype.cache import prepare as run_prepare

    _setup_logging(verbose)
    fileset = _resolve(data_dir, start, end, duration)

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            logger.info("Prepared %d/%d files", done, total)

    result = run_prepare(fileset.paths, cache_dir, workers=workers, progress=progress)
    click.echo(json.dumps({
        "newly_cached": len(result["cached"]),
        "already_cached": len(result["fresh"]),
        "skipped_unreadable": result["skipped"],
        "missing_windows": len(fileset.missing),
        "cache_dir": str(cache_dir),
    }, indent=2))


def _build_baselines(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
                     model_dir: Path, cache_dir: Path | None, workers: int,
                     port_map: Path | None) -> None:
    from netflow_prototype import baselines
    from netflow_prototype.enrich import load_port_map, save_port_map

    _setup_logging(verbose)
    fileset = _resolve(data_dir, start, end, duration)
    built = baselines.build_from_files(fileset.paths, load_port_map(port_map),
                                       workers=workers, cache_dir=cache_dir)
    counts = built.save(model_dir)
    save_port_map(load_port_map(port_map), model_dir)
    click.echo(json.dumps({**counts, "windows": len(built.apps.windows),
                           "model_dir": str(model_dir)}, indent=2))


@main.command("baselines")
@_window_options
@click.option("--model-dir", "-m", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Model folder to write the baselines to.")
@_io_options
@click.option("--port-map", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="port,application file adding to or overriding the built-in port names.")
def build_baselines(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
                    model_dir: Path, cache_dir: Path | None, workers: int,
                    port_map: Path | None) -> None:
    """Measure router-pair route_miles and application byte baselines from files (no training)."""
    _build_baselines(data_dir, start, end, duration, verbose, model_dir, cache_dir, workers,
                     port_map)


@main.command("pair-baseline", hidden=True)
@_window_options
@click.option("--model-dir", "-m", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@_io_options
@click.option("--port-map", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="port,application file adding to or overriding the built-in port names.")
def pair_baseline(data_dir: Path, start: str, end: str, duration: str, verbose: bool,
                  model_dir: Path, cache_dir: Path | None, workers: int,
                  port_map: Path | None) -> None:
    """Alias of the baselines command."""
    _build_baselines(data_dir, start, end, duration, verbose, model_dir, cache_dir, workers,
                     port_map)


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
