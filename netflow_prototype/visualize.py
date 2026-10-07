"""Render one window's router graph as an image (PNG, SVG, or PDF).

The picture shows the same graph the model uses, collapsed to router pairs:

    node   = router (ingress or egress), grouped by site, sized by bytes handled
    arrow  = ingress -> egress router pair, width ~ log(bytes) of all its flows

When an inference output directory is given, router pairs carrying flagged
flows are drawn in red (darker = higher anomaly score) and anomalous routers
get a red outline.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from netflow_prototype.schema import load_window

NORMAL_EDGE = "#8a96a3"
FLAG_EDGE = "#c0392b"
# Site colors deliberately exclude reds so they never read as "anomalous".
SITE_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b",
               "#bcbd22", "#7f7f7f", "#aec7e8", "#98df8a", "#ffbb78", "#c5b0d5",
               "#9edae5", "#c49c94", "#dbdb8d", "#5254a3"]


def router_sites(path: str | Path) -> dict[str, str]:
    """Map router -> site code from the src_snrc / dst_snrc columns when present."""
    cols = ["ingress", "egress", "src_snrc", "dst_snrc"]
    try:
        df = pd.read_csv(path, sep="|", usecols=lambda c: c in cols, dtype=str,
                         keep_default_na=False, compression="infer")
    except (EOFError, OSError, ValueError):
        return {}
    sites: dict[str, str] = {}
    for router_col, site_col in (("ingress", "src_snrc"), ("egress", "dst_snrc")):
        if router_col in df and site_col in df:
            pairs = df[[router_col, site_col]].drop_duplicates(router_col)
            sites.update(zip(pairs[router_col], pairs[site_col], strict=True))
    return {r: s for r, s in sites.items() if s}


def router_pairs(flows: pd.DataFrame) -> pd.DataFrame:
    """Collapse flows to one row per (ingress, egress) router pair."""
    f = flows.assign(_bm=flows["bytes"] * flows["route_miles"])
    pairs = f.groupby(["ingress", "egress"], sort=False).agg(
        flows=("bytes", "size"), bytes=("bytes", "sum"), _bm=("_bm", "sum"),
    ).reset_index()
    pairs["route_miles"] = pairs["_bm"] / pairs["bytes"].clip(lower=1e-9)
    return pairs.drop(columns="_bm")


def attach_scores(pairs: pd.DataFrame, results_dir: str | Path | None,
                  ts: datetime | None) -> tuple[pd.DataFrame, set[str]]:
    """Add flagged-flow counts per pair and the set of anomalous routers."""
    pairs = pairs.assign(flagged_flows=0, max_score=0.0, top_reason="")
    if results_dir is None:
        return pairs, set()
    results_dir = Path(results_dir)

    def this_window(df: pd.DataFrame) -> pd.DataFrame:
        if "detected_time" in df and "window" not in df:
            df = df.rename(columns={"detected_time": "window"})
        if ts is None or "window" not in df:
            return df
        return df[pd.to_datetime(df["window"]) == pd.Timestamp(ts)]

    bad_routers: set[str] = set()
    edges_path = results_dir / "edge_anomalies.csv"
    if edges_path.exists():
        edges = this_window(pd.read_csv(edges_path, dtype={"dstPort": str}))
        if len(edges):
            edges = edges.sort_values("score", ascending=False)
            agg = edges.groupby(["ingress", "egress"]).agg(
                flagged_flows=("score", "size"), max_score=("score", "max"),
                top_reason=("reason", "first"),
            ).reset_index()
            pairs = pairs.drop(columns=["flagged_flows", "max_score", "top_reason"]).merge(
                agg, on=["ingress", "egress"], how="left")
            pairs["flagged_flows"] = pairs["flagged_flows"].fillna(0).astype(int)
            pairs["max_score"] = pairs["max_score"].fillna(0.0)
            pairs["top_reason"] = pairs["top_reason"].fillna("")
    nodes_path = results_dir / "node_scores.csv"
    if nodes_path.exists():
        nodes = this_window(pd.read_csv(nodes_path))
        bad_routers = set(nodes.loc[nodes["is_anomalous"].astype(bool), "router"])
    return pairs, bad_routers


def layout(routers: list[str], sites: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Place sites on a circle and each site's routers on a small ring around it."""
    groups: dict[str, list[str]] = {}
    for r in routers:
        groups.setdefault(sites.get(r, "unknown"), []).append(r)
    names = sorted(groups)
    pos: dict[str, tuple[float, float]] = {}
    for i, site in enumerate(names):
        angle = 2 * math.pi * i / max(len(names), 1)
        cx, cy = (math.cos(angle), math.sin(angle)) if len(names) > 1 else (0.0, 0.0)
        members = sorted(groups[site])
        ring = 0.0 if len(members) == 1 else min(0.4, 0.12 + 0.03 * len(members))
        if len(names) == 1:
            ring = 1.0
        for j, r in enumerate(members):
            a = 2 * math.pi * j / len(members) + angle
            pos[r] = (cx + ring * math.cos(a), cy + ring * math.sin(a))
    return pos


def _scale(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0 or np.ptp(v) == 0:
        return np.full(len(v), (lo + hi) / 2)
    return lo + (hi - lo) * (v - v.min()) / np.ptp(v)


def render_window(
    file_path: str | Path,
    out_path: str | Path,
    results_dir: str | Path | None = None,
    timestamp: datetime | None = None,
    router: str | None = None,
    max_pairs: int = 300,
    labels: str = "auto",
) -> dict:
    """Draw the router graph of one NetFlow file and save it to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    flows = load_window(file_path)
    sites = router_sites(file_path)
    pairs, bad_routers = attach_scores(router_pairs(flows), results_dir, timestamp)
    if router:
        pairs = pairs[(pairs["ingress"] == router) | (pairs["egress"] == router)]
        if pairs.empty:
            raise ValueError(f"Router {router} has no flows in {Path(file_path).name}")

    # Keep every flagged pair plus the heaviest remaining pairs.
    total_pairs = len(pairs)
    flagged = pairs[pairs["flagged_flows"] > 0]
    rest = pairs[pairs["flagged_flows"] == 0].nlargest(max(0, max_pairs - len(flagged)), "bytes")
    shown = pd.concat([rest, flagged]).sort_values("flagged_flows")  # flagged drawn last

    routers = sorted(set(shown["ingress"]) | set(shown["egress"]))
    pos = layout(routers, sites)
    node_bytes = (pd.concat([
        pairs.groupby("ingress")["bytes"].sum(), pairs.groupby("egress")["bytes"].sum(),
    ]).groupby(level=0).sum().reindex(routers).fillna(0.0))

    n = len(routers)
    size = min(28.0, max(10.0, 6.0 + 0.12 * n))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_aspect("equal")
    ax.axis("off")

    widths = _scale(np.log1p(shown["bytes"].to_numpy()), 0.4, 4.5)
    max_score = max(float(shown["max_score"].max() or 0), 4.0)
    self_loops = 0
    for (_, row), width in zip(shown.iterrows(), widths, strict=True):
        if row["ingress"] == row["egress"]:
            self_loops += 1
            continue
        is_flagged = row["flagged_flows"] > 0
        if is_flagged:
            alpha = 0.55 + 0.45 * min(1.0, row["max_score"] / max_score)
            color, z = FLAG_EDGE, 3
        else:
            alpha, color, z = 0.35, NORMAL_EDGE, 1
        ax.add_patch(FancyArrowPatch(
            pos[row["ingress"]], pos[row["egress"]], connectionstyle="arc3,rad=0.12",
            arrowstyle="-|>", mutation_scale=7 + width * 2, lw=width, color=color,
            alpha=alpha, zorder=z, shrinkA=6, shrinkB=6,
        ))

    site_names = sorted({sites.get(r, "unknown") for r in routers})
    site_color = {s: SITE_COLORS[i % len(SITE_COLORS)] for i, s in enumerate(site_names)}
    xy = np.array([pos[r] for r in routers])
    node_size = _scale(np.sqrt(node_bytes.to_numpy()), 60, 900)
    ax.scatter(xy[:, 0], xy[:, 1], s=node_size, zorder=4,
               c=[site_color[sites.get(r, "unknown")] for r in routers],
               edgecolors=[FLAG_EDGE if r in bad_routers else "white" for r in routers],
               linewidths=[3.0 if r in bad_routers else 1.0 for r in routers])

    if labels != "none":
        top = set(node_bytes.nlargest(20).index)
        for r in routers:
            if labels == "all" or n <= 80 or r in bad_routers or r in top or r == router:
                x, y = pos[r]
                # Push the label away from the picture's center so site rings stay legible.
                norm = math.hypot(x, y) or 1.0
                dx, dy = x / norm, y / norm
                ax.annotate(r, (x, y), xytext=(12 * dx, 12 * dy), textcoords="offset points",
                            ha="left" if dx > 0.3 else "right" if dx < -0.3 else "center",
                            va="bottom" if dy > 0.3 else "top" if dy < -0.3 else "center",
                            fontsize=7, zorder=5,
                            color=FLAG_EDGE if r in bad_routers else "#222222")
    if len(site_names) > 1:
        # Same ordering as layout(): site k sits at angle 2*pi*k/num_sites.
        for k, s in enumerate(site_names):
            angle = 2 * math.pi * k / len(site_names)
            ax.text(1.9 * math.cos(angle), 1.9 * math.sin(angle), s, ha="center",
                    va="center", fontsize=10, fontweight="bold", color=site_color[s])

    window = timestamp.strftime("%Y-%m-%d %H:%M") if timestamp else Path(file_path).name
    n_flagged_pairs = int((shown["flagged_flows"] > 0).sum())
    title = (f"NetFlow router graph - window {window}"
             + (f" - router {router}" if router else "")
             + f"\n{n} routers, {total_pairs} router pairs ({len(shown)} shown), "
             f"{int(pairs['flows'].sum()):,} flows")
    if results_dir is not None:
        title += (f" | flagged: {int(pairs['flagged_flows'].sum())} flows on "
                  f"{n_flagged_pairs} pairs, {len(bad_routers & set(routers))} routers")
    ax.set_title(title, fontsize=12)
    ax.legend(handles=[
        Line2D([0], [0], color=NORMAL_EDGE, lw=2, label="router pair (width ~ log bytes)"),
        Line2D([0], [0], color=FLAG_EDGE, lw=2, label="pair with flagged flows"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb",
               markeredgecolor=FLAG_EDGE, markeredgewidth=2.5, markersize=11,
               label="anomalous router"),
    ], loc="lower left", fontsize=9, frameon=False)
    pad = 2.1 if len(site_names) > 1 else 1.3
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Companion table of the drawn pairs, flagged first.
    table = shown.sort_values(["flagged_flows", "bytes"], ascending=False)
    table.to_csv(out_path.with_suffix(".pairs.csv"), index=False, float_format="%.2f")
    return {
        "image": str(out_path),
        "pairs_csv": str(out_path.with_suffix(".pairs.csv")),
        "routers": n,
        "router_pairs": total_pairs,
        "pairs_shown": len(shown),
        "flagged_pairs_shown": n_flagged_pairs,
        "self_loops_skipped": self_loops,
    }
