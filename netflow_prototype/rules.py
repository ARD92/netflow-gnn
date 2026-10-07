"""Rules applied to model scores across consecutive windows.

The model scores each flow in each window. These rules add what a single
window cannot show:

- Repeated violations: a flow whose route_miles flip between a low and a high
  value within ``flap_window`` (default 20 minutes, i.e. three consecutive
  10-minute exports: low -> high -> low or high -> low -> high).
- New flows: a flow with no training baseline is reported as "new_flow"
  (observed, not an anomaly). Thousands of first-seen new flows from one
  ingress router in one window form a router-level anomaly (a burst).
- Baselines: for every anomalous flow, the last normal observation (original
  time and values) and the time the current anomaly started.
- Graph table: the scored graph as a_node (ingress) / z_node (egress) rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from netflow_prototype.windows import DEFAULT_INTERVAL

STATUS_NORMAL = "normal"
STATUS_ANOMALY = "anomaly"
STATUS_REPEATED = "repeated_violation"
STATUS_NEW = "new_flow"
ANOMALOUS_STATUSES = (STATUS_ANOMALY, STATUS_REPEATED)


@dataclass
class RuleConfig:
    flap_window: timedelta = timedelta(minutes=20)
    flap_min_change: float = 0.25   # a route_miles step must exceed 25% of the lower value...
    flap_min_miles: float = 50.0    # ...and at least 50 miles
    flap_min_reversals: int = 1     # direction reversals needed (1: low -> high -> low)
    new_flow_burst: int = 1000      # first-seen new flows from one ingress in one window
    interval: timedelta = DEFAULT_INTERVAL

    @property
    def flap_steps(self) -> int:
        """Window-to-window steps examined; flapping needs reversals + 1 steps."""
        return max(self.flap_min_reversals + 1, int(self.flap_window / self.interval))


def detect_flaps(edges: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    """Find flows whose route_miles flip direction within ``cfg.flap_window``.

    Walking back from each window over consecutive observations of the same
    flow, a step counts when it changes route_miles by more than
    ``flap_min_change`` of the lower value and at least ``flap_min_miles``.
    A flow is a repeated violation in a window when the step into that window
    counts and the counted steps in the lookback reverse direction at least
    ``flap_min_reversals`` times.

    Returns columns ``flap_changes``, ``repeated_violation`` and
    ``miles_history`` aligned with ``edges.index``.
    """
    k = cfg.flap_steps
    e = edges[["flow_id", "window", "route_miles"]].sort_values(["flow_id", "window"])
    grouped = e.groupby("flow_id", sort=False)
    miles = [e["route_miles"].to_numpy(dtype=float)]
    wins = [e["window"]]
    for j in range(1, k + 1):
        miles.append(grouped["route_miles"].shift(j).to_numpy(dtype=float))
        wins.append(grouped["window"].shift(j))

    n = len(e)
    chain = np.ones(n, dtype=bool)        # observations so far are consecutive
    depth = np.zeros(n, dtype=int)        # how many steps back the chain reaches
    changes = np.zeros(n, dtype=int)
    reversals = np.zeros(n, dtype=int)
    last_sign = np.zeros(n)
    latest_step = np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        for j in range(1, k + 1):
            chain &= ((wins[j - 1] - wins[j]) == cfg.interval).to_numpy()
            newer, older = miles[j - 1], miles[j]
            step = newer - older
            limit = np.maximum(cfg.flap_min_miles,
                               cfg.flap_min_change * np.minimum(newer, older))
            counted = chain & (np.abs(step) >= limit)
            sign = np.sign(step) * counted
            reversals += counted & (last_sign != 0) & (sign == -last_sign)
            last_sign = np.where(counted, sign, last_sign)
            changes += counted
            depth += chain
            if j == 1:
                latest_step = counted

    repeated = latest_step & (reversals >= cfg.flap_min_reversals)
    history = np.full(n, "", dtype=object)
    for i in np.flatnonzero(repeated):
        points = [(wins[j].iloc[i], miles[j][i]) for j in range(depth[i], -1, -1)]
        history[i] = " -> ".join(f"{pd.Timestamp(t):%H:%M} {m:.1f}" for t, m in points)

    out = pd.DataFrame({"flap_changes": changes, "repeated_violation": repeated,
                        "miles_history": history}, index=e.index)
    return out.reindex(edges.index)


def apply_rules(edges: pd.DataFrame, cfg: RuleConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each flow a status and find new-flow bursts.

    Status precedence: repeated_violation, then new_flow (no training
    baseline; observed, not an anomaly), then anomaly (model score at or
    above the edge threshold), else normal.

    Returns the edges with ``status``, ``is_anomalous``, ``first_seen`` and
    ``in_new_flow_burst`` added, and one row per (window, ingress) burst.
    """
    edges = edges.join(detect_flaps(edges, cfg))
    edges["status"] = np.select(
        [edges["repeated_violation"], edges["is_new_flow"], edges["model_flag"]],
        [STATUS_REPEATED, STATUS_NEW, STATUS_ANOMALY],
        default=STATUS_NORMAL,
    )
    edges["is_anomalous"] = edges["status"].isin(ANOMALOUS_STATUSES)

    # First observation of each flow in this run; only those count toward bursts.
    by_time = edges.sort_values("window", kind="stable")
    edges["first_seen"] = (~by_time["flow_id"].duplicated()).reindex(edges.index)
    first_new = edges["is_new_flow"] & edges["first_seen"]
    counts = (edges[first_new].groupby(["window", "ingress"]).size()
              .rename("new_flows").reset_index())
    bursts = counts[counts["new_flows"] >= cfg.new_flow_burst].reset_index(drop=True)
    in_burst = pd.MultiIndex.from_frame(edges[["window", "ingress"]]).isin(
        pd.MultiIndex.from_frame(bursts[["window", "ingress"]]))
    edges["in_new_flow_burst"] = first_new & in_burst
    return edges, bursts


def add_baselines(edges: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    """Add original (baseline) time and values plus anomaly start for anomalous flows.

    The baseline is the flow's last normal observation earlier in this run;
    failing that, its value in the immediately preceding window (when that
    window was not flagged); failing that, the model's expected values
    (``baseline_source = "model"``).
    """
    e = edges.sort_values(["flow_id", "window"])
    flow = e["flow_id"]
    normal = ~e["is_anomalous"]
    marks = pd.DataFrame({
        "baseline_time": e["window"].where(normal),
        "baseline_bytes": e["bytes"].where(normal),
        "baseline_route_miles": e["route_miles"].where(normal),
    })
    base = marks.groupby(flow).shift(1).groupby(flow).ffill()
    source = pd.Series(np.where(base["baseline_time"].notna(), "observed", ""), index=e.index)

    # Fallback 1: previous-window values loaded with this window (context file).
    use_prev = base["baseline_time"].isna() & e["prev_present"]
    base.loc[use_prev, "baseline_time"] = e.loc[use_prev, "window"] - cfg.interval
    base.loc[use_prev, "baseline_bytes"] = e.loc[use_prev, "prev_bytes"]
    base.loc[use_prev, "baseline_route_miles"] = e.loc[use_prev, "prev_route_miles"]
    source[use_prev] = "observed"

    # Fallback 2: the model's expectation.
    use_model = base["baseline_time"].isna()
    base.loc[use_model, "baseline_bytes"] = e.loc[use_model, "expected_bytes"]
    base.loc[use_model, "baseline_route_miles"] = e.loc[use_model, "expected_route_miles"]
    source[use_model] = "model"

    # When did the current anomalous streak start?
    prev_anom = e.groupby(flow)["is_anomalous"].shift(1).fillna(False).astype(bool)
    prev_win = e.groupby(flow)["window"].shift(1)
    continues = prev_anom & ((e["window"] - prev_win) == cfg.interval)
    start = e["window"].where(~continues).groupby(flow).ffill()

    anomalous = e["is_anomalous"]
    out = base.assign(baseline_source=source, anomaly_start_time=start)
    for col in out.columns:  # only meaningful for anomalous flows
        out[col] = out[col].where(anomalous)
    return edges.join(out.reindex(edges.index))


def graph_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """The scored graph per window: one row per a_node (ingress) -> z_node (egress)."""
    f = edges.assign(
        _bm=edges["bytes"] * edges["route_miles"],
        _rep=edges["status"] == STATUS_REPEATED,
    )
    out = f.groupby(["window", "ingress", "egress"], sort=True).agg(
        flows=("bytes", "size"),
        bytes=("bytes", "sum"),
        _bm=("_bm", "sum"),
        anomalous_flows=("is_anomalous", "sum"),
        repeated_violations=("_rep", "sum"),
        new_flows=("is_new_flow", "sum"),
        max_score=("score", "max"),
    ).reset_index()
    out.insert(5, "route_miles", out["_bm"] / out["bytes"].where(out["bytes"] > 0))
    out["status"] = np.where(out["anomalous_flows"] > 0, "anomalous", "normal")
    return (out.drop(columns="_bm")
            .rename(columns={"ingress": "a_node", "egress": "z_node"}))
