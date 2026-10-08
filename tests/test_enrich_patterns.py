"""Enrichment (ports, prefixes), application rules and anomaly patterns."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from netflow_prototype.appbaseline import (
    AppBaseline,
    AppBaselineBuilder,
    AppRuleConfig,
    app_summaries,
    detect_app_events,
)
from netflow_prototype.enrich import (
    PORT_APPLICATIONS,
    PrefixEnricher,
    applications,
    enrich_prefixes,
    load_port_map,
)
from netflow_prototype.patterns import find_patterns

PREFIX_FILE = """PREFIX|ASN|ASN_CUSTOMER|SERVICE|IP_MODE
::/0|3598|CATCHALL-V6|ADI|IPV6
::/0|3598|CATCHALL-V6|HSIA|IPV6
2620:149::/32|64600|Northwind Retail|ADI|IPV6
0.0.0.0/1|65171|UNKNOWN|ADI|IPV4
0.0.0.0/1|65171|UNKNOWN|HSIA|IPV4
10.0.0.0/8|64512|Northwind Retail|ADI|IPV4
10.20.0.0/16|64513|Fabrikam Logistics|ADI-Lite|IPV4
"""


def test_5g_ports_are_named():
    ports = pd.Series(["2152", "2123", "8805", "38412", "36412", "3868", "38472"])
    assert list(applications(ports, PORT_APPLICATIONS)) == [
        "gtp-u", "gtp-c", "pfcp", "ngap", "s1ap", "diameter", "f1ap"]


def test_port_applications_and_overrides(tmp_path):
    ports = pd.Series(["22", "443", "53", "*", "771", "30000", "60000", "9999"])
    assert list(applications(ports, PORT_APPLICATIONS)) == [
        "ssh", "https", "dns", "any", "other-well-known", "other-registered",
        "other-dynamic", "other-registered"]
    custom = tmp_path / "ports.csv"
    custom.write_text("port,application\n9999,internal-api\n443,web-tls\n")
    mapped = applications(ports, load_port_map(custom))
    assert mapped[1] == "web-tls" and mapped[-1] == "internal-api" and mapped[0] == "ssh"


def test_prefix_longest_match(tmp_path):
    path = tmp_path / "prefixes.txt"
    path.write_text(PREFIX_FILE)
    enricher = PrefixEnricher.from_file(path)
    prefixes = pd.Series(["10.20.5.0/24", "10.99.0.0/24", "11.0.0.0/24",
                          "2620:149:a44::/48", "2600:1702::/44", "200.1.1.0/24", "bogus"])
    out = enricher.lookup(prefixes)
    assert list(out["customer"]) == ["Fabrikam Logistics", "Northwind Retail", "UNKNOWN",
                                     "Northwind Retail", "CATCHALL-V6", "", ""]
    assert out.loc[2, "service"] == "ADI;HSIA"            # several services joined
    assert out.loc[0, "matched_prefix"] == "10.20.0.0/16"  # most specific wins
    assert out.loc[5, "matched_prefix"] == ""              # 200/8 is outside 0.0.0.0/1
    again = enricher.lookup(prefixes)                       # memoized path
    assert out.equals(again)


def test_flow_customer_uses_the_known_side(tmp_path):
    path = tmp_path / "prefixes.txt"
    path.write_text(PREFIX_FILE)
    edges = pd.DataFrame({"srcIpPrefix": ["11.0.0.0/24", "10.20.1.0/24", "200.1.1.0/24"],
                          "dstIpPrefix": ["10.1.0.0/24", "11.1.0.0/24", "201.1.1.0/24"]})
    enrich_prefixes(edges, PrefixEnricher.from_file(path))
    assert list(edges["customer"]) == ["Northwind Retail", "Fabrikam Logistics", "unmatched"]
    assert list(edges["service"]) == ["ADI", "ADI-Lite", "unmatched"]


T0 = datetime(2026, 10, 6, 10, 0)


def _window(dns_bytes=1e6, https_bytes=1e8, pair_https=None, dns=True, any_port=True):
    rows = []
    for i in range(10):
        rows.append(("PE-A", "PE-B", f"10.0.{i}.0/24", "443", https_bytes / 10))
        if dns:
            rows.append(("PE-A", "PE-C", f"10.0.{i}.0/24", "53", dns_bytes / 10))
        if any_port:
            rows.append(("PE-D", "PE-E", f"10.1.{i}.0/24", "*", 5e6))
    if pair_https is not None:
        rows.append(("PE-X", "PE-Y", "10.9.0.0/24", "443", pair_https))
    return pd.DataFrame(rows, columns=["ingress", "egress", "srcIpPrefix", "dstPort", "bytes"])


def _app_baseline():
    rng = np.random.default_rng(0)
    builder = AppBaselineBuilder()
    for w in range(24):
        f = _window(dns_bytes=1e6 * rng.uniform(0.9, 1.1), https_bytes=1e8,
                    pair_https=1e6 * rng.uniform(0.8, 1.2))
        builder.add(T0 + timedelta(minutes=10 * w), *app_summaries(f, PORT_APPLICATIONS))
    apps, pairs = builder.build()
    return AppBaseline(apps.set_index(["application", "hour"]),
                       pairs.set_index(["application", "ingress", "egress"]))


def _events(f, cfg=None):
    f = f.assign(application=applications(f["dstPort"], PORT_APPLICATIONS))
    return detect_app_events(f, T0, _app_baseline(), cfg or AppRuleConfig(pair_min_windows=5))


def test_dns_disappearance_is_flagged():
    events = _events(_window(dns=False, pair_https=1e6))
    kinds = set(zip(events["kind"], events["application"]))
    assert ("application_disappeared", "dns") in kinds
    assert ("app_pair_disappeared", "dns") in kinds


def test_https_burst_is_expected_but_dns_surge_is_not():
    https_burst = _events(_window(https_bytes=1e9, pair_https=1e6))
    assert not (https_burst["kind"] == "application_surge").any()
    dns_surge = _events(_window(dns_bytes=1e8, pair_https=1e6))
    assert ((dns_surge["kind"] == "application_surge")
            & (dns_surge["application"] == "dns")).any()


def test_huge_https_on_one_pair_is_flagged():
    events = _events(_window(pair_https=1e9))
    hit = events[events["kind"] == "app_pair_surge"]
    assert list(zip(hit["application"], hit["ingress"], hit["egress"])) == [
        ("https", "PE-X", "PE-Y")]


def test_patterns_find_concentration_not_volume():
    rng = np.random.default_rng(1)
    n = 200
    edges = pd.DataFrame({
        "window": T0, "ingress": rng.choice(["PE-A", "PE-B", "PE-C", "PE-D"], n),
        "egress": rng.choice(["PE-E", "PE-F"], n),
        "application": np.where(np.arange(n) < 180, "https", "dns"),
        "bytes": 1.0, "is_anomalous": False,
    })
    # 6 anomalies, all on PE-D -> PE-F; 5 of them https (which is 90% of traffic).
    hot = edges.index[(edges["ingress"] == "PE-D") & (edges["egress"] == "PE-F")][:6]
    edges.loc[hot, "is_anomalous"] = True
    events = pd.DataFrame(columns=["window", "level", "ingress", "egress", "application",
                                   "bytes"])
    pats = find_patterns(edges, events).set_index(["scope", "dimension"])
    pair = pats.loc[("all windows", "router_pair")]
    assert pair["value"] == "PE-D -> PE-F" and pair["share"] == 1.0 and pair["is_pattern"]
    assert pair["statement"].startswith("All 6 anomalies are on router pair PE-D -> PE-F")
    app = pats.loc[("all windows", "application")]
    assert app["value"] == "https" and app["lift"] < 2.0
    assert not app["is_pattern"]  # https anomalies just follow where the traffic is


def test_untyped_any_traffic_is_never_judged():
    events = _events(_window(any_port=False, pair_https=1e6))
    assert not (events["application"] == "any").any()   # '*' vanished: no events
    assert events.empty


def test_only_tracked_applications_get_drop_rules():
    cfg = AppRuleConfig(pair_min_windows=5, tracked=("ntp",))
    events = _events(_window(dns=False, pair_https=1e6), cfg)
    assert events.empty  # DNS vanished but is not tracked in this configuration


def test_port_map_drift(tmp_path):
    from netflow_prototype.enrich import port_map_drift, save_port_map

    assert port_map_drift(tmp_path) is None  # no saved map: older baselines
    old = {p: a for p, a in PORT_APPLICATIONS.items() if a not in ("gtp-u", "pfcp")}
    save_port_map(old, tmp_path)
    assert port_map_drift(tmp_path) == ["gtp-u", "pfcp"]
    save_port_map(PORT_APPLICATIONS, tmp_path)
    assert port_map_drift(tmp_path) == []
