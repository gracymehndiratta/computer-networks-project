"""Resolver-Side Traditional DNS Feature Extractor (Stage 2, active).

Computes the eleven traditional DNS traffic features of the base paper for
escalated flows, using resolver/gateway visibility into the *decrypted* DNS
transaction. This module runs in parallel with the blackhole-probing path,
not after it.

The eleven features:
 1. txn_interarrival_mean — mean inter-arrival between DNS transactions
 2. ttl_variance          — variance of response TTLs
 3. query_entropy         — Shannon entropy over query-name label tokens
 4. subdomain_depth_mean  — mean number of labels per query name
 5. domain_switch_rate    — rate of switches between registrable domains
 6. qname_len_mean        — mean query-name length (chars)
 7. nxdomain_ratio        — fraction of NXDOMAIN responses
 8. response_size_mean    — mean DNS response size (bytes)
 9. qtype_diversity       — distinct qtypes / total queries
10. query_rate            — queries per second within the flow
11. qname_payload_ratio   — query-name bytes / total time (payload traffic)
"""
from __future__ import annotations

import math
from collections import Counter

from ..datamodel import Flow

EPS = 1e-9

DNS_FEATURE_NAMES = [
    "txn_interarrival_mean",
    "ttl_variance",
    "query_entropy",
    "subdomain_depth_mean",
    "domain_switch_rate",
    "qname_len_mean",
    "nxdomain_ratio",
    "response_size_mean",
    "qtype_diversity",
    "query_rate",
    "qname_payload_ratio",
]


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counter.values()))


def _registrable(qname: str) -> str:
    parts = [p for p in qname.strip(".").lower().split(".") if p]
    return ".".join(parts[-2:]) if len(parts) >= 2 else qname.lower()


def extract_traditional_dns_features(flow: Flow) -> dict[str, float]:
    """Eleven traditional DNS features; all zeros when the input has no
    resolver-side visibility (e.g., a bare on-path encrypted capture)."""
    txns = sorted(flow.txns, key=lambda x: x.t)
    if not txns:
        return {name: 0.0 for name in DNS_FEATURE_NAMES}

    times = [t.t for t in txns]
    iats = [b - a for a, b in zip(times, times[1:])]
    interarrival_mean = sum(iats) / len(iats) if iats else 0.0

    ttls = [float(t.ttl) for t in txns]
    ttl_mean = sum(ttls) / len(ttls)
    ttl_variance = sum((x - ttl_mean) ** 2 for x in ttls) / len(ttls)

    # Entropy over label tokens across all query names (word-level).
    token_counter: Counter = Counter()
    for t in txns:
        for label in t.qname.strip(".").lower().split("."):
            token_counter[label] += 1
    query_entropy = _entropy(token_counter)

    depths = [len([p for p in t.qname.strip(".").split(".") if p]) for t in txns]
    subdomain_depth_mean = sum(depths) / len(depths)

    domains = [_registrable(t.qname) for t in txns]
    switches = sum(1 for a, b in zip(domains, domains[1:]) if a != b)
    domain_switch_rate = switches / max(len(domains) - 1, 1)

    qname_len_mean = sum(len(t.qname) for t in txns) / len(txns)
    nxdomain_ratio = sum(1 for t in txns if t.nxdomain) / len(txns)
    response_size_mean = sum(t.resp_size for t in txns) / len(txns)
    qtype_diversity = len({t.qtype for t in txns}) / len(txns)

    duration = max(times[-1] - times[0], EPS)
    query_rate = len(txns) / duration if len(txns) > 1 else 0.0
    qname_payload_bytes = sum(len(t.qname) for t in txns)
    qname_payload_ratio = qname_payload_bytes / duration

    return {
        "txn_interarrival_mean": float(interarrival_mean),
        "ttl_variance": float(ttl_variance),
        "query_entropy": float(query_entropy),
        "subdomain_depth_mean": float(subdomain_depth_mean),
        "domain_switch_rate": float(domain_switch_rate),
        "qname_len_mean": float(qname_len_mean),
        "nxdomain_ratio": float(nxdomain_ratio),
        "response_size_mean": float(response_size_mean),
        "qtype_diversity": float(qtype_diversity),
        "query_rate": float(query_rate),
        "qname_payload_ratio": float(qname_payload_ratio),
    }
