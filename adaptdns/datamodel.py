"""Core data model shared by all AdaptDNS components.

The pipeline works on *flows*. A flow is a grouping of packets belonging to
one encrypted DNS session (one TCP/TLS connection to a resolver endpoint).
Because DoT/DoH multiplex several DNS transactions per connection, a flow is
a flow-level grouping — not an exact one-query-per-session reconstruction.

`Flow.packets` is all a bare on-path observer of encrypted traffic could see.
`Flow.txns` and `Flow.retry_delays` model resolver/gateway-side visibility
(the deployment assumption in the case study, Section 1) and may be empty for
pcap-only inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LABEL_BENIGN = "benign"
LABEL_TUNNEL = "tunnel"

VARIANT_BENIGN = "benign"
VARIANT_NAIVE = "naive_tunnel"
VARIANT_ADAPTIVE = "adaptive_tunnel"


@dataclass
class Packet:
    """One packet of an encrypted flow.

    t:      timestamp (seconds, absolute or flow-relative — only deltas used)
    size:   on-wire size in bytes
    direction: +1 client -> resolver, -1 resolver -> client
    """

    t: float
    size: int
    direction: int


@dataclass
class DnsTxn:
    """Resolver-side view of one DNS transaction carried inside the flow."""

    t: float
    qname: str
    qtype: str
    ttl: int
    resp_size: int
    nxdomain: bool = False


@dataclass
class Flow:
    flow_id: str
    src: str
    dst: str
    proto: str = "tcp"          # transport to the resolver endpoint
    packets: list[Packet] = field(default_factory=list)
    txns: list[DnsTxn] = field(default_factory=list)

    # Client retry schedule if the resolver suppresses (blackholes) the first
    # response of the probed transaction: list of delay seconds per retry.
    # None = unknown (e.g., pcap-only input where the client is not modelled).
    retry_delays: list[float] | None = None

    label: str = LABEL_BENIGN    # ground truth for evaluation
    variant: str = VARIANT_BENIGN
    resolver_visible: bool = True  # False => Stage-2 DNS features unavailable

    # Populated by Stage 1 / Stage 2 during a pipeline run.
    stage1_features: dict[str, float] = field(default_factory=dict)
    suspicion_score: float | None = None
    escalated: bool = False
    stage2_features: dict[str, float] = field(default_factory=dict)
    verdict: str | None = None
    alert: bool = False

    @property
    def duration(self) -> float:
        if not self.packets:
            return 0.0
        ts = [p.t for p in self.packets]
        return max(ts) - min(ts)
