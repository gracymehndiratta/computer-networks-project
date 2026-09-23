"""DoH/DoT flow identification & session reconstruction (Stage 1, step 1).

Groups connection-level packets into flows. Only connection metadata is used:
the 5-tuple (and, where available, the resolver endpoint being talked to),
plus an idle timeout. Because DoT/DoH multiplex several DNS transactions per
TLS/HTTP connection, a flow is a *flow-level grouping* — an exact
one-query-per-session reconstruction is explicitly out of scope here.
"""
from __future__ import annotations

from ..datamodel import DnsTxn, Flow, Packet

DEFAULT_IDLE_TIMEOUT = 30.0  # seconds without a packet closes a flow


def _key(p: tuple) -> tuple:
    src, dst, proto = p
    return (src, dst, proto)


def reconstruct_flows(
    packets: list[tuple[str, str, str, float, int, int]],
    resolver_endpoints: set[str] | None = None,
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
) -> list[Flow]:
    """Group raw packets into flows.

    Each packet tuple is (src, dst, proto, timestamp, size, direction).
    If `resolver_endpoints` is given, only packets whose destination or
    source matches a known enterprise resolver / DoH-DoT gateway endpoint
    are kept — this mirrors "identifies sessions to known resolver
    endpoints from connection-level metadata".
    """
    buckets: dict[tuple, list[tuple[float, int, int]]] = {}
    for src, dst, proto, t, size, direction in packets:
        if resolver_endpoints is not None:
            if src not in resolver_endpoints and dst not in resolver_endpoints:
                continue
        buckets.setdefault((src, dst, proto), []).append((t, size, direction))

    flows: list[Flow] = []
    for i, (key, recs) in enumerate(sorted(buckets.items())):
        recs.sort(key=lambda r: r[0])
        src, dst, proto = key
        # Split on idle gaps (new TLS connection after a long pause).
        segments: list[list[tuple[float, int, int]]] = [[recs[0]]]
        for prev, cur in zip(recs, recs[1:]):
            if cur[0] - prev[0] > idle_timeout:
                segments.append([cur])
            else:
                segments[-1].append(cur)
        for j, seg in enumerate(segments):
            flows.append(
                Flow(
                    flow_id=f"flow-{i:04d}-{j}",
                    src=src,
                    dst=dst,
                    proto=proto,
                    packets=[Packet(t=t, size=size, direction=d) for t, size, d in seg],
                )
            )
    return flows


def identify_doh_dot_sessions(
    flows: list[Flow],
    doh_ports: set[int] | None = None,
    dot_ports: set[int] | None = None,
) -> list[Flow]:
    """Keep only flows that are sessions to known DoH/DoT resolver endpoints.

    In this prototype the transport/port metadata is already part of the flow
    `proto` field (e.g. ``doh/443`` or ``dot/853``); anything else is dropped.
    """
    doh_ports = doh_ports or {443}
    dot_ports = dot_ports or {853}
    keep: list[Flow] = []
    for f in flows:
        proto = f.proto.lower()
        if proto.startswith("doh") or proto.startswith("dot"):
            keep.append(f)
    return keep


def attach_transactions(flow: Flow, txns: list[DnsTxn]) -> None:
    """Attach resolver-side DNS transactions to a flow (gateway visibility)."""
    flow.txns = sorted(txns, key=lambda x: x.t)
