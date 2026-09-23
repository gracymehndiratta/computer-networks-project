"""PCAP ingestion — the seam for real captures (future work).

Same output interface as the synthetic generator: a list of `Flow` objects
with encrypted-flow packet metadata. Behaviour by capture type:

* **Plaintext DNS (UDP/53)**      — transactions parsed from the DNS layer;
  fills `Flow.txns`, so Stage-2 resolver-side features work.
* **DoT/DoH (TLS)**              — connection metadata only (packet sizes,
  timings, 5-tuple), exactly what a bare on-path observer sees. `txns` stays
  empty, which the feature extractors handle as zeros — matching the case
  study's deployment discussion: Stage 2 needs resolver-side visibility.
* `retry_delays` is left `None` (client behaviour not observable), so
  blackhole probing yields an empty retry trace.

Requires scapy: `pip install scapy` (kept optional in requirements.txt).
"""
from __future__ import annotations

from collections import defaultdict

from ..datamodel import DnsTxn, Flow, Packet

RESOLVER_PORT = 53
DOT_PORT = 853
DOH_PORT = 443
IDLE_TIMEOUT = 30.0


def load_flows_from_pcap(path: str, resolver_endpoints: set[str] | None = None) -> list[Flow]:
    try:
        from scapy.all import DNS, DNSQR, IP, IPv6, TCP, UDP, rdpcap  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pcap ingestion requires scapy: pip install scapy"
        ) from exc

    packets = rdpcap(path)
    groups: dict[tuple, list[tuple[float, int, int]]] = defaultdict(list)
    dns_msgs: dict[tuple, list[DnsTxn]] = defaultdict(list)

    for pkt in packets:
        if IP not in pkt and IPv6 not in pkt:
            continue
        src = pkt[IP].src if IP in pkt else pkt[IPv6].src
        dst = pkt[IP].dst if IP in pkt else pkt[IPv6].dst
        t = float(pkt.time)
        size = len(pkt)

        if UDP in pkt and (pkt[UDP].sport == RESOLVER_PORT or pkt[UDP].dport == RESOLVER_PORT):
            proto = "dns/udp"
            direction = 1 if pkt[UDP].dport == RESOLVER_PORT else -1
            if DNS in pkt and pkt[DNS].qr == 0:  # query
                q = pkt[DNS].qd
                if q is not None:
                    dns_msgs[(src, dst, proto)].append(
                        DnsTxn(
                            t=t,
                            qname=q.qname.decode(errors="replace") if isinstance(q.qname, bytes) else str(q.qname),
                            qtype=str(q.qtype),
                            ttl=0,
                            resp_size=0,
                        )
                    )
            elif DNS in pkt and pkt[DNS].qr == 1:  # response: enrich last query
                key = (dst, src, proto)
                if dns_msgs.get(key):
                    ans = pkt[DNS].an
                    ttl = int(ans.ttl) if ans is not None and getattr(ans, "ttl", None) else 0
                    dns_msgs[key][-1].ttl = ttl
                    dns_msgs[key][-1].resp_size = size
                    dns_msgs[key][-1].nxdomain = pkt[DNS].rcode == 3
        elif TCP in pkt:
            if pkt[TCP].dport == DOT_PORT:
                proto, direction = "dot/853", 1
            elif pkt[TCP].sport == DOT_PORT:
                proto, direction = "dot/853", -1
            elif pkt[TCP].dport == DOH_PORT:
                proto, direction = "doh/443", 1
            elif pkt[TCP].sport == DOH_PORT:
                proto, direction = "doh/443", -1
            else:
                continue
        else:
            continue

        if resolver_endpoints is not None and dst not in resolver_endpoints and src not in resolver_endpoints:
            continue
        groups[(src, dst, proto)].append((t, size, direction))

    flows: list[Flow] = []
    for i, ((src, dst, proto), recs) in enumerate(sorted(groups.items())):
        recs.sort(key=lambda r: r[0])
        segs: list[list[tuple[float, int, int]]] = [recs]
        for prev, cur in zip(recs, recs[1:]):
            if cur[0] - prev[0] > IDLE_TIMEOUT:
                segs.append([cur])
            else:
                segs[-1].append(cur)
        for j, seg in enumerate(segs):
            f = Flow(
                flow_id=f"pcap-{i:04d}-{j}",
                src=src,
                dst=dst,
                proto=proto,
                packets=[Packet(t=t, size=s, direction=d) for t, s, d in seg],
                retry_delays=None,
                resolver_visible=proto.startswith("dns"),
            )
            if proto == "dns/udp":
                f.txns = sorted(dns_msgs.get((src, dst, proto), []), key=lambda x: x.t)
            flows.append(f)
    return flows
