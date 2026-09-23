"""Synthetic DoH/DoT flow generator.

Produces three populations of flows that share the same output interface a
real pcap ingestor would fill (`Flow.packets`, `Flow.txns`, `Flow.retry_delays`):

* **benign**        — page-load style bursts of ordinary queries to real-
                      looking domains; heavy-jitter retry backoff on failure.
* **naive_tunnel**  — chunked data exfiltration over DNS tunnel: fixed query
                      interval, long high-entropy subdomains, constant TTL,
                      strictly periodic retries.
* **adaptive_tunnel** — the adaptive adversary from the case study: same DNS
                      semantics, but deliberately *reshaped retry timing*
                      (heavy jitter, benign-like retry counts) and jittered
                      query timing to stress both stages.

Labels: benign -> 'benign'; both tunnel variants -> 'tunnel'
        (`Flow.variant` distinguishes naive vs adaptive for evaluation).
"""
from __future__ import annotations

import base64
import secrets

import numpy as np

from ..config import Config
from ..datamodel import (
    DnsTxn,
    Flow,
    LABEL_BENIGN,
    LABEL_TUNNEL,
    Packet,
    VARIANT_ADAPTIVE,
    VARIANT_BENIGN,
    VARIANT_NAIVE,
)

# --- vocabularies -----------------------------------------------------------
BENIGN_DOMAINS = [
    "google.com", "www.google.com", "youtube.com", "fonts.gstatic.com",
    "cdn.jsdelivr.net", "api.github.com", "twitter.com", "instagram.com",
    "reddit.com", "netflix.com", "microsoft.com", "apple.com",
    "amazon.com", "cloudflare.com", "wikipedia.org", "en.wikipedia.org",
    "spotify.com", "zoom.us", "slack.com", "notion.so",
]
BENIGN_SUBDOMAINS = ["", "", "", "www.", "api.", "img1.", "img2.", "assets.",
                     "static.", "a.b.", "mail.", "news."]
TUNNEL_DOMAINS = [
    "cdn-up.example-net.com", "stats.metrics-collect.example.org",
    "sync.mobile-data.example.io", "telemetry.edge-relay.example.net",
]
QTYPES = ["A", "AAAA", "HTTPS", "A", "A", "TXT"]

# DNS-over-HTTPS framing overhead (HTTP/2 + TLS record) added to the DNS
# message length to obtain an encrypted-flow packet size.
DOH_OVERHEAD = 62


def _doh_pkt(t: float, dns_len: int, direction: int) -> Packet:
    return Packet(t=t, size=DOH_OVERHEAD + dns_len, direction=direction)


def _tls_handshake(t0: float) -> list[Packet]:
    """TLS handshake + DoH connection setup (connection-level metadata only)."""
    return [
        Packet(t=t0, size=517, direction=1),                 # ClientHello
        Packet(t=t0 + 0.012, size=1454, direction=-1),       # ServerHello + cert
        Packet(t=t0 + 0.024, size=890, direction=-1),
        Packet(t=t0 + 0.036, size=61, direction=1),           # Finished / SETTINGS ack
    ]


def _qname_entropy_label(rng: np.random.Generator, n: int = 48) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz234567"
    idx = rng.integers(0, len(alphabet), size=n)
    return "".join(alphabet[i] for i in idx)


# --- retry schedules --------------------------------------------------------
def _benign_retry_schedule(rng: np.random.Generator) -> list[float]:
    """Few retries, exponential-ish backoff with heavy jitter (client app
    timeouts vary a lot); some clients give up before ever retrying."""
    n = int(rng.choice([0, 1, 1, 2, 2, 3]))
    gaps: list[float] = []
    gap = float(rng.lognormal(mean=np.log(0.35), sigma=0.6))
    for _ in range(n):
        gaps.append(max(0.05, gap))
        gap *= float(rng.uniform(1.8, 3.5))
    return gaps


def _naive_retry_schedule(rng: np.random.Generator, interval: float) -> list[float]:
    """Tunnel client: never gives up easily, retries at a fixed period."""
    n = int(rng.integers(6, 11))
    jitter = lambda: float(rng.normal(0.0, 0.015))  # noqa: E731
    return [max(0.05, interval + jitter()) for _ in range(n)]


def _adaptive_retry_schedule(rng: np.random.Generator, strength: float = 1.0) -> list[float]:
    """Adaptive adversary: forges the *levels* of benign retry timing while
    (at full strength) destroying its temporal structure.

    Benign clients back off multiplicatively — gap[i+1] = gap[i] * U(1.8,3.5),
    so the gap sequence strictly grows. The attacker at strength 1.0 matches
    every per-flow point statistic (retry count drawn from the benign
    distribution, first delay from the benign base-delay distribution,
    remaining gaps iid from the pooled benign gap population) but emits them
    UNSHUFFLED-IN-ORDER — i.e., no backoff trend. Point-ratio features
    (count / first / mean / max) see benign-matching values; order-sensitive
    distribution features still see the missing backoff structure.

    Partial strength blends this with the jittered-periodic schedule.
    """
    def benign_gaps() -> list[float]:
        return _benign_retry_schedule(rng)

    if rng.random() >= strength:
        # simple randomisation: periodic core + strength-scaled jitter
        interval = float(rng.uniform(0.9, 1.1))
        jitter_sd = 0.30 * strength
        n = int(rng.integers(6, 11))
        return [max(0.05, interval + float(rng.normal(0.0, jitter_sd)))
                for _ in range(n)]

    # Level-matching forgery: benign-like count, benign first delay,
    # remaining gaps iid from the benign pooled population (order lost).
    # The count MUST come from the same distribution as benign clients —
    # otherwise retries/queries alone leaks the label and the
    # point-ratio-vs-distribution comparison is confounded.
    n = int(rng.choice([0, 1, 1, 2, 2, 3]))
    if n == 0:
        return []
    first = max(0.05, float(rng.lognormal(mean=np.log(0.35), sigma=0.6)))
    pooled: list[float] = []
    for _ in range(6):
        pooled.extend(benign_gaps())
    if not pooled:
        pooled = [first]
    gaps = [first]
    for _ in range(n - 1):
        gaps.append(float(rng.choice(pooled)))
    return gaps


# --- flow builders ----------------------------------------------------------
def _benign_flow(rng: np.random.Generator, flow_id: str) -> Flow:
    t = float(rng.uniform(0, 1_000))
    t0 = t
    packets = _tls_handshake(t)
    txns: list[DnsTxn] = []

    # DoH clients reuse one HTTP/2 connection for many queries (the case
    # study notes flows multiplex several transactions), so benign flows
    # carry 30-120+ queries — comparable to tunnel flows. This prevents
    # any feature from separating classes purely by flow length.
    n_bursts = int(rng.integers(2, 9))
    for _ in range(n_bursts):
        t += float(rng.exponential(4.0)) + 0.05
        n_q = int(rng.integers(4, 16))
        for _ in range(n_q):
            t += float(rng.exponential(0.08)) + 0.004
            domain = str(rng.choice(BENIGN_DOMAINS))
            qname = str(rng.choice(BENIGN_SUBDOMAINS)) + domain
            qtype = str(rng.choice(QTYPES))
            q_len = len(qname) + 30
            r_len = int(rng.integers(60, 1400))
            txns.append(
                DnsTxn(
                    t=t,
                    qname=qname,
                    qtype=qtype,
                    ttl=int(rng.integers(60, 3000)),
                    resp_size=r_len,
                    nxdomain=bool(rng.random() < 0.02),
                )
            )
            packets.append(_doh_pkt(t, q_len, 1))
            packets.append(_doh_pkt(t + float(rng.uniform(0.004, 0.05)), r_len, -1))

    return Flow(
        flow_id=flow_id,
        src=f"10.0.0.{int(rng.integers(2, 250))}",
        dst="doh.enterprise-gw.local:443",
        proto="doh/443",
        packets=packets,
        txns=txns,
        retry_delays=_benign_retry_schedule(rng),
        label=LABEL_BENIGN,
        variant=VARIANT_BENIGN,
        resolver_visible=True,
    )


def _tunnel_flow(rng: np.random.Generator, flow_id: str, adaptive: bool,
                 strength: float = 1.0) -> Flow:
    """Build one tunneling flow.

    `strength` ∈ [0, 1] controls the red-team's evasion effort (adaptive
    flows only): query pacing, retry reshaping, packet-size padding and
    DNS-semantic mimicry (shallower names, varied TTL/qtype) are all
    blended toward the benign distribution as strength rises.
    """
    s = float(np.clip(strength, 0.0, 1.0)) if adaptive else 0.0
    t = float(rng.uniform(0, 1_000))
    packets = _tls_handshake(t)
    txns: list[DnsTxn] = []

    domain = str(rng.choice(TUNNEL_DOMAINS))
    base_ttl = int(rng.choice([1, 60, 300]))
    interval = float(rng.uniform(0.9, 1.1))
    if adaptive:
        # Match the benign flow-scope distribution THROUGH THE SAME BURST
        # PROCESS: the escalation gate selects atypical benign flows, so
        # drawing n_q exactly the way benign clients do is the only way for
        # `retries/queries` (rate) to stop leaking the label after gating.
        # A real adaptive client splits payload across benign-length flows.
        n_bursts = int(rng.integers(2, 9))
        n_q = sum(int(rng.integers(4, 16)) for _ in range(n_bursts))
    else:
        n_q = int(rng.integers(40, 101))
    payload_chunk = bytes(secrets.token_bytes(48))
    queries_in_burst = int(rng.integers(4, 16))

    for k in range(n_q):
        # --- query pacing: fixed period (s=0) blended to the REAL benign
        # burst process (s=1): between-burst gap ~ Exp(4s), within-burst
        # gap ~ Exp(80ms) — the same parameters `_benign_flow` uses.
        fixed_gap = float(rng.normal(interval, 0.02))
        if k % queries_in_burst == 0:
            benign_gap = float(rng.exponential(4.0)) + 0.05
        else:
            benign_gap = float(rng.exponential(0.08)) + 0.004
        t += (1.0 - s) * fixed_gap + s * benign_gap

        # --- query-name semantics blended toward benign ------------------
        chunk = payload_chunk[:(k % 3 + 1) * 12 % len(payload_chunk) or 12]
        label_len = int((1.0 - s) * 40 + s * 6)
        label = _qname_entropy_label(rng, label_len) + base64.b32encode(chunk).decode().lower()
        if rng.random() < s:               # shallower name at high strength
            qname = f"{label[:12]}.{domain}"
        else:
            qname = f"{label}.c{k:05d}.data.{domain}"
        qtype = "TXT" if rng.random() > s else str(rng.choice(QTYPES))

        # --- packet sizes padded toward the BENIGN distribution -----------
        # NOTE: q_len/r_len are DNS-message lengths BEFORE the 62B DoH
        # overhead (_doh_pkt adds it). Benign query messages are
        # len(qname)+30 ≈ 40-58B and responses span 60-1400B; padding to
        # any other range would make the attacker trivially separable in
        # Stage 1 (this exact off-by-overhead bug inflated queries before).
        q_len = min(253, len(qname)) + 30
        r_len = int(rng.integers(78, 132))
        if s > 0:
            q_len = int((1.0 - s) * q_len + s * rng.uniform(40, 58))
            r_len = int((1.0 - s) * r_len + s * rng.uniform(60, 1400))

        txns.append(
            DnsTxn(
                t=t,
                qname=qname,
                qtype=qtype,
                ttl=base_ttl if rng.random() > s else int(rng.integers(60, 3000)),
                resp_size=r_len,
                nxdomain=False,
            )
        )
        packets.append(_doh_pkt(t, q_len, 1))
        packets.append(_doh_pkt(t + float(rng.uniform(0.004, 0.02)), r_len, -1))

    # --- retry schedule: periodic (s=0) vs level-forging (s=1) -----------
    if not adaptive:
        retry_delays = _naive_retry_schedule(rng, interval)
    else:
        retry_delays = _adaptive_retry_schedule(rng, strength=s)

    return Flow(
        flow_id=flow_id,
        src=f"10.0.0.{int(rng.integers(2, 250))}",
        dst="doh.enterprise-gw.local:443",
        proto="doh/443",
        packets=packets,
        txns=txns,
        retry_delays=retry_delays,
        label=LABEL_TUNNEL,
        variant=VARIANT_ADAPTIVE if adaptive else VARIANT_NAIVE,
        resolver_visible=True,
    )


def generate_population(cfg: Config, seed: int | None = None) -> list[Flow]:
    """Generate the full labelled flow population (benign + both tunnel variants)."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    flows: list[Flow] = []
    for i in range(cfg.n_benign):
        flows.append(_benign_flow(rng, f"benign-{i:05d}"))
    for i in range(cfg.n_tunnel):
        flows.append(_tunnel_flow(rng, f"tunnel-{i:05d}", adaptive=False))
    for i in range(cfg.n_adaptive):
        flows.append(_tunnel_flow(rng, f"adaptive-{i:05d}", adaptive=True,
                                  strength=cfg.attack_strength))
    rng.shuffle(flows)
    return flows
