"""Selective Blackhole Probing Module (Stage 2, active).

Reuses the base paper's blackhole technique — deliberate suppression of a DNS
response — but ONLY for flows already escalated by Stage 1, and only where
AdaptDNS runs at an enterprise recursive resolver / DoH-DoT gateway that can
suppress an individual response. A bare on-path observer of encrypted traffic
cannot do this; the deployment assumption is stated in the case study (§1).

In the prototype the resolver side is simulated: `selective_blackhole_probe`
suppresses the first response of a chosen transaction and records the client's
observed retry behaviour (delays until each retry / give-up).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..datamodel import Flow


@dataclass
class RetryTrace:
    """Observed retry behaviour of one probed flow."""

    flow_id: str
    suppressed: bool                 # a response was actually suppressed
    delays: list[float] = field(default_factory=list)  # seconds per retry
    probed: bool = True

    @property
    def n_retries(self) -> int:
        return len(self.delays)


def selective_blackhole_probe(flow: Flow, cfg: Config, rng: np.random.Generator) -> RetryTrace:
    """Suppress one DNS response for an already-escalated flow and observe
    retry timing/frequency until give-up or `cfg.max_probe_retries`.

    Only escalated flows ever reach this function — the escalation gate in
    `adaptdns.escalation` is the pre-filtering stage the base paper lacks.
    """
    if not flow.escalated:
        # Defensive: never probe a flow that did not pass the gate.
        return RetryTrace(flow_id=flow.flow_id, suppressed=False, probed=False, delays=[])

    if flow.retry_delays is None:
        # Input without modelled client behaviour (e.g., pcap-only capture):
        # probing is attempted but no retry observations are available.
        return RetryTrace(flow_id=flow.flow_id, suppressed=True, probed=True, delays=[])

    delays = [float(d) for d in flow.retry_delays[: cfg.max_probe_retries]]
    return RetryTrace(flow_id=flow.flow_id, suppressed=True, probed=True, delays=delays)
