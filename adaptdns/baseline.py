"""Baseline Update (feedback loop).

Two strictly separate paths, per the case study (§3.4):

* Analyst-confirmed **benign** traffic updates the passive layer's
  normal-behaviour baseline (Stage-1 features + retry population).
* Confirmed **malicious** detections update the classifier/signature store
  through a separate path and are NEVER fed into the benign baseline, so the
  baseline cannot be contaminated by attack traffic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .datamodel import Flow, LABEL_BENIGN, LABEL_TUNNEL
from .stage2.blackhole import RetryTrace


@dataclass
class SignatureStore:
    """Sink for confirmed-malicious updates (signatures / model refresh)."""

    entries: list[dict] = field(default_factory=list)

    def add(self, flow: Flow, reason: str = "analyst-confirmed") -> None:
        self.entries.append(
            {
                "flow_id": flow.flow_id,
                "label": flow.label,
                "variant": flow.variant,
                "reason": reason,
                "n_packets": len(flow.packets),
            }
        )


class BaselineManager:
    """Owns the benign baseline used by Stage 1 (scorer refit) and Stage 2
    (retry-distribution reference population)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.benign_stage1_features: list[dict[str, float]] = []
        self.benign_retry_traces: list[RetryTrace] = []
        self.signatures = SignatureStore()

    # --- benign path (the only path that touches the baseline) ----------
    def confirm_benign(self, stage1_feats: dict[str, float], trace: RetryTrace | None = None) -> None:
        self.benign_stage1_features.append(dict(stage1_feats))
        if trace is not None and trace.suppressed and trace.delays:
            self.benign_retry_traces.append(trace)
        # Cap memory for long-running deployments.
        cap = self.cfg.retry_history_size * 4
        if len(self.benign_stage1_features) > cap:
            self.benign_stage1_features = self.benign_stage1_features[-cap:]
        if len(self.benign_retry_traces) > cap:
            self.benign_retry_traces = self.benign_retry_traces[-cap:]

    # --- malicious path (separate; never touches the baseline) ----------
    def confirm_malicious(self, flow: Flow) -> None:
        self.signatures.add(flow)

    # --- reporting -------------------------------------------------------
    @property
    def benign_count(self) -> int:
        return len(self.benign_stage1_features)

    @property
    def signature_count(self) -> int:
        return len(self.signatures.entries)
