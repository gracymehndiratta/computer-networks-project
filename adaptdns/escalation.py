"""Escalation decision: suspicion threshold calibrated against a budget.

Low-suspicion flows are NOT escalated and stay under passive monitoring —
this reflects current suspicion, not a guarantee of benignity. Only flows
whose score exceeds the budget-calibrated threshold proceed to Stage 2,
which keeps active blackhole probing rare and targeted.
"""
from __future__ import annotations

import numpy as np

from .datamodel import Flow


def calibrate_threshold(benign_scores: list[float], budget: float) -> float:
    """Threshold such that at most `budget` of benign calibration traffic
    would be escalated (1 - budget empirical quantile)."""
    if not benign_scores:
        raise ValueError("no benign scores to calibrate against")
    if not 0.0 < budget < 1.0:
        raise ValueError("budget must be in (0, 1)")
    return float(np.quantile(np.asarray(benign_scores, dtype=float), 1.0 - budget))


def apply_escalation(
    flows: list[Flow],
    scores: list[float],
    threshold: float,
) -> list[Flow]:
    """Tag flows with their score and escalation flag; return escalated flows."""
    escalated: list[Flow] = []
    for flow, score in zip(flows, scores):
        flow.suspicion_score = score
        flow.escalated = score >= threshold
        if flow.escalated:
            escalated.append(flow)
    return escalated


def escalation_rate(flows: list[Flow]) -> float:
    if not flows:
        return 0.0
    return sum(1 for f in flows if f.escalated) / len(flows)
