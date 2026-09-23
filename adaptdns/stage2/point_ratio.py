"""Base-paper-style POINT-RATIO blackhole features.

The base paper [1] reduces blackhole observations to scalar point ratios and
threshold indicators per flow (retry counts, retry-to-query ratio, first-retry
delay, "is the schedule fixed-interval?" as a hard bit).  AdaptDNS instead
keeps the full retry *distribution* and compares it against a benign
reference population (`adaptdns.stage2.retry`).

This module exists so the two formulations can be benchmarked head-to-head
under the adaptive adversary: point ratios are compact but collapse when the
attacker reshapes timing; distributional features degrade more gracefully.
"""
from __future__ import annotations

from ..datamodel import Flow
from .blackhole import RetryTrace

POINT_RATIO_NAMES = [
    "pr_retry_count",
    "pr_retry_ratio",
    "pr_first_delay",
    "pr_mean_delay",
    "pr_max_delay",
    "pr_fixed_interval",
]


class PointRatioAnalyzer:
    """Extracts the base-paper point-ratio feature set from one probe trace."""

    @staticmethod
    def extract(trace: RetryTrace, flow: Flow) -> dict[str, float]:
        d = list(trace.delays)
        n = len(d)
        n_queries = max(len(flow.txns), 1)
        mean = sum(d) / n if n else 0.0
        # Hard threshold indicator: "regular" schedule when jitter < 50 ms.
        if n >= 2:
            var = sum((x - mean) ** 2 for x in d) / n
            fixed = 1.0 if var ** 0.5 < 0.05 else 0.0
        else:
            fixed = 0.0
        return {
            "pr_retry_count": float(n),
            "pr_retry_ratio": float(n) / n_queries,
            "pr_first_delay": float(d[0]) if n else 0.0,
            "pr_mean_delay": float(mean),
            "pr_max_delay": float(max(d)) if n else 0.0,
            "pr_fixed_interval": fixed,
        }

    @staticmethod
    def feature_names() -> list[str]:
        return list(POINT_RATIO_NAMES)
