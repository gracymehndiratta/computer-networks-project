"""Hybrid Feature Fusion (Stage 2 → classification).

Merges the Stage-1 passive statistical features with the Stage-2 active
retry-distribution and resolver-side DNS features into one combined feature
vector per escalated flow.

Because only escalated flows reach this stage, the classifier must be trained
and evaluated on data produced through the same Stage-1 gating process to
avoid selection bias between stages — `adaptdns.pipeline` enforces this.
"""
from __future__ import annotations

import math

import numpy as np

from .datamodel import Flow


def fuse_features(
    stage1: dict[str, float],
    retry_feats: dict[str, float],
    dns_feats: dict[str, float],
) -> dict[str, float]:
    merged = {}
    merged.update({f"s1_{k}": v for k, v in stage1.items()})
    merged.update({f"retry_{k}" if not k.startswith("retry_") else k: v for k, v in retry_feats.items()})
    merged.update({f"dns_{k}" if not k.startswith("dns_") else k: v for k, v in dns_feats.items()})
    # Sanitise: no NaN/Inf may reach the classifier.
    return {k: (float(v) if math.isfinite(float(v)) else 0.0) for k, v in merged.items()}


def fuse_flow(flow: Flow) -> dict[str, float]:
    """Convenience: fuse a flow's already-extracted Stage-1/Stage-2 features."""
    return fuse_features(flow.stage1_features, flow.stage2_features, {})


def feature_matrix(feature_dicts: list[dict[str, float]], names: list[str]) -> np.ndarray:
    return np.array([[fd.get(n, 0.0) for n in names] for fd in feature_dicts], dtype=float)


def aligned_names(feature_dicts: list[dict[str, float]]) -> list[str]:
    """Stable column order: union of keys, ordered by first appearance."""
    names: list[str] = []
    seen: set[str] = set()
    for fd in feature_dicts:
        for k in fd:
            if k not in seen:
                seen.add(k)
                names.append(k)
    return names
