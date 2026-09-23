"""Multi-scale encrypted-flow statistical feature extractor (Stage 1).

Everything here is computed from connection-level metadata only — packet
sizes, directions and inter-arrival times — which is visible even when the
DNS messages themselves are encrypted inside DoH/DoT.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..datamodel import Flow

_EPS = 1e-9


def _cv(x: np.ndarray) -> float:
    """Coefficient of variation (std / mean), 0 for degenerate inputs."""
    if x.size < 2:
        return 0.0
    m = float(x.mean())
    if m <= _EPS:
        return 0.0
    return float(x.std(ddof=0) / m)


def _stats(prefix: str, x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_cv": 0.0}
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std(ddof=0)),
        f"{prefix}_cv": _cv(x),
    }


def extract_stage1_features(flow: Flow, cfg: Config) -> dict[str, float]:
    """Packet-size distribution, timing burstiness and inter-arrival variance
    at multiple time-window scales, purely from encrypted-flow metadata."""
    feats: dict[str, float] = {}
    pkts = sorted(flow.packets, key=lambda p: p.t)
    n = len(pkts)
    if n == 0:
        return {"n_packets": 0.0, "duration": 0.0}

    sizes = np.array([p.size for p in pkts], dtype=float)
    dirs = np.array([p.direction for p in pkts], dtype=float)
    t0, t1 = pkts[0].t, pkts[-1].t
    duration = max(t1 - t0, _EPS)

    if n >= 2:
        iats = np.diff(np.array([p.t for p in pkts], dtype=float))
        iats = np.clip(iats, 0.0, None)
    else:
        iats = np.zeros(0)

    # --- Overall flow-level statistics -----------------------------------
    send_bytes = float(sizes[dirs > 0].sum())
    recv_bytes = float(sizes[dirs < 0].sum())
    feats["n_packets"] = float(n)
    feats["duration"] = duration
    feats["pps"] = n / duration
    feats["bytes_total"] = float(sizes.sum())
    feats["bytes_per_sec"] = float(sizes.sum()) / duration
    feats.update(_stats("size", sizes))
    feats["size_p50"] = float(np.percentile(sizes, 50))
    feats["size_p90"] = float(np.percentile(sizes, 90))
    feats["size_max"] = float(sizes.max())
    feats["large_pkt_frac"] = float((sizes > 700).mean())
    feats["small_pkt_frac"] = float((sizes < 120).mean())
    feats["dir_ratio"] = send_bytes / (recv_bytes + _EPS)
    if iats.size:
        feats.update(_stats("iat", iats))
        feats["iat_cv"] = _cv(iats)
        # first-transmission burst: how front-loaded is the flow?
        feats["front_load"] = float((iats < iats.mean()).mean()) if iats.mean() > 0 else 0.0
    else:
        feats.update({"iat_mean": 0.0, "iat_std": 0.0, "iat_cv": 0.0, "front_load": 0.0})

    # --- Per-scale windowed burstiness ------------------------------------
    # Bin the flow into windows of length W and characterise how traffic
    # energy is distributed across windows at that granularity.
    for w in cfg.scales:
        tag = f"w{int(w)}s"
        edges = (np.array([p.t for p in pkts]) - t0) / w
        bin_idx = np.floor(edges).astype(int)
        n_bins = int(bin_idx.max()) + 1
        counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        byte_counts = np.bincount(bin_idx, weights=sizes, minlength=n_bins).astype(float)
        active = float((counts > 0).mean()) if n_bins else 0.0
        feats[f"{tag}_count_cv"] = _cv(counts)
        feats[f"{tag}_byte_cv"] = _cv(byte_counts)
        feats[f"{tag}_active_frac"] = active
        feats[f"{tag}_peak_frac"] = float(counts.max() / max(counts.sum(), _EPS))

    # Sanitise: no NaN/Inf may reach the model.
    for k, v in list(feats.items()):
        if not math.isfinite(v):
            feats[k] = 0.0
    return feats


def feature_vector(feature_dicts: list[dict[str, float]], names: list[str]) -> np.ndarray:
    """Stack feature dicts into a matrix honouring a fixed column order."""
    return np.array([[fd.get(name, 0.0) for name in names] for fd in feature_dicts], dtype=float)


def stage1_feature_names(sample: dict[str, float]) -> list[str]:
    return list(sample.keys())
