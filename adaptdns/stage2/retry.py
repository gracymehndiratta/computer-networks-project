"""Retry-Behaviour Distribution Analyzer (Stage 2, active).

Models retry timing and frequency as statistical *distributions* compared
against a benign baseline population, instead of the raw point ratios used
by the base paper. The intent is to make simple retry-timing randomisation
harder to exploit; robustness against a more capable adaptive adversary is
an explicit evaluation target of the project, not an assumed property.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import Config
from .blackhole import RetryTrace

_EPS = 1e-9


def _ks_statistic(sample: np.ndarray, reference: np.ndarray) -> float:
    """Two-sample-free KS distance: sup |ECDF_sample - ECDF_reference|."""
    if sample.size == 0 or reference.size == 0:
        return 0.0
    s = np.sort(sample)
    r = np.sort(reference)
    all_vals = np.concatenate([s, r])
    cdf_s = np.searchsorted(s, all_vals, side="right") / s.size
    cdf_r = np.searchsorted(r, all_vals, side="right") / r.size
    return float(np.max(np.abs(cdf_s - cdf_r)))


def _hist_entropy(sample: np.ndarray, bins: int, lo: float, hi: float) -> float:
    if sample.size == 0:
        return 0.0
    hist, _ = np.histogram(sample, bins=bins, range=(lo, hi))
    p = hist.astype(float) / sample.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class RetryBehaviorAnalyzer:
    """Fits the benign retry-delay population and extracts distributional
    features for an individual probed flow."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.baseline: np.ndarray = np.zeros(0)
        self._lo = 0.0
        self._hi = 1.0

    # --- baseline -------------------------------------------------------
    def fit_baseline(self, benign_traces: list[RetryTrace]) -> "RetryBehaviorAnalyzer":
        delays = [d for tr in benign_traces if tr.suppressed for d in tr.delays][: self.cfg.retry_history_size]
        self.baseline = np.asarray(delays, dtype=float)
        if self.baseline.size:
            self._lo = float(min(0.0, self.baseline.min()))
            self._hi = float(max(1.0, np.percentile(self.baseline, 99) * 1.5))
        return self

    # --- per-flow features ---------------------------------------------
    def extract(self, trace: RetryTrace, n_queries: int = 0) -> dict[str, float]:
        d = np.asarray(trace.delays, dtype=float)
        n = float(d.size)
        feats: dict[str, float] = {
            "n_retries": n,
            # retry frequency, for parity with the base paper's retry ratio
            "retry_rate": n / max(n_queries, 1),
            "retry_suppressed": 1.0 if trace.suppressed else 0.0,
        }

        if d.size == 0:
            feats.update({
                "retry_mean": 0.0, "retry_std": 0.0, "retry_cv": 0.0,
                "retry_skew": 0.0, "retry_min": 0.0, "retry_max": 0.0,
                "retry_diff_cv": 0.0, "retry_periodicity": 0.0,
                "retry_entropy": 0.0, "retry_ks_vs_benign": 0.0,
                "retry_first_delay": 0.0,
                "retry_growth": 0.0, "retry_trend": 0.0,
            })
            return feats

        mean = float(d.mean())
        std = float(d.std(ddof=0))
        feats["retry_mean"] = mean
        feats["retry_std"] = std
        feats["retry_cv"] = std / (mean + _EPS)
        feats["retry_min"] = float(d.min())
        feats["retry_max"] = float(d.max())
        feats["retry_first_delay"] = float(d[0])
        # skewness (population)
        feats["retry_skew"] = float(((d - mean) ** 3).mean() / (std ** 3 + _EPS)) if d.size >= 3 else 0.0

        # Successive-difference CV: a fixed-interval retry schedule has near-
        # zero diff variance; a randomised schedule does not.
        if d.size >= 2:
            diffs = np.diff(d)
            dm = float(diffs.mean())
            feats["retry_diff_cv"] = float(diffs.std(ddof=0) / (abs(dm) + _EPS))
            feats["retry_periodicity"] = 1.0 / (1.0 + feats["retry_diff_cv"])
            # --- order/shape statistics: these survive level-forging ------
            # Benign clients back off multiplicatively: gaps strictly grow.
            # An attacker can match count/first/mean/max (point statistics)
            # by emitting benign-distributed values in arbitrary order, but
            # then the growth trend disappears.
            pos = d[d > 0]
            ratios = d[1:] / np.maximum(d[:-1], _EPS)
            feats["retry_growth"] = float(np.median(ratios)) if ratios.size else 0.0
            feats["retry_trend"] = float((diffs > 0).mean())  # fraction increasing
            _ = pos
        else:
            feats["retry_diff_cv"] = 0.0
            feats["retry_periodicity"] = 0.0
            feats["retry_growth"] = 0.0
            feats["retry_trend"] = 0.0

        feats["retry_entropy"] = _hist_entropy(d, self.cfg.retry_bins, self._lo, self._hi)
        feats["retry_ks_vs_benign"] = _ks_statistic(d, self.baseline) if self.baseline.size else 0.0
        return feats

    @staticmethod
    def feature_names(sample: dict[str, float]) -> list[str]:
        return list(sample.keys())
