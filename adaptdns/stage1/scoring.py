"""Suspicion Scoring Engine (Stage 1, final step).

Converts the multi-scale statistical features of a flow into a single
suspicion score. The score is the Mahalanobis distance of the flow's feature
vector from the benign population (robust to feature correlations), fitted
only on benign calibration traffic. No DNS response is ever suppressed or
modified here — scoring is pure observation.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from ..datamodel import Flow
from .features import extract_stage1_features, feature_vector


class SuspicionScoringEngine:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.cov: LedoitWolf | None = None
        self.feature_names: list[str] = []

    def fit(self, benign_features: list[dict[str, float]]) -> "SuspicionScoringEngine":
        if not benign_features:
            raise ValueError("cannot fit suspicion engine on an empty benign set")
        self.feature_names = list(benign_features[0].keys())
        X = feature_vector(benign_features, self.feature_names)
        Xs = self.scaler.fit_transform(X)
        self.cov = LedoitWolf().fit(Xs)
        return self

    def score_features(self, features: dict[str, float]) -> float:
        if self.cov is None:
            raise RuntimeError("SuspicionScoringEngine not fitted")
        x = np.array([[features.get(n, 0.0) for n in self.feature_names]], dtype=float)
        xs = self.scaler.transform(x)
        d = self.cov.mahalanobis(xs)          # length-1 array
        return float(d[0])

    def score_flows(self, flows: list[Flow], cfg) -> list[float]:
        feats = [extract_stage1_features(f, cfg) for f in flows]
        for f, fd in zip(flows, feats):
            f.stage1_features = fd
        return [self.score_features(fd) for fd in feats]
