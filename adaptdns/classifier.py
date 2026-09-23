"""Random Forest Classification Engine + verdict/alerting.

Trained on the fused, encryption-aware feature set of escalated flows;
outputs a benign / tunneling label per the classifier family validated in
the base paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import Config
from .datamodel import LABEL_BENIGN, LABEL_TUNNEL, Flow


@dataclass
class Verdict:
    flow_id: str
    label: str            # predicted: benign | tunnel
    probability: float    # P(tunnel)
    alert: bool           # tunnel => logged + raised for SOC review


class TunnelingClassifier:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.rf_max_depth,
            class_weight=cfg.class_weight,
            random_state=cfg.seed,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: list[str], feature_names: list[str]) -> "TunnelingClassifier":
        self.feature_names = list(feature_names)
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba_tunnel(self, X: np.ndarray) -> np.ndarray:
        idx = list(self.model.classes_).index(LABEL_TUNNEL)
        return self.model.predict_proba(X)[:, idx]

    def verdicts(self, flows: list[Flow], X: np.ndarray) -> list[Verdict]:
        """Apply the model and attach verdicts/alerts to the flows.

        Flows classified benign after probing 'resume normal processing'
        (no alert); tunneling flows are logged and raised for SOC review.
        """
        probs = self.predict_proba_tunnel(X)
        preds = self.predict(X)
        out: list[Verdict] = []
        for flow, pred, p in zip(flows, preds, probs):
            verdict = Verdict(
                flow_id=flow.flow_id,
                label=str(pred),
                probability=float(p),
                alert=bool(pred == LABEL_TUNNEL),
            )
            flow.verdict = verdict.label
            flow.alert = verdict.alert
            out.append(verdict)
        return out

    def feature_importances(self) -> dict[str, float]:
        return dict(
            sorted(
                zip(self.feature_names, self.model.feature_importances_),
                key=lambda kv: -kv[1],
            )
        )
