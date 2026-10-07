"""Classification engine (Random Forest or XGBoost) + verdict/alerting.

Trained on the fused, encryption-aware feature set of escalated flows;
outputs a benign / tunneling label per the classifier family validated in
the base paper.  The model backend is configurable via ``Config.classifier_kind``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:  # pragma: no cover - optional heavy dependency
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    _XGBOOST_AVAILABLE = False

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
        self.model: Any = self._build_model(cfg)
        self.feature_names: list[str] = []
        self._label_encoder: LabelEncoder | None = None
        self._tunnel_proba_idx: int | None = None

    @staticmethod
    def _build_model(cfg: Config) -> Any:
        kind = cfg.classifier_kind.lower()
        if kind == "rf":
            return RandomForestClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.rf_max_depth,
                class_weight=cfg.class_weight,
                random_state=cfg.seed,
                n_jobs=-1,
            )
        if kind == "xgboost":
            if not _XGBOOST_AVAILABLE:
                raise ImportError(
                    "xgboost is not installed. Install it with: pip install xgboost"
                )
            # scale_pos_weight fallback: balance classes from the synthetic mix
            # (n_benign vs n_tunnel + n_adaptive). None keeps the model default.
            scale_pos_weight = cfg.xgboost_scale_pos_weight
            if scale_pos_weight is None:
                n_pos = cfg.n_tunnel + cfg.n_adaptive
                n_neg = cfg.n_benign
                if n_pos > 0:
                    scale_pos_weight = n_neg / n_pos
            return XGBClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.xgboost_max_depth,
                learning_rate=cfg.xgboost_learning_rate,
                subsample=cfg.xgboost_subsample,
                colsample_bytree=cfg.xgboost_colsample_bytree,
                min_child_weight=cfg.xgboost_min_child_weight,
                scale_pos_weight=scale_pos_weight,
                eval_metric=cfg.xgboost_eval_metric,
                random_state=cfg.seed,
                n_jobs=-1,
            )
        raise ValueError(f"unknown classifier_kind: {cfg.classifier_kind!r}")

    def fit(self, X: np.ndarray, y: list[str], feature_names: list[str]) -> "TunnelingClassifier":
        self.feature_names = list(feature_names)
        y_arr = np.asarray(y)

        if self.cfg.classifier_kind.lower() == "xgboost":
            self._label_encoder = LabelEncoder()
            y_enc = self._label_encoder.fit_transform(y_arr)
            self.model.fit(X, y_enc)
            # XGBoost's classes_ are encoded integers; remember the mapping.
            self._tunnel_proba_idx = list(self._label_encoder.classes_).index(LABEL_TUNNEL)
        else:
            self.model.fit(X, y_arr)
            self._tunnel_proba_idx = list(self.model.classes_).index(LABEL_TUNNEL)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        if self._label_encoder is not None:
            return self._label_encoder.inverse_transform(preds.astype(int))
        return preds

    def predict_proba_tunnel(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, self._tunnel_proba_idx]

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
