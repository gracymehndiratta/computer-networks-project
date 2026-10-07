"""Central configuration for the AdaptDNS prototype."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Stage 1: multi-scale encrypted-flow feature extraction -----------
    # Time-window granularities (seconds) for burstiness / aggregation stats.
    scales: tuple[float, ...] = (1.0, 10.0, 60.0)

    # --- Escalation decision --------------------------------------------
    # Fraction of flows the escalation budget allows to reach Stage 2.
    # The suspicion threshold is calibrated as the (1 - budget) quantile of
    # the score distribution of a calibration set of benign flows.
    escalation_budget: float = 0.05

    # --- Stage 2: selective blackhole probing ----------------------------
    # Maximum retries observed per probe (the resolver/gateway watches the
    # flagged flow until this many retries or client give-up).
    max_probe_retries: int = 12

    # --- Retry-behaviour distribution analyzer ---------------------------
    # Number of histogram bins over retry delay values used for the
    # distributional (entropy / modality) features.
    retry_bins: int = 8
    retry_history_size: int = 500  # benign retry samples kept in the baseline

    # --- Classification engine -------------------------------------------
    # "rf"      -> scikit-learn RandomForestClassifier (default, paper baseline)
    # "xgboost" -> XGBoost gradient-boosted trees
    classifier_kind: str = "rf"
    n_estimators: int = 300
    rf_max_depth: int | None = None
    class_weight: str = "balanced"

    # XGBoost-specific hyperparameters (used only when classifier_kind="xgboost")
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_subsample: float = 0.8
    xgboost_colsample_bytree: float = 0.8
    xgboost_min_child_weight: int = 1
    xgboost_scale_pos_weight: float | None = None  # None -> let the model learn
    xgboost_eval_metric: str = "logloss"

    # --- Synthetic data generation ---------------------------------------
    n_benign: int = 1200
    n_tunnel: int = 300          # naive (fixed-interval) tunneling flows
    n_adaptive: int = 300        # adaptive tunneling: reshaped retry timing
    calibration_frac: float = 0.20  # benign split used to fit scorer/threshold

    seed: int = 42

    # --- Red-team attack strength -----------------------------------------
    # Strength of the adaptive tunneling adversary, in [0, 1]:
    #   0.0 = naive tunnel (fixed query period, periodic retries, tunnel-sized
    #            packets, deep high-entropy qnames)
    #   1.0 = full evasion attempt: benign-like query pacing, reshaped retry
    #            timing, request/response sizes padded toward the benign
    #            distribution, shallower query names and varied TTLs/qtypes.
    # Intermediate values interpolate; used by the robustness sweep
    # (experiments/study.py robustness).
    attack_strength: float = 1.0

    # --- Experiment artifacts --------------------------------------------
    artifacts_dir: str = "artifacts"

    extra: dict = field(default_factory=dict)


DEFAULT = Config()
