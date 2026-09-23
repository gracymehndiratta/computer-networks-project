"""End-to-end AdaptDNS pipeline: Stage 1 → escalation → Stage 2 → fusion →
Random Forest → verdict, plus the analyst-feedback baseline update.

Selection-bias control: the classifier is trained and evaluated ONLY on flows
produced through the same Stage-1 gating process it will face at inference
time (i.e., on the escalated subset), as required by the case study §3.4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baseline import BaselineManager
from .classifier import TunnelingClassifier, Verdict
from .config import Config
from .datamodel import Flow, LABEL_BENIGN, LABEL_TUNNEL, VARIANT_ADAPTIVE, VARIANT_NAIVE
from .escalation import apply_escalation, calibrate_threshold, escalation_rate
from .fusion import aligned_names, feature_matrix, fuse_features
from .stage1.features import extract_stage1_features
from .stage1.scoring import SuspicionScoringEngine
from .stage2.blackhole import RetryTrace, selective_blackhole_probe
from .stage2.dns_features import extract_traditional_dns_features
from .stage2.point_ratio import PointRatioAnalyzer
from .stage2.retry import RetryBehaviorAnalyzer
from .synthetic.generator import generate_population


@dataclass
class ExperimentResult:
    metrics: dict[str, float] = field(default_factory=dict)
    classifier: TunnelingClassifier | None = None
    scorer: SuspicionScoringEngine | None = None
    baseline: BaselineManager | None = None
    threshold: float = 0.0
    test_flows: list[Flow] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    # Stage-2 feature groups per escalated flow (for analysis/diagnostics):
    train_groups: list[dict[str, dict[str, float]]] = field(default_factory=list)
    test_groups: list[dict[str, dict[str, float]]] = field(default_factory=list)
    # Fitted Stage-2 pieces, so live/streaming callers can score new flows:
    retry_analyzer: "RetryBehaviorAnalyzer | None" = None


def _split_benign(benign: list[Flow], frac: float, rng: np.random.Generator):
    idx = rng.permutation(len(benign))
    n_cal = int(len(benign) * frac)
    cal_idx, rest = idx[:n_cal], idx[n_cal:]
    half = len(rest) // 2
    return [benign[i] for i in cal_idx], [benign[i] for i in rest[:half]], [benign[i] for i in rest[half:]]


def _stage2_features(flow: Flow, cfg: Config, retry_analyzer: RetryBehaviorAnalyzer,
                     rng: np.random.Generator) -> tuple[dict[str, dict[str, float]], RetryTrace]:
    """Stage 2 runs in parallel on an escalated flow: blackhole probing →
    retry-distribution features, alongside resolver-side DNS features.

    Returns the three feature groups (passive / retry / dns) plus the probe
    trace so the pipeline can build both the fused vector and ablations.
    """
    trace = selective_blackhole_probe(flow, cfg, rng)
    groups = {
        "passive": dict(flow.stage1_features),
        "retry": retry_analyzer.extract(trace, n_queries=len(flow.txns)),
        "dns": extract_traditional_dns_features(flow),
        # Base-paper-style scalar point ratios from the SAME probe trace —
        # enables the head-to-head point-ratio vs distribution comparison.
        "point_ratio": PointRatioAnalyzer.extract(trace, flow),
    }
    return groups, trace


def run_experiment(cfg: Config) -> ExperimentResult:
    rng = np.random.default_rng(cfg.seed)
    result = ExperimentResult()

    # ---- generate population -------------------------------------------
    flows = generate_population(cfg)
    benign = [f for f in flows if f.label == LABEL_BENIGN]
    malicious = [f for f in flows if f.label == LABEL_TUNNEL]

    cal_benign, train_benign, test_benign = _split_benign(benign, cfg.calibration_frac, rng)
    rng2 = np.random.default_rng(cfg.seed + 1)
    perm = rng2.permutation(len(malicious))
    half = len(perm) // 2
    train_mal = [malicious[i] for i in perm[:half]]
    test_mal = [malicious[i] for i in perm[half:]]

    # ---- Stage 1: passive profiling (always-on, all flows) --------------
    for f in flows:
        f.stage1_features = extract_stage1_features(f, cfg)

    # The scorer is fit on the calibration benign split ONLY; the escalation
    # threshold is then calibrated on DISJOINT benign samples (train_benign).
    # Calibrating on the scorer's own fit set would use in-sample Mahalanobis
    # distances — biased low — and silently overshoot the budget at test time.
    scorer = SuspicionScoringEngine().fit([f.stage1_features for f in cal_benign])
    threshold_pool = [scorer.score_features(f.stage1_features) for f in train_benign]
    threshold = calibrate_threshold(threshold_pool, cfg.escalation_budget)

    # Score + gate the train/test populations (calibration flows never reach
    # training or evaluation).
    eval_flows = train_benign + train_mal + test_benign + test_mal
    scores = [scorer.score_features(f.stage1_features) for f in eval_flows]
    apply_escalation(eval_flows, scores, threshold)

    # ---- escalation-budget accounting ------------------------------------
    # Budget is calibrated on benign traffic; verify it on held-out benign test.
    result.metrics["threshold"] = threshold
    result.metrics["escalation_rate_benign_test"] = escalation_rate(test_benign)
    result.metrics["escalation_rate_benign_train"] = escalation_rate(train_benign)

    # Escalation recall: fraction of tunneling flows sent to Stage 2.
    esc_train_mal = [f for f in train_mal if f.escalated]
    esc_test_mal = [f for f in test_mal if f.escalated]
    esc_train_ben = [f for f in train_benign if f.escalated]
    esc_test_ben = [f for f in test_benign if f.escalated]
    result.metrics["escalation_recall_train"] = len(esc_train_mal) / max(len(train_mal), 1)
    result.metrics["escalation_recall_test"] = len(esc_test_mal) / max(len(test_mal), 1)
    # Per-variant gate recall (how much of each adversary the BUDGET GATE
    # alone catches — the first line of defence before any probing).
    for variant in (VARIANT_NAIVE, VARIANT_ADAPTIVE):
        v_flows = [f for f in test_mal if f.variant == variant]
        v_esc = [f for f in v_flows if f.escalated]
        result.metrics[f"gate_recall_{variant}"] = (
            len(v_esc) / max(len(v_flows), 1))

    # ---- Stage 2 baseline: benign retry population -----------------------
    # The retry-distribution baseline comes from benign escalated flows only
    # (analyst-confirmed benign traffic through the feedback loop).
    retry_analyzer = RetryBehaviorAnalyzer(cfg)
    baseline = BaselineManager(cfg)
    benign_traces: list[RetryTrace] = []
    for f in esc_train_ben:
        trace = RetryTrace(f.flow_id, suppressed=True, delays=list(f.retry_delays or []))
        benign_traces.append(trace)
        baseline.confirm_benign(f.stage1_features, trace)
    retry_analyzer.fit_baseline(benign_traces)

    # ---- Stage 2 feature extraction on escalated flows --------------------
    train_esc = esc_train_ben + esc_train_mal
    test_esc = esc_test_ben + esc_test_mal
    rng_probe = np.random.default_rng(cfg.seed + 2)

    train_groups, test_groups = [], []
    train_dicts, test_dicts = [], []
    for f in train_esc:
        groups, _ = _stage2_features(f, cfg, retry_analyzer, rng_probe)
        fused = fuse_features(groups["passive"], groups["retry"], groups["dns"])
        f.stage2_features = fused
        train_groups.append(groups)
        train_dicts.append(fused)
    for f in test_esc:
        groups, _ = _stage2_features(f, cfg, retry_analyzer, rng_probe)
        fused = fuse_features(groups["passive"], groups["retry"], groups["dns"])
        f.stage2_features = fused
        test_groups.append(groups)
        test_dicts.append(fused)

    if not train_dicts or not test_dicts:
        raise RuntimeError("escalation gate produced empty train/test Stage-2 sets; "
                           "raise escalation_budget")

    # ---- fusion + Random Forest ------------------------------------------
    names = aligned_names(train_dicts + test_dicts)
    X_train = feature_matrix(train_dicts, names)
    y_train = [f.label for f in train_esc]
    X_test = feature_matrix(test_dicts, names)
    y_test = [f.label for f in test_esc]

    # ---- ablations: where does the detection power come from? -------------
    # Same RF trained on ONE feature group at a time, evaluated on the same
    # gated test split — shows what the retry features contribute, and how
    # much power survives the adaptive adversary's retry-timing reshaping.
    # "basepaper" composes the feature types the base paper uses on
    # plaintext DNS (11 traditional DNS features + blackhole point ratios),
    # giving a head-to-head against the full AdaptDNS fusion.
    adaptive_idx = [i for i, f in enumerate(test_esc) if f.variant == VARIANT_ADAPTIVE]
    naive_idx = [i for i, f in enumerate(test_esc) if f.variant == VARIANT_NAIVE]
    y_test_arr_ = np.asarray(y_test)
    ablation_groups: dict[str, tuple[str, ...]] = {
        "passive": ("passive",),
        "retry": ("retry",),
        "point_ratio": ("point_ratio",),
        "dns": ("dns",),
        "basepaper": ("dns", "point_ratio"),
    }
    for abl_name, members in ablation_groups.items():
        def _compose(group_dicts, members=members):
            out = []
            for gd in group_dicts:
                merged: dict[str, float] = {}
                for m in members:
                    merged.update({f"{m}__{k}": v for k, v in gd[m].items()})
                out.append(merged)
            return out

        tr_dicts = _compose(train_groups)
        te_dicts = _compose(test_groups)
        g_names = aligned_names(tr_dicts + te_dicts)
        abl = TunnelingClassifier(cfg).fit(
            feature_matrix(tr_dicts, g_names), y_train, g_names)
        preds_abl = abl.predict(feature_matrix(te_dicts, g_names))

        tp = int(((preds_abl == LABEL_TUNNEL) & (y_test_arr_ == LABEL_TUNNEL)).sum())
        fp = int(((preds_abl == LABEL_TUNNEL) & (y_test_arr_ == LABEL_BENIGN)).sum())
        fn = int(((preds_abl == LABEL_BENIGN) & (y_test_arr_ == LABEL_TUNNEL)).sum())
        result.metrics[f"ablation_{abl_name}_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
        if adaptive_idx:
            result.metrics[f"ablation_{abl_name}_detect_adaptive"] = float(
                np.mean(preds_abl[adaptive_idx] == LABEL_TUNNEL))
        if naive_idx:
            result.metrics[f"ablation_{abl_name}_detect_naive"] = float(
                np.mean(preds_abl[naive_idx] == LABEL_TUNNEL))

    clf = TunnelingClassifier(cfg).fit(X_train, y_train, names)
    verdicts = clf.verdicts(test_esc, X_test)
    preds = np.array([v.label for v in verdicts])
    y_test_arr = y_test_arr_

    # ---- metrics ----------------------------------------------------------
    tp = int(((preds == LABEL_TUNNEL) & (y_test_arr == LABEL_TUNNEL)).sum())
    fp = int(((preds == LABEL_TUNNEL) & (y_test_arr == LABEL_BENIGN)).sum())
    tn = int(((preds == LABEL_BENIGN) & (y_test_arr == LABEL_BENIGN)).sum())
    fn = int(((preds == LABEL_BENIGN) & (y_test_arr == LABEL_TUNNEL)).sum())
    result.metrics.update({
        "stage2_test_flows": float(len(test_esc)),
        "stage2_precision": tp / max(tp + fp, 1),
        "stage2_recall": tp / max(tp + fn, 1),
        "stage2_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "stage2_fpr": fp / max(fp + tn, 1),
        "confusion_tp": tp, "confusion_fp": fp, "confusion_tn": tn, "confusion_fn": fn,
    })

    # Per-variant detection (adaptive adversary reported separately).
    for variant in (VARIANT_NAIVE, VARIANT_ADAPTIVE):
        yt = np.array([f.variant == variant for f in test_esc])
        detected = (preds == LABEL_TUNNEL) & yt
        result.metrics[f"detect_rate_{variant}"] = float(detected.sum() / max(yt.sum(), 1))

    # End-to-end: a tunneling flow is caught only if escalated AND detected.
    caught = {f.flow_id for f, v in zip(test_esc, verdicts) if v.alert}
    n_mal_total = len(test_mal)
    n_ben_total = len(test_benign)
    result.metrics["e2e_detection_rate"] = len([f for f in test_mal if f.flow_id in caught]) / max(n_mal_total, 1)
    result.metrics["e2e_false_alert_rate"] = len([f for f in test_benign if f.flow_id in caught]) / max(n_ben_total, 1)

    # ---- feedback loop: analyst confirmation ------------------------------
    for f, v in zip(test_esc, verdicts):
        if f.label == LABEL_BENIGN and not v.alert:
            baseline.confirm_benign(f.stage1_features)           # benign path
        elif v.alert:
            baseline.confirm_malicious(f)                         # separate path

    result.classifier = clf
    result.scorer = scorer
    result.baseline = baseline
    result.threshold = threshold
    result.test_flows = test_esc
    result.verdicts = verdicts
    result.train_groups = train_groups
    result.test_groups = test_groups
    result.retry_analyzer = retry_analyzer
    return result
