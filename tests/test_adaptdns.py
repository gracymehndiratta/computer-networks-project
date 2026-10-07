"""End-to-end and component tests for the AdaptDNS prototype.

Run with:  pytest -q    (or: python -m pytest -q)
"""
from __future__ import annotations

import numpy as np
import pytest

from adaptdns.baseline import BaselineManager
from adaptdns.config import Config
from adaptdns.datamodel import Flow, LABEL_BENIGN, LABEL_TUNNEL, Packet
from adaptdns.escalation import calibrate_threshold
from adaptdns.pipeline import run_experiment
from adaptdns.stage1.features import extract_stage1_features
from adaptdns.stage1.flow_id import identify_doh_dot_sessions, reconstruct_flows
from adaptdns.stage1.scoring import SuspicionScoringEngine
from adaptdns.stage2.dns_features import DNS_FEATURE_NAMES, extract_traditional_dns_features
from adaptdns.stage2.retry import RetryBehaviorAnalyzer
from adaptdns.stage2.blackhole import RetryTrace, selective_blackhole_probe
from adaptdns.synthetic.generator import generate_population


@pytest.fixture(scope="module")
def small_cfg() -> Config:
    return Config(n_benign=300, n_tunnel=80, n_adaptive=80, n_estimators=150, seed=7)


@pytest.fixture(scope="module")
def population(small_cfg):
    return generate_population(small_cfg, seed=7)


# --- Stage 1 ----------------------------------------------------------------
def test_stage1_features_finite_and_scale_aware(small_cfg, population):
    for flow in population[:30]:
        feats = extract_stage1_features(flow, small_cfg)
        assert feats, "no features extracted"
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} not finite for {flow.flow_id}"
        # multi-scale windows present
        for w in (1, 10, 60):
            assert f"w{w}s_count_cv" in feats


def test_stage1_passive_only_touches_packet_metadata(small_cfg, population):
    """Stage 1 must be computable from packets alone — drop resolver-side
    visibility and the feature vector must be unchanged."""
    flow = population[0]
    base = extract_stage1_features(flow, small_cfg)
    stripped = Flow(flow_id="x", src=flow.src, dst=flow.dst, proto=flow.proto,
                    packets=list(flow.packets))
    assert extract_stage1_features(stripped, small_cfg) == base


def test_flow_reconstruction_groups_by_connection():
    pkts = [
        ("10.0.0.2", "doh.local:443", "doh/443", 0.0, 500, 1),
        ("10.0.0.2", "doh.local:443", "doh/443", 0.1, 800, -1),
        ("10.0.0.3", "doh.local:443", "doh/443", 0.2, 400, 1),
        ("10.0.0.2", "doh.local:443", "doh/443", 100.0, 500, 1),  # idle gap -> new flow
    ]
    flows = reconstruct_flows(pkts)
    assert len(flows) == 3
    kept = identify_doh_dot_sessions(flows + [Flow("a", "s", "d", proto="http/80",
                                                   packets=[Packet(0, 100, 1)])])
    assert all(f.proto.startswith(("doh", "dot")) for f in kept)


def test_suspicion_engine_ranks_tunnels_above_benign(small_cfg, population):
    feats = [extract_stage1_features(f, small_cfg) for f in population]
    by_id = {f.flow_id: fd for f, fd in zip(population, feats)}
    benign = [by_id[f.flow_id] for f in population if f.label == LABEL_BENIGN][:200]
    engine = SuspicionScoringEngine().fit(benign)
    benign_mean = np.mean([engine.score_features(fd) for fd in benign])
    tunnel_flows = [f for f in population if f.label == LABEL_TUNNEL][:50]
    tunnel_mean = np.mean([engine.score_features(by_id[f.flow_id]) for f in tunnel_flows])
    assert tunnel_mean > benign_mean, "tunnels should score more suspicious than benign"


def test_threshold_respects_budget():
    scores = list(np.random.default_rng(0).random(1000))
    thr = calibrate_threshold(scores, 0.05)
    escalated = sum(1 for s in scores if s >= thr)
    assert escalated <= 80  # ~5% with quantile slack


# --- Stage 2 ----------------------------------------------------------------
def test_blackhole_never_probes_unescalated_flows(small_cfg, population):
    rng = np.random.default_rng(0)
    flow = population[0]
    flow.escalated = False
    trace = selective_blackhole_probe(flow, small_cfg, rng)
    assert not trace.suppressed and not trace.probed


def test_retry_analyzer_separates_periodic_from_jittered():
    cfg = Config()
    benign_traces = [RetryTrace(f"b{i}", True, list(np.random.default_rng(i).lognormal(0.0, 0.6, 3)))
                     for i in range(50)]
    az = RetryBehaviorAnalyzer(cfg).fit_baseline(benign_traces)

    periodic = RetryTrace("t", True, [1.0 + 0.01 * i for i in range(8)])
    jittered = RetryTrace("j", True, list(np.random.default_rng(3).lognormal(0.0, 0.9, 3)))

    f_periodic = az.extract(periodic, n_queries=50)
    f_jittered = az.extract(jittered, n_queries=50)
    assert f_periodic["retry_diff_cv"] < f_jittered["retry_diff_cv"]


def test_eleven_traditional_dns_features(small_cfg, population):
    tunnel = next(f for f in population if f.label == LABEL_TUNNEL)
    feats = extract_traditional_dns_features(tunnel)
    assert list(feats.keys()) == DNS_FEATURE_NAMES
    assert len(feats) == 11
    # No resolver visibility -> all zeros, no crash.
    empty = Flow("e", "s", "d", packets=[Packet(0, 100, 1)])
    assert all(v == 0.0 for v in extract_traditional_dns_features(empty).values())


# --- Feedback loop ----------------------------------------------------------
def test_baseline_keeps_benign_and_signatures_separate(small_cfg):
    mgr = BaselineManager(small_cfg)
    from adaptdns.stage1.features import extract_stage1_features
    evil = Flow("evil", "s", "d", packets=[Packet(0, 100, 1)], label=LABEL_TUNNEL)
    mgr.confirm_benign(extract_stage1_features(evil, small_cfg))  # mis-labelled benign confirm
    mgr.confirm_malicious(evil)
    assert mgr.benign_count == 1 and mgr.signature_count == 1
    # the two paths write to different stores by construction
    assert mgr.benign_stage1_features is not mgr.signatures.entries


# --- End-to-end -------------------------------------------------------------
def test_end_to_end_experiment(small_cfg):
    result = run_experiment(small_cfg)
    m = result.metrics

    # The escalation budget must roughly hold on held-out benign traffic.
    assert m["escalation_rate_benign_test"] <= small_cfg.escalation_budget + 0.05

    # Stage 2 must classify escalated flows well above chance.
    assert m["stage2_f1"] > 0.80
    assert m["stage2_fpr"] < 0.15

    # At FULL adversary strength the attacker mimics encrypted-flow
    # metadata, so the budget gate is the limiting factor (this is the
    # measured finding, not a defect): naive flows are still gated at
    # ~100%, adaptive gate recall is far lower...
    assert m["gate_recall_naive_tunnel"] > 0.95
    assert m["gate_recall_adaptive_tunnel"] < m["gate_recall_naive_tunnel"]
    # ...and end-to-end detection is gate-limited but well above chance.
    assert m["e2e_detection_rate"] > 0.45
    assert m["e2e_false_alert_rate"] < 0.15

    # OF the adaptive flows that DO reach Stage 2, the fused classifier
    # must still catch them (resolver-side DNS features carry this).
    assert m["detect_rate_adaptive_tunnel"] > 0.85

    # Feedback loop populated both paths independently.
    assert result.baseline.benign_count > 0


def test_intermediate_strength_holds_budget_and_detection():
    """At moderate adversary strength (the case study's 'simple retry-timing
    randomisation') the system should still catch nearly everything."""
    cfg = Config(n_benign=300, n_tunnel=80, n_adaptive=80, n_estimators=150,
                 seed=7, attack_strength=0.5)
    m = run_experiment(cfg).metrics
    assert m["e2e_detection_rate"] > 0.85
    assert m["escalation_rate_benign_test"] <= cfg.escalation_budget + 0.05
    # distributional retry features should not lag point ratios here
    assert (m["ablation_retry_detect_adaptive"]
            >= m["ablation_point_ratio_detect_adaptive"] - 0.10)


def test_xgboost_backend_matches_rf_quality():
    """The optional XGBoost backend should satisfy the same end-to-end
    quality gates as the default Random Forest."""
    pytest.importorskip("xgboost")
    cfg = Config(classifier_kind="xgboost", n_benign=300, n_tunnel=80,
                 n_adaptive=80, n_estimators=150, seed=7)
    result = run_experiment(cfg)
    m = result.metrics
    assert m["escalation_rate_benign_test"] <= cfg.escalation_budget + 0.05
    assert m["stage2_f1"] > 0.80
    assert m["stage2_fpr"] < 0.15
    assert m["e2e_detection_rate"] > 0.45
    assert m["detect_rate_adaptive_tunnel"] > 0.85
    assert result.classifier.feature_importances()
