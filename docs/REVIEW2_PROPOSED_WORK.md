# Review 2: Proposed Work Definition — AdaptDNS

**Project:** AdaptDNS — Hybrid Two-Stage DNS-Tunneling Detection over Encrypted DoH/DoT Flows  
**Base Paper:** W. S. Alorainy, “Echoes From the Void: Detecting DNS Tunneling With Blackhole Features in Encrypted Scenarios With High Accuracy,” *IEEE Access*, vol. 13, pp. 138551–138567, Aug. 2025.

---

## 1. Proposed Work (One-line Definition)

**AdaptDNS** is a hybrid, two-stage intrusion-detection system that detects DNS tunneling over encrypted DNS (DoH/DoT) by:
1. running an always-on passive Stage 1 on encrypted-flow metadata,
2. gating suspicious flows through a budget-calibrated escalation threshold, and
3. applying selective blackhole probing + retry-distribution analysis + resolver-side DNS feature extraction only to the gated flows, fusing everything into a tree-based classifier.

The proposal explicitly addresses three limitations of the base paper and adds a parametric red-team model to measure robustness.

---

## 2. Why This Is the Proposed Work (Gaps in the Base Paper)

The base paper [1] introduced blackhole-based features for DNS tunneling, but it left three gaps:

| Gap ID | Limitation in Base Paper | Our Proposed Solution |
|--------|--------------------------|------------------------|
| **G1** | Validation performed on **unencrypted DNS-over-UDP** traffic. | Stage 1 of AdaptDNS uses **only encrypted-flow metadata** (packet sizes, timings, directions), so it works even when DNS payloads are inside TLS/DoH/DoT. |
| **G2** | Blackhole probing applied **without selective pre-filtering**; every flow is actively probed, which is intrusive and not deployable at scale. | We add a **budget-calibrated escalation gate** so active probing is applied to at most a β-fraction (e.g., 5%) of benign traffic. |
| **G3** | No evaluation against an **adaptive tunneling client** that reshapes retry timing to evade detection. | We introduce a **parametric red-team model** (`attack_strength e ∈ [0,1]`) that blends tunneling behavior toward benign statistics and evaluate robustness across strengths. |

Thus, the proposed work is **not** just “use a different classifier”; it is a complete detection architecture that upgrades the base paper’s blackhole idea to an encrypted, budgeted, adversarially-tested setting.

---

## 3. Problem Definition (Formal)

**Input:** A stream of encrypted DNS flows observed at an enterprise recursive resolver / TLS-terminating DoH gateway.

Each flow is represented as:

```
f = (P_f, T_f, R_f)
```

where:
- `P_f = {(t_i, s_i, d_i)}` — encrypted packets (time, size, direction)
- `T_f = {(t_j, q_j, τ_j, r_j, x_j)}` — DNS transactions visible only at resolver
- `R_f = (r_1, …, r_K)` — observed retry delays after a suppressed response

**Output:** A label `ŷ(f) ∈ {benign, tunneling}` and a confidence score `p(f) = P(tunneling | f)`.

**Constraints:**
1. **C1 — Encryption:** Stage 1 may only use `(t_i, s_i, d_i)`.
2. **C2 — Budget:** Active probing may be applied to at most fraction `β` of benign traffic.
3. **C3 — Adversary:** A tunneling client may reshape retry timing and flow statistics; robustness must be measured, not assumed.

---

## 4. Mathematical Concepts Used

### 4.1 Stage 1 — Multi-scale passive features
For each flow, extract a 30-dimensional feature vector `x^(1)(f)` from encrypted metadata:
- Flow-level statistics: packet count, duration, bytes/sec, PPS, size mean/std/CV/p50/p90/max, large/small packet fractions, direction ratio.
- Inter-arrival statistics: mean, std, CV, front-load.
- Per-scale burstiness for windows `W ∈ {1, 10, 60}` seconds:
  - `CV_W = σ(n_c) / μ(n_c)`
  - peak fraction `p_W = max_c n_c / Σ_c n_c`
  - active fraction `a_W = |{c : n_c > 0}| / |C|`

All 30 features satisfy **C1** because they need no decryption.

### 4.2 Suspicion Score — Regularized Mahalanobis Distance
Fit on a benign calibration set `C`:
1. Standardize: `z = D^(-1/2)(x^(1) − μ̂)` where `D = diag(Σ̂)`
2. Compute shrinkage covariance `Σ̂_λ = (1−λ)S + λ·(tr(S)/d_1)·I` using Ledoit-Wolf
3. Suspicion score:

```
s(f) = sqrt( z^T Σ̂_λ^(-1) z )
```

This is scale-invariant, correlation-aware, and stable when `d_1 = 30` features are used with limited calibration data.

### 4.3 Escalation Gate — Budget as Quantile
```
θ = Quantile_{1−β}( { s(f) : f ∈ B_train } )
escalate(f) ⇔ s(f) ≥ θ
```

**Property:** If benign scores are exchangeable between `B_train` and test traffic, then `P(escalate | benign) → β`.

### 4.4 Stage 2a — Blackhole Probe + Retry Distribution
For an escalated flow, suppress one response and observe retry delays `r_1, …, r_K` (`K ≤ 12`).

We extract a 16-dimensional vector `x^(R)(f)`:
- Count/rate: `n_retries`, `retry_rate = K / n_q`
- Moments: mean, std, CV, skew, min, max, first delay
- Order/shape statistics:
  - `retry_diff_cv = σ(Δr) / μ(Δr)`
  - `retry_periodicity = 1 / (1 + retry_diff_cv)`
  - `retry_growth = median_k (r_{k+1} / r_k)`
  - `retry_trend = (1/(K−1)) Σ_k 1{r_{k+1} > r_k}`
- Distribution comparison:
  - `retry_entropy` — binned entropy (8 bins)
  - `retry_ks_vs_benign` — two-sample KS distance to benign reference population `F̂_B`

**Why this resists adaptive attacks:** An attacker can sample benign-looking delays to match count/mean/max, but a benign client backs off multiplicatively (`r_{k+1} ≈ r_k · U(1.8,3.5)`), giving `trend ≈ 1` and `growth > 1`. Destroying that order while keeping point statistics benign is the adversarial tension we measure.

### 4.5 Stage 2b — Resolver-side DNS Features
From decrypted transactions `T_f`, extract the 11 features of the base paper:

```
x^(D)(f) = [
  txn_interarrival_mean, ttl_variance, query_entropy,
  subdomain_depth_mean, domain_switch_rate, qname_len_mean,
  nxdomain_ratio, response_size_mean, qtype_diversity,
  query_rate, qname_payload_ratio
]
```

These satisfy the deployment assumption: Stage 2 requires resolver/gateway position.

### 4.6 Fusion and Classification
Fused vector:

```
x(f) = [ x^(1)(f) ; x^(R)(f) ; x^(D)(f) ] ∈ ℝ^57
```

Classifier (default RF, now also XGBoost):

```
ŷ = argmax_c (1/M) Σ_{m=1}^M 1{ h_m(x) = c }
alert ⇔ P(tunneling | x) ≥ 0.5
```

**Selection-bias control:** The classifier is trained only on flows that passed the same escalation gate used at inference, so training distribution equals deployment distribution.

### 4.7 Evaluation Metrics
- Stage 2: Precision, Recall, F1, FPR
- Budget utilization: `U = |{f ∈ B : escalate}| / |B|`
- Gate recall: `R_g(e) = P(escalate | tunneling, strength e)`
- Conditional detection: `D_c(e) = P(ŷ = tunnel | escalate, tunneling, e)`
- End-to-end: `R_e2e(e) = R_g(e) · D_c(e)`

---

## 5. Proposed Algorithms

### Algorithm 1 — Offline Training & Budget Calibration
```
Input: benign flows B, tunneling flows M, budget β, seed
1.  x^(1)(f) ← Stage1Features(f)            for all f ∈ B ∪ M
2.  (B_cal, B_train, B_test) ← disjoint_split(B, 0.2, 0.4, 0.4)
3.  μ̂, Σ̂_λ ← LedoitWolf( { x^(1)(f) : f ∈ B_cal } )
4.  θ ← Quantile_{1−β}( { Mahalanobis(x^(1)(f)) : f ∈ B_train } )
5.  E ← { f ∈ (B_train ∪ M_train) : s(f) ≥ θ }      // gated training set
6.  F̂_B ← pooled retry delays of benign f ∈ E       // KS reference
7.  for each f ∈ E in parallel:
       R_f      ← BlackholeProbe(f)
       x^(R)(f) ← RetryDistFeatures(R_f, F̂_B)
       x^(D)(f) ← DNSFeatures(f)
8.  x(f) ← [ x^(1)(f) ; x^(R)(f) ; x^(D)(f) ]
9.  Classifier ← Train( { (x(f), y(f)) : f ∈ E } )
Output: scorer (μ̂, Σ̂_λ^(-1)), threshold θ, KS reference F̂_B, classifier
```

### Algorithm 2 — Online Detection (Per Flow)
```
Input: flow f, trained scorer, threshold θ, KS reference F̂_B, classifier
1.  x^(1) ← Stage1Features(f)                 // passive, no interference
2.  s ← Mahalanobis(x^(1))
3.  if s < θ:
4.      return MONITOR                        // low suspicion, not proof of benignity
5.  R ← BlackholeProbe(f)                     // suppress response, K ≤ 12
6.  x^(R) ← RetryDistFeatures(R, F̂_B)   ∥   x^(D) ← DNSFeatures(f)
7.  x ← [ x^(1) ; x^(R) ; x^(D) ]
8.  p ← Classifier.predict_proba_tunnel(x)
9.  if p ≥ 0.5:
        ALERT → SOC, add to signature store
    else:
        resume normal processing
10. on analyst confirmation:
        benign → update benign baseline only
        malicious → update signature store only
```

### Algorithm 3 — Red-team Generation (Attack Strength e)
```
For each tunneling flow:
    pacing, sizes, scope, retry, qname_semantics ← blend(naive, benign, e)
    if e > 0:
        K       ← drawn from benign retry-count distribution
        r_1     ← drawn from benign first-delay distribution
        r_2..r_K ← i.i.d. from pooled benign gap population
        // order is destroyed → backoff trend/growth collapse
    else:
        K       ← U[6,10]
        retries ← fixed period + tiny jitter
    qname length, TTL, qtype ← blended toward benign by e
```

---

## 6. Implementation Mapping

| Component | File |
|-----------|------|
| Data model (Flow, Packet, DnsTxn) | `adaptdns/datamodel.py` |
| Stage 1: flow ID + session reconstruction | `adaptdns/stage1/flow_id.py` |
| Stage 1: 30 passive features | `adaptdns/stage1/features.py` |
| Stage 1: Mahalanobis suspicion scorer | `adaptdns/stage1/scoring.py` |
| Escalation gate / budget threshold | `adaptdns/escalation.py` |
| Stage 2: blackhole probe | `adaptdns/stage2/blackhole.py` (live: `gateway/doh_gateway.py`) |
| Stage 2: retry-distribution features | `adaptdns/stage2/retry.py` |
| Stage 2: 11 resolver-side DNS features | `adaptdns/stage2/dns_features.py` |
| Base-paper point-ratio baseline | `adaptdns/stage2/point_ratio.py` |
| Feature fusion | `adaptdns/fusion.py` |
| Classifier (RF / XGBoost) + verdicts | `adaptdns/classifier.py` |
| Feedback loop (two-path baseline) | `adaptdns/baseline.py` |
| End-to-end pipeline | `adaptdns/pipeline.py` |
| Synthetic traffic + red-team | `adaptdns/synthetic/generator.py` |
| Real pcap ingestion | `adaptdns/pcap/ingest.py` |
| Real DoH gateway | `gateway/doh_gateway.py` |
| Evaluation studies | `experiments/study.py` |
| Live dashboard | `experiments/live_demo.py` |
| One-shot demo | `experiments/run_demo.py` |
| Diagnostics | `experiments/debug_ablation.py` |
| Tests | `tests/test_adaptdns.py` |

---

## 7. Current Implementation Status

- [x] Core two-stage pipeline implemented and tested
- [x] Encrypted-flow Stage-1 features (30-dim)
- [x] Ledoit-Wolf Mahalanobis suspicion scorer
- [x] Budget-calibrated escalation gate
- [x] Selective blackhole probing (simulated + real gateway)
- [x] Retry-distribution features (16-dim) + KS distance
- [x] Resolver-side DNS features (11-dim)
- [x] Feature fusion + Random Forest classifier
- [x] XGBoost classifier backend added
- [x] Two-path feedback loop
- [x] Parametric red-team generator (`attack_strength`)
- [x] Evaluation studies: budget sweep, robustness sweep, multi-seed
- [x] Real DoH gateway with actual blackhole suppression
- [x] 12 passing tests

---

## 8. Expected Empirical Claims (Measured in README)

1. **Budget holds:** benign escalation rate ≈ β (measured 3.96–4.9% at β = 5%).
2. **Distributional retry features beat point ratios** under adaptive attack at most strengths (largest margin at full strength: 0.25 vs 0.19 adaptive detection).
3. **Gate is the bottleneck:** at `e = 1`, `R_g ≈ 0.15` while `D_c = 1.00`, so end-to-end detection is gate-limited.
4. **Resolver-side DNS features are invariant to adaptive attack:** detection rate 1.00 at every strength, because payload-carrying qnames cannot hide semantic fingerprints.

---

## 9. Summary for Review 2

**Proposed work:** AdaptDNS upgrades the base paper’s blackhole-feature approach from unencrypted DNS-over-UDP to encrypted DoH/DoT by adding (i) a passive encrypted-flow stage, (ii) a budget-calibrated escalation gate to limit active probing, and (iii) distributional retry features evaluated against a parametric adaptive red-team. The system is mathematically formulated (Mahalanobis scoring, quantile gating, KS-based retry distributions, feature fusion, tree-based classification), algorithmically specified, and fully implemented with both synthetic and live-gateway evaluation.
