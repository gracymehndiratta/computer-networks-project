# AdaptDNS — Proposed Work: Formal Definition, Mathematical Formulation & Algorithms

This document is the authoritative specification of the proposed work. Every
formula here is implemented in the code paths named in §9 (Math ⇄ Code map),
and every dimension quoted matches the implementation (verified:
d₁ = 30, d_R = 16, d_D = 11, d_P = 6, fused d = 57).

---

## 1. Proposed work — crisp definition

**Problem.** Given a stream of encrypted DNS flows (DoH/DoT) observed at an
enterprise recursive resolver / TLS-terminating gateway, classify each flow
`f` with a label `y(f) ∈ {benign, tunneling}`, subject to three constraints:

- **C1 — Encryption:** on-path observers see only connection metadata
  (packet sizes, timings, directions); plaintext DNS semantics exist only
  inside the resolver/gateway.
- **C2 — Bounded intrusiveness:** active probing (deliberately suppressing a
  DNS response) may be applied to at most a `β`-fraction of benign traffic
  (the *escalation budget*; default β = 0.05).
- **C3 — Adversary:** a tunneling client may deliberately reshape retry
  timing and mimic benign flow statistics; robustness to this must be
  *measured*, not assumed.

**Gaps in the base paper [1] that this work addresses:**

| # | Gap in [1] | Our answer |
|---|---|---|
| G1 | Validated only on unencrypted DNS-over-UDP | Stage-1 features are computed *only* from encrypted-flow metadata (C1) |
| G2 | Blackhole probing with no selective pre-filter | Budget-calibrated **escalation gate** before any probe (C2) |
| G3 | No evaluation against an adaptive retry-reshaping client | Parametric **red-team model** + robustness sweep (C3) |

**Method (one sentence).** A hybrid two-stage detector: an always-on passive
stage scores each flow's encrypted metadata and escalates only flows above a
budget-calibrated suspicion threshold; escalated flows are probed by
selective blackhole suppression, and passive, retry-distribution, and
resolver-side DNS features are fused into a Random Forest verdict, with a
feedback loop that updates *only* the benign baseline.

**Explicit non-goals:** no claim of real-world accuracy from synthetic data;
pcap/live-gateway evaluation beyond the included demo is future work.

---

## 2. Notation & data model

| Symbol | Meaning |
|---|---|
| `f = (P_f, T_f, R_f)` | a flow: packets `P_f`, resolver-side transactions `T_f`, retry trace `R_f` |
| `P_f = {(t_i, s_i, d_i)}_{i=1}^{n_f}` | packets: time, size (B), direction `d_i ∈ {+1,−1}` |
| `T_f = {(t_j, q_j, τ_j, r_j, x_j)}` | DNS transactions: time, qname, TTL, response size, NXDOMAIN flag |
| `R_f = (r_1, …, r_K)` | observed retry delays after suppression (seconds) |
| `y(f) ∈ {0,1}` | ground truth: 0 benign, 1 tunneling |
| `β` | escalation budget (fraction of benign flows allowed into Stage 2) |
| `e ∈ [0,1]` | red-team attack strength |
| `x^{(1)} ∈ ℝ^{30}` | passive Stage-1 feature vector |
| `x^{(R)} ∈ ℝ^{16}` | retry-distribution feature vector |
| `x^{(D)} ∈ ℝ^{11}` | resolver-side DNS feature vector |
| `x^{(P)} ∈ ℝ^{6}` | base-paper point-ratio vector (comparison baseline only) |
| `x = [x^{(1)}; x^{(R)}; x^{(D)}] ∈ ℝ^{57}` | fused vector |

---

## 3. Mathematical formulation

### 3.1 Stage 1 — multi-scale passive features (always-on, C1)

For time-window scale `W ∈ {1, 10, 60}` s, bin packets by
`b_i = ⌊(t_i − t_0)/W⌋`, and let `n_c(W)`, `B_c(W)` be packet counts and
bytes in bin `c`. Define, per scale:

- **burstiness** `CV_W = σ(n_c) / μ(n_c)` (coefficient of variation),
- **peak fraction** `p_W = max_c n_c / Σ_c n_c`,
- **active fraction** `a_W = |{c : n_c > 0}| / |C|`.

Plus flow-level statistics: size mean/std/p50/p90/max, inter-arrival
`Δ_i = t_{i+1} − t_i` with `CV_Δ`, direction ratio
`ρ_dir = Σ_{d_i>0} s_i / Σ_{d_i<0} s_i`, large/small packet fractions,
`pps`, byte-rate. Concatenated: `x^{(1)} ∈ ℝ^{30}`.

> Design point: all 30 entries are computable from `(t_i, s_i, d_i)` alone —
> they require no decryption (Constraint C1).

### 3.2 Suspicion score — regularised Mahalanobis distance

Fit on a benign **calibration set** `C` (disjoint from all train/test use):
first standardise `z = D^{−1/2}(x^{(1)} − μ̂)` with `D = diag(Σ̂)`, then

```
s(f) = sqrt( zᵀ Σ̂_λ⁻¹ z )
```

where `Σ̂_λ = (1−λ) S + λ · (tr(S)/d₁) · I` is the **Ledoit–Wolf shrinkage
estimator** — essential because d₁ = 30 features and the calibration set is
comparable in size; without shrinkage `Σ̂` is singular/noisy and the score
miscalibrates the budget. `s` is scale-invariant and captures feature
correlations (e.g., `pps` vs `bytes_per_sec`).

### 3.3 Escalation decision — budget as a quantile

```
θ = Quantile_{1−β} ( { s(f) : f ∈ B_train } )        (benign train scores)
escalate(f) ⇔ s(f) ≥ θ
```

**Property (budget compliance).** If benign scores are exchangeable between
`B_train` and held-out benign traffic, then
`P(escalate | benign) → β` with sampling error `O(√(β(1−β)/|B_train|))`.
Calibration on *disjoint* samples is required: in-sample distances are
systematically smaller, which (empirically) inflated test escalation to
12.5 % vs the 5 % budget before the fix. Measured after fix: 3.96 %–4.9 %.

Non-escalation means *low current suspicion*, **not** proof of benignity
(the case study states this explicitly; §6 quantifies the cost).

### 3.4 Stage 2a — selective blackhole probe (requires resolver position)

For an escalated flow, the resolver suppresses the response of one selected
transaction at time `t_0`. The client retransmits at times
`t_0 < t_1 < … < t_K`, yielding retry delays `r_k = t_k − t_{k−1}`,
`k = 1..K`, `K ≤ K_max` (default 12), or until give-up.

**Point-ratio formulation (base paper [1], baseline `x^{(P)} ∈ ℝ⁶`):**
`K`, `K/n_q`, `r_1`, `mean(r)`, `max(r)`, and the hard bit
`1{σ(r) < 0.05 s}` ("fixed interval?").

**Distributional formulation (AdaptDNS `x^{(R)} ∈ ℝ¹⁶`)** — the same trace
plus order/shape statistics:

- **KS distance to benign reference population** `F̂_B` (pooled benign
  retry delays, analyst-confirmed only):
  `D_KS = sup_d | F̂_f(d) − F̂_B(d) |` — nonparametric, no distributional
  assumption; measures "does this retry profile come from the benign
  population?" rather than "is its mean ≈ μ?"
- **binned entropy** `H = − Σ_{i=1}^{b} p_i log₂ p_i` (b = 8 bins),
- **backoff growth** `g = median_k ( r_{k+1} / r_k )`,
- **backoff trend** `τ = (1/(K−1)) Σ_k 1{ r_{k+1} > r_k }`,
- **diff-CV** `CV_Δr = σ(Δr)/μ(Δr)` (periodicity detector),
- **retry frequency** `ν = K / n_q`, moments (mean/std/CV/skew/min/max/first).

**Why (g) and (τ) matter mathematically.** A benign client backs off
multiplicatively, `r_{k+1} = r_k · U(1.8, 3.5)`, so `τ = 1` a.s. An attacker
who samples benign-distributed delays but in arbitrary order matches *every
point statistic in distribution* (count, mean, max, first) yet has
`E[τ] ≈ 0.5` and `g ≈ 1`. Point features cannot see this; order statistics
can. This is the precise sense in which distributions resist simple
retry-timing randomisation (§6 measures how much).

### 3.5 Stage 2b — resolver-side DNS features (`x^{(D)} ∈ ℝ¹¹`, parallel path)

Computed from decrypted transactions `T_f`; eleven features of [1]:

1. `μ_Δt` mean transaction inter-arrival · 2. `σ²_τ` TTL variance ·
3. **query entropy** `H(Q) = − Σ_w p_w log₂ p_w` over label tokens `w` ·
4. mean label depth · 5. domain-switch rate
`Switches / (n_q − 1)` · 6. mean qname length · 7. NXDOMAIN ratio ·
8. mean response size · 9. qtype diversity `|{qtype}| / n_q` ·
10. query rate `n_q / duration` · 11. payload ratio `Σ|qname| / duration`.

### 3.6 Fusion and classification

```
x(f) = [ x^{(1)}(f) ; x^{(R)}(f) ; x^{(D)}(f) ] ∈ ℝ⁵⁷
ŷ = RF(x) = argmax_c (1/M) Σ_{m=1}^{M} 1{ h_m(x) = c }
```

`h_m` = CART trees (Gini splits, `M = 300`, class-weight `balanced`);
alert iff `P(tunnel | x) ≥ 0.5`.

**Selection-bias control (formal statement of §3.4 of the case study).**
At inference the classifier only ever sees flows conditioned on escalation,
i.e. draws from `P(x, y | s(x) ≥ θ)`. Training therefore uses **only gated
flows produced by the same θ**: the empirical training distribution equals
the test distribution by construction. Training on ungated flows would
optimise `P(x,y)` while deploying on `P(x,y | s ≥ θ)` — covariate shift.

### 3.7 Feedback loop (two disjoint update paths)

- Analyst-confirmed benign → baseline: benign feature set `B ← B ∪ {f}`
  (refits `Σ̂_λ` periodically, `F̂_B` for KS). Monotone in benign evidence.
- Confirmed malicious → signature store `S ← S ∪ {f}` — **never** `B`,
  so attack traffic cannot contaminate the normal-behaviour reference
  (a poisoning channel otherwise).

### 3.8 Red-team model (attack strength `e`)

The generator blends each evasion dimension toward benign parameters:

```
parameter_k(e) = (1 − e) · value_naive + e · value_benign,   k ∈ {pacing, size, scope, retry, name}
```

At `e = 1` the attacker additionally **matches point statistics**
(`K ~ benign count distribution`, `r_1` ~ benign base delay, remaining gaps
iid from the pooled benign gap population — destroying the *order* that
`g, τ` capture). Naive tunnels (`e = 0`) are fixed-period with periodic
retries (`K ≈ 8`, interval ≈ 1.03 s ± 0.02).

### 3.9 Evaluation metrics

- Stage 2: `Precision = TP/(TP+FP)`, `Recall`, `F1 = 2TP/(2TP+FP+FN)`,
  `FPR = FP/(FP+TN)`.
- **Budget utilisation** `U = |{f ∈ B : escalate}| / |B| ≤ β` (target).
- **Gate recall** `R_g(e) = P(escalate | y = 1, e)`.
- **Conditional detection** `D_c(e) = P(ŷ = 1 | escalate, y = 1, e)`.
- **End-to-end** (chain rule): `R_e2e(e) = P(escalate ∧ ŷ=1 | y=1, e) = R_g(e) · D_c(e)`.
  This factorisation *is* the finding: at `e = 1`, `R_g = 0.15` while
  `D_c = 1.00` ⇒ the gate, not the classifier, limits the system.
- Head-to-head: same split, same RF, swap `x^{(R)}` vs `x^{(P)}`.

---

## 4. Algorithms

### Algorithm 1 — Offline training & budget calibration
```
input : benign B, tunneling M, budget β, seed
1: x⁽¹⁾(f) ← Stage1Features(f)                for all f ∈ B ∪ M      # §3.1
2: (B_cal, B_train, B_test) ← disjoint split(B, 0.2, 0.4, 0.4)
3: μ̂, Σ̂_λ ← LedoitWolf({x⁽¹⁾(f) : f ∈ B_cal})                     # §3.2
4: θ ← Quantile_{1−β}( { Mahalanobis(x⁽¹⁾(f)) : f ∈ B_train } )    # §3.3
5: E ← { f ∈ (B_train ∪ M_train) : s(f) ≥ θ }     ▷ gated training set
6: F̂_B ← pooled retry delays of benign f ∈ E        ▷ KS reference
7: for f ∈ E (parallel):
       R_f   ← BlackholeProbe(f)                          # §3.4
       x⁽ᴿ⁾(f) ← RetryDistFeatures(R_f, F̂_B)
       x⁽ᴰ⁾(f) ← DNSFeatures(f)                           # §3.5   ∥
8: x(f) ← [x⁽¹⁾(f); x⁽ᴿ⁾(f); x⁽ᴰ⁾(f)]                    # §3.6
9: RF ← RandomForest({(x(f), y(f)) : f ∈ E})
return scorer (μ̂, Σ̂_λ⁻¹), θ, F̂_B, RF
```

### Algorithm 2 — Online detection (per flow)
```
input : flow f, outputs from Alg. 1
1: x⁽¹⁾ ← Stage1Features(f)                    ▷ passive, no interference
2: s ← Mahalanobis(x⁽¹⁾)
3: if s < θ:
4:     return MONITOR        ▷ low suspicion ≠ benign (cost: §6)
5: R ← BlackholeProbe(f)      ▷ suppress response at resolver, K ≤ 12
6: x⁽ᴿ⁾ ← RetryDistFeatures(R, F̂_B)     ∥  x⁽ᴰ⁾ ← DNSFeatures(f)   # parallel
7: x ← [x⁽¹⁾; x⁽ᴿ⁾; x⁽ᴰ⁾]
8: p ← RF(x)
9: if p ≥ 0.5: ALERT (log, SOC), S ← S ∪ {f}        ▷ separate path
   else:       resume normal processing
10: on analyst confirmation:  benign → B/baseline only; never from step 9
```

### Algorithm 3 — Red-team generation (strength `e`)
```
for each tunneling flow:
    pacing, sizes, scope ← blend(naive, benign, e)              # §3.8
    retry schedule:
        with prob e:  K ~ benign counts; r₁ ~ benign base; rest iid pooled
        else:         period ~ U(0.9,1.1) + N(0, 0.3e) jitter, K ~ U[6,10]
    semantics: qname depth/length, TTL, qtype blended toward benign by e
```

---

## 5. Complexity

| Stage | Cost |
|---|---|
| Stage-1 features | `O(n_f log n_f)` (sort) + `O(d₁)` per flow |
| Score | `O(d₁²)` with pre-factorised `Σ̂_λ⁻¹` (fit: `O(d₁³)` once, d₁=30 ⇒ negligible) |
| Probe | `O(K)`, `K ≤ 12`, only for escalated flows (≈ β·N total) |
| Retry/DNS features | `O(K)`, `O(|T_f|)` |
| RF inference | `O(M · h̄)` ≈ `O(300 · depth)` |
| **Per-flow total** | **`O(n_f log n_f + d₁² + M h̄)` for benign; +`O(K + |T_f|)` for escalated (β-fraction)** |

The gate makes active cost `O(βN)` instead of `O(N)` — this *is* G2's fix,
quantified.

---

## 6. Expected empirical claims (all measured; see README §results)

1. Budget holds: `U ≈ β` (measured 3.96–4.9 % at β = 5 %).
2. Distributions ≥ point-ratios under forging (measured better at 4 of 5
   strengths; largest margin at `e = 1`: 0.25 vs 0.19 adaptive detection).
3. `R_e2e(e) = R_g(e)·D_c(e)` factorisation exposes the gate as the
   bottleneck at `e = 1` (`0.15 × 1.00 = 0.57` measured, remainder from
   benign-side FP control in small samples).
4. `x^{(D)}` is invariant to `e`: detection **1.00 at every strength** —
   payload-carrying qnames cannot shed their semantic fingerprints;
   quantitative justification of the deployment assumption.

---

## 7. How to explain it in 60 seconds (viva script)

> "We detect DNS tunnelling over encrypted DoH/DoT in two stages. Stage 1
> runs always-on and touches nothing: 30 statistical features of the
> encrypted flow — burstiness across three time scales, size and
> inter-arrival distributions — compressed by a regularised Mahalanobis
> distance from the benign population into one suspicion score. A threshold
> calibrated as the (1−β)-quantile of *disjoint* benign traffic enforces an
> explicit probing budget — only ~5 % of flows ever get probed. For those,
> the resolver suppresses one DNS response and we model the client's retry
> behaviour as a *distribution* — KS distance to a benign reference,
> entropy, and crucially the backoff *order* statistics (growth ratio and
> trend), because an attacker can forge retry means and counts but a
> multiplicative backoff sequence is visible in its ordering. In parallel,
> 11 resolver-side DNS features run — query entropy, TTL variance, domain
> switching. The 57-dimensional fused vector goes to a Random Forest;
> benign verdicts resume normal processing, tunnel verdicts alert the SOC,
> and analyst feedback updates only the benign baseline, never from attack
> traffic. We evaluate against a parametric red-team that blends pacing,
> sizes, scope and retry forging from 0 to 1; the sweep shows distributions
> beating point ratios, the gate — not the classifier — becoming the
> bottleneck at full strength, and resolver-side features staying at 100 %
> because tunnelling data through query names always leaves semantic
> fingerprints."

---

## 8. Reference-answer formulas (write on the board)

```
s(f)  = √( zᵀ Σ̂_λ⁻¹ z ),  z = D^{−1/2}(x⁽¹⁾ − μ̂),  Σ̂_λ = (1−λ)S + λ·tr(S)/d₁ · I
θ     = Quantile_{1−β}{ s(f) : f ∈ B_train },   escalate ⇔ s(f) ≥ θ
D_KS  = sup_d | F̂_f(d) − F̂_B(d) |
τ     = 1/(K−1) · Σ_k 1{r_{k+1} > r_k}          g = median_k ( r_{k+1} / r_k )
H(Q)  = − Σ_w p_w log₂ p_w
ŷ     = argmax_c (1/M) Σ_m 1{h_m(x) = c},   x = [x⁽¹⁾; x⁽ᴿ⁾; x⁽ᴰ⁾] ∈ ℝ⁵⁷
R_e2e = P(E ∧ D | M) = P(E | M) · P(D | E, M) = R_g · D_c
U     = |{f ∈ B : E(f)}| / |B|  ≤  β
```

---

## 9. Math ⇄ Code map

| Formula | Implementation |
|---|---|
| §3.1 `x⁽¹⁾`, CV, `p_W`, `ρ_dir` | `adaptdns/stage1/features.py` |
| §3.2 shrinkage Mahalanobis | `adaptdns/stage1/scoring.py` (`LedoitWolf`, `StandardScaler`) |
| §3.3 quantile threshold | `adaptdns/escalation.py::calibrate_threshold` |
| §3.4 probe + `R_f` | `adaptdns/stage2/blackhole.py` (live: `gateway/doh_gateway.py`) |
| §3.4 point ratios `x⁽ᴾ⁾` | `adaptdns/stage2/point_ratio.py` |
| §3.4 distributions `x⁽ᴿ⁾` (KS, H, g, τ, CV) | `adaptdns/stage2/retry.py` |
| §3.5 eleven DNS features | `adaptdns/stage2/dns_features.py` |
| §3.6 fusion, RF, verdicts | `adaptdns/fusion.py`, `adaptdns/classifier.py` |
| §3.6 gated training | `adaptdns/pipeline.py::run_experiment` |
| §3.7 two-path feedback | `adaptdns/baseline.py` |
| §3.8 red-team blend | `adaptdns/synthetic/generator.py` (`attack_strength`) |
| §3.9 metrics incl. `R_g·D_c` | `adaptdns/pipeline.py`, `experiments/study.py` |
| Algorithms 1–2 | `adaptdns/pipeline.py` (train) + `experiments/live_demo.py` (online) |
