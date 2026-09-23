#!/usr/bin/env python3
"""AdaptDNS live dashboard — watch flows get scored, escalated, probed and
alerted in real time.

Two modes:

  synthetic (default)   Trains the full pipeline once, then streams FRESH,
                        previously-unseen flows through Stage 1 → gate →
                        Stage 2 → Random Forest with a live terminal view.
                        Nothing about a flow's fate is precomputed.

        python experiments/live_demo.py [--delay 0.35] [--limit 60]

  --gateway             Tails the REAL DoH gateway's admin API and renders
                        live blackhole-suppression and retry-observation
                        events as external clients hit it.

        # terminal 1: python gateway/doh_gateway.py
        # terminal 2: python experiments/live_demo.py --gateway

Stdlib only (ANSI colours); degrades to plain scrolling output when piped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from adaptdns.config import Config
from adaptdns.fusion import feature_matrix, fuse_features
from adaptdns.pipeline import _stage2_features, run_experiment
from adaptdns.stage1.features import extract_stage1_features
from adaptdns.synthetic.generator import generate_population

TTY = sys.stdout.isatty()

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"


def c(text: str, *codes: str) -> str:
    if not TTY:
        return text
    return "".join(codes) + text + RESET


def bar(score: float, max_score: float, width: int = 24) -> str:
    frac = min(score / max(max_score, 1e-9), 1.0)
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


# --------------------------------------------------------------------------
def run_synthetic(delay: float, limit: int, seed: int) -> None:
    cfg = Config(seed=seed)
    print(c("AdaptDNS live · training Stage-1 scorer + Random Forest "
            "(~10 s)…", DIM))
    t0 = time.time()
    result = run_experiment(cfg)
    print(c(f"trained in {time.time() - t0:.1f}s · threshold="
            f"{result.threshold:.1f} · budget={cfg.escalation_budget:.0%}\n",
            DIM))

    # Fresh, unseen flows — none of these were involved in training/eval.
    rng = np.random.default_rng(seed + 999)
    flows = generate_population(cfg, seed=seed + 999)
    rng.shuffle(flows)
    flows = flows[:limit] if limit > 0 else flows

    counts = {"flows": 0, "escalated": 0, "alerts": 0, "benign_ok": 0}
    max_score = result.threshold * 2.0
    np.random.seed(seed)

    print(c(f"{'flow':<16} {'suspicion':<8} {'bar':<26} action", DIM))
    print(c("─" * 92, DIM))

    for flow in flows:
        # ---- Stage 1: always-on passive profiling ------------------------
        feats = extract_stage1_features(flow, cfg)
        flow.stage1_features = feats
        score = result.scorer.score_features(feats)
        counts["flows"] += 1

        if score < result.threshold:
            counts["benign_ok"] += 1
            line = (f"{flow.flow_id:<16} {score:>7.1f}  "
                    f"{bar(score, max_score)}  "
                    + c("passively monitored (not escalated)", DIM))
            print(line)
        else:
            # ---- escalation gate ------------------------------------------
            flow.escalated = True
            counts["escalated"] += 1
            # ---- Stage 2: real blackhole trace + fusion -------------------
            groups, trace = _stage2_features(flow, cfg, result.retry_analyzer,
                                             np.random.default_rng(0))
            fused = fuse_features(groups["passive"], groups["retry"],
                                  groups["dns"])
            X = feature_matrix([fused], result.classifier.feature_names)
            prob = float(result.classifier.predict_proba_tunnel(X)[0])
            tunnel = prob >= 0.5

            retries = f"{trace.n_retries} retries@[" + \
                ", ".join(f"{d:.2f}s" for d in trace.delays[:4]) + "]"
            header = (f"{flow.flow_id:<16} {score:>7.1f}  "
                      f"{bar(score, max_score)}  "
                      + c(f"ESCALATE → probe ({retries}) → RF P(tun)"
                          f"={prob:.2f}", YELLOW))
            print(header)
            if tunnel:
                counts["alerts"] += 1
                print(" " * 17 + c("⚠ ALERT — tunneling verdict raised "
                                   "for SOC review "
                                   f"(variant={flow.variant}, label="
                                   f"{flow.label})", RED, BOLD))
            else:
                print(" " * 17 + c("✓ benign after probing — resumed "
                                   "normal processing", GREEN))
        time.sleep(delay)

    # ---- summary ----------------------------------------------------------
    # The budget governs BENIGN escalation (tunnels are *supposed* to be
    # escalated), so the rate below counts benign flows only.
    benign_seen = sum(1 for f in flows if f.label == "benign")
    benign_esc = sum(1 for f in flows
                     if f.label == "benign" and f.escalated)
    esc_rate = benign_esc / max(benign_seen, 1)
    print("\n" + c("─" * 92, DIM))
    print(c("session summary", BOLD))
    print(f"  flows processed      : {counts['flows']}")
    print(f"  escalated (Stage 2)  : {counts['escalated']}")
    print(f"  benign escalation    : {benign_esc}/{benign_seen} "
          f"({esc_rate:.1%} · budget {cfg.escalation_budget:.0%})")
    print(f"  alerts raised        : {counts['alerts']}")
    print(f"  passively monitored  : {counts['benign_ok']}")


# --------------------------------------------------------------------------
GATEWAY_ADMIN = "http://localhost:9080"


def run_gateway_view(poll: float = 1.0) -> None:
    print(c("AdaptDNS live · tailing REAL gateway blackhole events "
            f"({GATEWAY_ADMIN})", BOLD))
    print(c("start it with:  python gateway/doh_gateway.py", DIM))
    seen = 0
    try:
        while True:
            try:
                with urllib.request.urlopen(f"{GATEWAY_ADMIN}/admin/flows",
                                            timeout=2) as resp:
                    snap = json.loads(resp.read())
            except Exception as exc:
                print(c(f"  gateway unreachable ({exc}) — retrying…", DIM))
                time.sleep(poll)
                continue

            events = snap.get("events", [])
            for ev in events[seen:]:
                if ev["type"] == "blackhole":
                    print(c("  ⛔ blackhole ", RED, BOLD)
                          + f"suppressed response for {ev['qname']}")
                elif ev["type"] == "retry":
                    print(c("  ↻ retry observed ", YELLOW)
                          + f"{ev['qname']} after {ev['delay']:.3f}s "
                            "(real client timing)")
            seen = len(events)

            summary = [
                (fl["flow_id"], fl["n_queries"],
                 len(fl["observed_retry_delays"]))
                for fl in snap.get("flows", [])
            ]
            status = f"rules={snap.get('rules', [])} · flows={summary}"
            print(c("\r" + status.ljust(100), DIM), end="")
            sys.stdout.flush()
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\n" + c("stopped.", DIM))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gateway", action="store_true",
                    help="tail the real DoH gateway instead of streaming "
                         "synthetic flows")
    ap.add_argument("--delay", type=float, default=0.35,
                    help="seconds between streamed flows")
    ap.add_argument("--limit", type=int, default=60,
                    help="number of flows to stream (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.gateway:
        run_gateway_view()
    else:
        try:
            run_synthetic(args.delay, args.limit, args.seed)
        except KeyboardInterrupt:
            print("\n" + c("interrupted.", DIM))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
