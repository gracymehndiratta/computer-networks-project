#!/usr/bin/env python3
"""Run the AdaptDNS end-to-end experiment on synthetic traffic.

Usage:
    python experiments/run_demo.py [--seed 42] [--budget 0.05]

Outputs a metrics report and saves plots under artifacts/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptdns.config import Config
from adaptdns.pipeline import run_experiment


def print_report(cfg: Config, result) -> None:
    m = result.metrics
    print("=" * 68)
    print("AdaptDNS — end-to-end experiment")
    print("=" * 68)
    print(f"population: {cfg.n_benign} benign / {cfg.n_tunnel} naive tunnel / "
          f"{cfg.n_adaptive} adaptive tunnel   (seed {cfg.seed})")
    print(f"escalation budget: {cfg.escalation_budget:.0%}\n")

    print("-- Stage 1: passive profiling & escalation gate " + "-" * 21)
    print(f"  calibrated suspicion threshold     : {m['threshold']:.3f}")
    print(f"  benign escalation rate (train)     : {m['escalation_rate_benign_train']:.2%}")
    print(f"  benign escalation rate (test)      : {m['escalation_rate_benign_test']:.2%}  (budget {cfg.escalation_budget:.0%})")
    print(f"  escalation recall — tunnels (train): {m['escalation_recall_train']:.2%}")
    print(f"  escalation recall — tunnels (test) : {m['escalation_recall_test']:.2%}")

    print("\n-- Stage 2: blackhole probing + fusion + Random Forest " + "-" * 11)
    print(f"  escalated test flows               : {int(m['stage2_test_flows'])}")
    print(f"  precision / recall / F1            : {m['stage2_precision']:.3f} / "
          f"{m['stage2_recall']:.3f} / {m['stage2_f1']:.3f}")
    print(f"  false-positive rate (stage 2)      : {m['stage2_fpr']:.2%}")
    print(f"  confusion TP/FP/TN/FN              : "
          f"{int(m['confusion_tp'])}/{int(m['confusion_fp'])}/"
          f"{int(m['confusion_tn'])}/{int(m['confusion_fn'])}")

    print("\n-- End-to-end (escalation gate AND classifier) " + "-" * 22)
    print(f"  detection rate — naive tunnels     : {m['detect_rate_naive_tunnel']:.2%}")
    print(f"  detection rate — adaptive tunnels  : {m['detect_rate_adaptive_tunnel']:.2%}  (the adversary)")
    print(f"  e2e detection rate (all tunnels)   : {m['e2e_detection_rate']:.2%}")
    print(f"  e2e false-alert rate (benign)      : {m['e2e_false_alert_rate']:.2%}")

    print("\n-- Ablation: feature group alone (same gated split) " + "-" * 15)
    print(f"  {'group':<12} {'F1':>7} {'naive':>8} {'adaptive':>9}")
    for group in ("passive", "retry", "point_ratio", "dns", "basepaper"):
        print(f"  {group:<12} {m[f'ablation_{group}_f1']:>7.3f} "
              f"{m[f'ablation_{group}_detect_naive']:>8.2%} "
              f"{m[f'ablation_{group}_detect_adaptive']:>9.2%}")
    print("  (F1 overall / detection on naive / on adaptive flows;")
    print("   'basepaper' = 11 DNS features + blackhole point ratios [1],")
    print("   'retry' = AdaptDNS distributional retry features)")
    print(f"  fused AdaptDNS   {m['stage2_f1']:>7.3f} "
          f"{m['detect_rate_naive_tunnel']:>8.2%} "
          f"{m['detect_rate_adaptive_tunnel']:>9.2%}")

    print("\n-- Feedback loop " + "-" * 46)
    print(f"  benign baseline entries            : {result.baseline.benign_count}")
    print(f"  signatures (separate path)         : {result.baseline.signature_count}")

    print("\n-- Top 10 features (Random Forest) " + "-" * 32)
    for name, imp in list(result.classifier.feature_importances().items())[:10]:
        print(f"  {name:<40} {imp:.4f}")


def save_plots(cfg: Config, result) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib not available; skipping plots")
        return

    out = Path(cfg.artifacts_dir)
    out.mkdir(exist_ok=True)

    # Confusion matrix
    m = result.metrics
    cm = np.array([[m["confusion_tn"], m["confusion_fp"]],
                   [m["confusion_fn"], m["confusion_tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["benign", "tunnel"])
    ax.set_yticks([0, 1], ["benign", "tunnel"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("AdaptDNS Stage-2 classification (escalated test flows)")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Feature importances
    imp = result.classifier.feature_importances()
    top = dict(list(imp.items())[:15])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(range(len(top))[::-1], top.values(), color="#3572b0")
    ax.set_yticks(range(len(top))[::-1], top.keys(), fontsize=8)
    ax.set_xlabel("importance")
    ax.set_title("Top-15 fused features")
    fig.tight_layout()
    fig.savefig(out / "feature_importances.png", dpi=150)
    plt.close(fig)

    print(f"[plot] saved {out / 'confusion_matrix.png'} and {out / 'feature_importances.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=0.05, help="escalation budget (fraction of flows)")
    ap.add_argument("--benign", type=int, default=1200)
    ap.add_argument("--tunnel", type=int, default=300)
    ap.add_argument("--adaptive", type=int, default=300)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        seed=args.seed,
        escalation_budget=args.budget,
        n_benign=args.benign,
        n_tunnel=args.tunnel,
        n_adaptive=args.adaptive,
    )
    result = run_experiment(cfg)
    print_report(cfg, result)
    if not args.no_plots:
        save_plots(cfg, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
