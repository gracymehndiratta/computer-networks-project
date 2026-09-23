#!/usr/bin/env python3
"""Evaluation studies for the AdaptDNS case study.

Three studies, each producing a CSV + figure under artifacts/:

  budget      Escalation-budget sweep: does the gate respect the budget,
              and how does end-to-end detection trade off against it?
  robustness  Red-team strength sweep: how fast does each feature group
              (passive / retry-distribution / point-ratio / DNS / base-paper
              composite) and the fused system degrade as the adaptive
              adversary reshapes harder?
  seeds       Multi-seed headline metrics: mean ± std over N seeds.

Usage:
    python experiments/study.py budget      [--seeds 3]
    python experiments/study.py robustness  [--seeds 3]
    python experiments/study.py seeds       [--seeds 5]
    python experiments/study.py all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from adaptdns.config import Config
from adaptdns.pipeline import run_experiment

ART = Path("artifacts")

HEADLINE = [
    "escalation_rate_benign_test",
    "gate_recall_naive_tunnel",
    "gate_recall_adaptive_tunnel",
    "stage2_f1",
    "stage2_fpr",
    "detect_rate_naive_tunnel",
    "detect_rate_adaptive_tunnel",
    "e2e_detection_rate",
    "e2e_false_alert_rate",
    "ablation_point_ratio_detect_adaptive",
    "ablation_retry_detect_adaptive",
    "ablation_basepaper_detect_adaptive",
    "ablation_passive_detect_adaptive",
    "ablation_dns_detect_adaptive",
]


def _run(seed: int, budget: float, strength: float) -> dict:
    cfg = Config(seed=seed, escalation_budget=budget, attack_strength=strength)
    t0 = time.time()
    res = run_experiment(cfg)
    row = {"seed": seed, "budget": budget, "strength": strength,
           "runtime_s": round(time.time() - t0, 1)}
    row.update({k: v for k, v in res.metrics.items() if np.isscalar(v)})
    return row


def _save(df: pd.DataFrame, name: str) -> Path:
    ART.mkdir(exist_ok=True)
    csv = ART / f"{name}.csv"
    df.to_csv(csv, index=False, float_format="%.4f")
    print(f"[csv]  {csv}")
    return csv


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------
def study_budget(seeds: list[int], budgets: list[float]) -> pd.DataFrame:
    rows = [_run(s, b, 1.0) for b in budgets for s in seeds]
    df = pd.DataFrame(rows)
    _save(df, "sweep_budget")

    g = df.groupby("budget")
    plot = g[["escalation_rate_benign_test", "e2e_detection_rate",
              "e2e_false_alert_rate", "stage2_f1"]].mean()

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(plot.index, plot["escalation_rate_benign_test"], "o--",
            label="benign escalation rate (test)")
    ax.plot(plot.index, plot["e2e_detection_rate"], "s-",
            label="e2e tunnel detection")
    ax.plot(plot.index, plot["e2e_false_alert_rate"], "^-",
            label="e2e false-alert rate")
    ax.plot(plot.index, plot["stage2_f1"], "d-",
            label="Stage-2 F1")
    ax.plot(budgets, budgets, ":", color="gray", label="budget = x (target)")
    ax.set_xscale("log")
    ax.set_xlabel("escalation budget (log scale)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("AdaptDNS escalation-budget sweep (mean over seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ART / "sweep_budget.png", dpi=150)
    print(f"[plot] {ART / 'sweep_budget.png'}")
    return df


# --------------------------------------------------------------------------
def study_robustness(seeds: list[int], strengths: list[float]) -> pd.DataFrame:
    rows = [_run(s, 0.05, st) for st in strengths for s in seeds]
    df = pd.DataFrame(rows)
    _save(df, "sweep_robustness")

    g = df.groupby("strength")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    series = [
        ("ablation_point_ratio_detect_adaptive", "point-ratio (base paper) [1]"),
        ("ablation_retry_detect_adaptive", "retry distributions (AdaptDNS §3.3)"),
        ("ablation_basepaper_detect_adaptive", "base-paper composite (DNS+point)"),
        ("ablation_passive_detect_adaptive", "passive encrypted-flow (Stage 1)"),
        ("ablation_dns_detect_adaptive", "resolver-side DNS (11 features)"),
        ("detect_rate_naive_tunnel", "FUSED AdaptDNS (naive, ref.)"),
    ]
    for col, label in series:
        if col in g.mean().columns:
            ax.plot(g.mean().index, g.mean()[col], "o-", label=label, alpha=0.85)
    fused_adapt = g.mean()["detect_rate_adaptive_tunnel"]
    ax.plot(fused_adapt.index, fused_adapt, "o-", lw=3,
            label="FUSED AdaptDNS (adaptive)")

    ax.set_xlabel("red-team attack strength")
    ax.set_ylabel("detection rate on adaptive flows")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Robustness vs. adaptive adversary strength (mean over seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ART / "sweep_robustness.png", dpi=150)
    print(f"[plot] {ART / 'sweep_robustness.png'}")
    return df


# --------------------------------------------------------------------------
def study_seeds(n_seeds: int) -> pd.DataFrame:
    seeds = list(range(1, n_seeds + 1))
    rows = [_run(s, 0.05, 1.0) for s in seeds]
    df = pd.DataFrame(rows)
    _save(df, "seeds")

    print(f"\nHeadline metrics over {n_seeds} seeds "
          f"(budget=0.05, attack strength=1.0)")
    print(f"  {'metric':<40} {'mean':>8} {'std':>8}")
    for col in HEADLINE:
        if col in df.columns:
            print(f"  {col:<40} {df[col].mean():>8.3f} {df[col].std():>8.3f}")
    return df


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", choices=["budget", "robustness", "seeds", "all"])
    ap.add_argument("--seeds", type=int, default=3,
                    help="seeds per sweep point (budget/robustness)")
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds for the seeds study")
    ap.add_argument("--budgets", default="0.01,0.025,0.05,0.10,0.20")
    ap.add_argument("--strengths", default="0.0,0.25,0.5,0.75,1.0")
    args = ap.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    strengths = [float(s) for s in args.strengths.split(",")]
    seeds = list(range(1, args.seeds + 1))

    if args.study in ("budget", "all"):
        study_budget(seeds, budgets)
    if args.study in ("robustness", "all"):
        study_robustness(seeds, strengths)
    if args.study in ("seeds", "all"):
        study_seeds(args.n_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
