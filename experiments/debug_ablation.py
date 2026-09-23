"""Diagnostic: why do point-ratio features beat distributional retry
features against the adaptive adversary (and is that an artifact?)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from adaptdns.config import Config
from adaptdns.pipeline import run_experiment

cfg = Config()
res = run_experiment(cfg)
flows = res.test_flows
groups = res.test_groups

def report(group):
    print(f"\n=== {group} feature means by class ===")
    keys = groups[0][group].keys()
    rows = []
    for k in keys:
        benign = [g[group][k] for g, f in zip(groups, flows) if f.label == "benign"]
        naive = [g[group][k] for g, f in zip(groups, flows) if f.variant == "naive_tunnel"]
        adap = [g[group][k] for g, f in zip(groups, flows) if f.variant == "adaptive_tunnel"]
        rows.append((k, np.mean(benign), np.mean(naive), np.mean(adap),
                     np.std(benign)))
    print(f"  {'key':<22} {'benign':>10} {'naive':>10} {'adaptive':>10} {'b-std':>10}")
    for k, b, n, a, s in rows:
        # mark features where adaptive is many benign stds away
        sep = " <<<" if s > 0 and abs(a - b) > 3 * s else ""
        print(f"  {k:<22} {b:>10.3f} {n:>10.3f} {a:>10.3f} {s:>10.3f}{sep}")

print("escalated test flows:", len(flows))
for g in ("point_ratio", "retry"):
    report(g)

# Single-feature AUC (benign vs adaptive ONLY — naive excluded from negatives)
print("\n=== single-feature AUC (benign vs adaptive) ===")
mask = np.array([f.variant in ("benign", "adaptive_tunnel") for f in flows])
for g in ("point_ratio", "retry"):
    for k in groups[0][g]:
        vals = np.array([gg[g][k] for gg in groups])[mask]
        y = np.array([f.variant == "adaptive_tunnel" for f in flows])[mask]
        n1, n0 = int(y.sum()), int((~y).sum())
        if n1 == 0 or n0 == 0:
            continue
        # tie-aware rank AUC (1-based ranks)
        order = np.argsort(np.argsort(vals)) + 1
        r1 = order[y].sum()
        auc = (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)
        flag = " <<<" if (auc > 0.85 or auc < 0.15) else ""
        print(f"  {g:<12} {k:<22} auc={auc:.3f}{flag}")

# Train-set class counts
from collections import Counter
print("\ntrain labels:", Counter(f.label for f in res.test_flows))
print("ablations:",
      {k: round(v, 3) for k, v in res.metrics.items() if k.startswith("ablation")})
