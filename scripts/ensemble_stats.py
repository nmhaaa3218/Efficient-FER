#!/usr/bin/env python
"""A4 — Paired statistical tests + CI for the 2-model ensemble gain.

Compares 2-net Spatial+Occlusion (per-seed) vs the best single on the SAME private-test samples.
- McNemar test (matched binary predictions) for each same-seed pair.
- Bootstrap CI (percentile) for the aggregated (3-seed) ensemble-vs-single accuracy difference.
All from cached per-seed logits; no retraining.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from torchvision import transforms
from fer.data.datasets import FERDataset

ROOT = Path(__file__).resolve().parents[1]

def labels():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = FERDataset(ROOT / "data/fer2013", "test", transform=tf, dataset="fer2013")
    return np.array([ds[i][1] for i in range(len(ds))])

def L(v, s):
    return np.load(f"/tmp/logits_{v}_s{s}.npy")

def mcnemar(yA, yB, y):
    """McNemar test on matched predictions; yA/yB correct/incorrect per system."""
    cA = (yA == y).astype(int)
    cB = (yB == y).astype(int)
    # discordant cells
    b = int(((cA == 1) & (cB == 0)).sum())  # A right, B wrong
    c = int(((cA == 0) & (cB == 1)).sum())  # A wrong, B right
    if b + c == 0:
        return 1.0, b, c
    # Edwards continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # two-sided p from chi2 (1 df)
    from math import erf, sqrt
    p = 1.0 - erf(sqrt(chi2 / 2.0))
    return p, b, c

def main():
    y = labels()
    out = {}
    same_seed_accs = []
    for s in (42, 43, 44):
        v2 = L("v2_50ep_ls01", s).argmax(1)
        v4 = L("v4_50ep_ls01", s).argmax(1)
        # 2-net ensemble at w=0.5
        ens = (0.5 * L("v2_50ep_ls01", s) + 0.5 * L("v4_50ep_ls01", s)).argmax(1)
        # best single
        a2 = float((v2 == y).mean()); a4 = float((v4 == y).mean())
        best_single = v2 if a2 >= a4 else v4
        a_single = max(a2, a4)
        a_ens = float((ens == y).mean())
        same_seed_accs.append(a_ens)
        p_s, b, c = mcnemar(ens, best_single, y)
        out[s] = {
            "ens_acc": round(a_ens, 5),
            "best_single_acc": round(a_single, 5),
            "gain": round(a_ens - a_single, 5),
            "mcnemar_p_vs_best_single": round(p_s, 4),
            "discordant_b_c": [b, c],
        }
        print(f"s{s}: ens {a_ens*100:.2f}% vs best single {a_single*100:.2f}%  gain +{(a_ens-a_single)*100:.2f}%  McNemar p={p_s:.4f}")

    same_seed_accs = np.array(same_seed_accs)
    mean_gain = same_seed_accs.mean() - np.array([
        max(float((L("v2_50ep_ls01", s).argmax(1) == y).mean()), float((L("v4_50ep_ls01", s).argmax(1) == y).mean()))
        for s in (42, 43, 44)
    ]).mean()
    # Bootstrap CI on per-sample ensemble-vs-best-single accuracy difference (pooled across 3 seeds)
    rng = np.random.default_rng(0)
    diffs = []
    for s in (42, 43, 44):
        v2 = L("v2_50ep_ls01", s).argmax(1); v4 = L("v4_50ep_ls01", s).argmax(1)
        ens = (0.5 * L("v2_50ep_ls01", s) + 0.5 * L("v4_50ep_ls01", s)).argmax(1)
        a2 = float((v2 == y).mean()); a4 = float((v4 == y).mean())
        best = v2 if a2 >= a4 else v4
        diffs.append((ens == y).astype(int) - (best == y).astype(int))
    diffs = np.concatenate(diffs)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(diffs), len(diffs))
        boots.append(diffs[idx].mean())
    boots = np.sort(boots)
    ci = (boots[int(0.025 * 2000)], boots[int(0.975 * 2000)])
    out["aggregate"] = {
        "mean_ens_acc": round(float(same_seed_accs.mean()), 5),
        "std_ens_acc": round(float(same_seed_accs.std(ddof=1)), 5),
        "mean_gain": round(float(mean_gain), 5),
        "bootstrap_ci_95_gain": [round(float(ci[0]), 5), round(float(ci[1]), 5)],
    }
    print("aggregate:", out["aggregate"])
    f = ROOT / "results" / "ensemble_stats.json"
    json.dump(out, open(f, "w"), indent=2)
    print("wrote", f)

if __name__ == "__main__":
    main()
