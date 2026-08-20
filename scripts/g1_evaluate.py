#!/usr/bin/env python
"""G1 ablation evaluation: per-seed acc, variant-mean acc, ensemble accuracy,
McNemar tests, bootstrap CI on per-seed ensemble gain.

Reads logits cached by `scripts/g1_cache_logits.py`. Writes JSON results to
`results/g1_controlled_ablation.json` and `results/g1_stats.json`.

Usage:
  python scripts/g1_cache_logits.py           # one-time cache (~15s on MPS)
  python scripts/g1_evaluate.py               # compute and write results
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom
from torchvision import transforms

from fer.data.datasets import FERDataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/g1_logits")
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
labels = np.array([FERDataset(ROOT / "data" / "fer2013", "test",
                              transform=TFM, dataset="fer2013")[i][1]
                   for i in range(3589)])

CKPT_DIRS = {
    "v2":   "v2 Spatial,  wd=1e-4, no clip, LS=0.1",
    "v3":   "v3 S-Ph,     wd=0.1,  clip=2.0, LS=0.1",
    "v4":   "v4 S-Oc,     wd=0.1,  clip=2.0, LS=0.1",
    "g1v3": "v3 S-Ph G1,  wd=1e-4, no clip, LS=0.1",
    "g1v4": "v4 S-Oc G1,  wd=1e-4, no clip, LS=0.1",
}


def L(tag: str, seed: int) -> np.ndarray:
    return np.load(CACHE / f"logits_{tag}_s{seed}.npy")


def variant_mean(tag: str) -> np.ndarray:
    return np.stack([L(tag, s) for s in (42, 43, 44)]).mean(0)


def mcnemar_two_sided(a_pred: np.ndarray, b_pred: np.ndarray, y: np.ndarray):
    n10 = int(((a_pred == y) & (b_pred != y)).sum())
    n01 = int(((a_pred != y) & (b_pred == y)).sum())
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    k = min(n10, n01)
    p = 2 * binom.cdf(k, n, 0.5)
    if p > 1:
        p = 1.0
    return float(p), n10, n01


def holm(ps: list[float]) -> list[float]:
    ps = list(ps)
    n = len(ps)
    order = sorted(range(n), key=lambda i: ps[i])
    adj = [0.0] * n
    prev = 0.0
    for rank, i in enumerate(order):
        a = ps[i] * (n - rank)
        a = max(a, prev)
        a = min(a, 1.0)
        adj[i] = a
        prev = a
    return adj


# ---------- per-method summary ----------
summary = {"note": "G1 controlled regularization ablation (FER-2013 PrivateTest). "
                   "v3/v4 trained at matched regularization (wd=1e-4 / no grad clip / "
                   "LS=0.1, 50ep, 3 seeds) to isolate augmentation contribution.",
           "device": "mps", "split": "PrivateTest", "n": int(labels.size),
           "methods": {}}

for tag, desc in CKPT_DIRS.items():
    per_seed = [float((L(tag, s).argmax(1) == labels).mean()) for s in (42, 43, 44)]
    vm = variant_mean(tag)
    vm_acc = float((vm.argmax(1) == labels).mean())
    summary["methods"][tag] = {
        "desc": desc,
        "per_seed_acc": per_seed,
        "mean_acc": float(np.mean(per_seed)),
        "std_acc": float(np.std(per_seed, ddof=1)),
        "variant_mean_acc": vm_acc,
    }

# ---------- 2-model ensembles, w=0.5 (raw logits, matches paper code) ----------
def ens_pair(a: str, b: str, weight: float = 0.5):
    return [float(((weight * L(a, s) + (1 - weight) * L(b, s)).argmax(1) == labels).mean())
            for s in (42, 43, 44)]


ENSEMBLE_PAIRS = [
    ("v2",   "v4"),
    ("v3",   "v4"),
    ("v2",   "v3"),
    ("v2",   "g1v4"),
    ("g1v3", "g1v4"),
    ("v2",   "g1v3"),
    ("v3",   "g1v4"),
    ("v4",   "g1v3"),
]
summary["ensembles_w05_per_seed"] = {}
for a, b in ENSEMBLE_PAIRS:
    per_seed = ens_pair(a, b)
    best_single = max(summary["methods"][t]["mean_acc"] for t in (a, b))
    summary["ensembles_w05_per_seed"][f"{a}+{b}"] = {
        "per_seed_acc": per_seed,
        "mean_acc": float(np.mean(per_seed)),
        "std_acc": float(np.std(per_seed, ddof=1)),
        "best_single_mean": best_single,
        "gain_over_best_single_pt": float(np.mean(per_seed) - best_single),
    }

(RESULTS / "g1_controlled_ablation.json").write_text(json.dumps(summary, indent=2))
print("wrote results/g1_controlled_ablation.json")


# ---------- statistics for matched-reg G1 pair (v3+v4 matched reg) ----------
def per_seed_gain_vs_best_single(a_logits_per_seed, b_tags, seeds=(42, 43, 44)):
    """Gain of ensemble over per-seed best single (between two regimes)."""
    gains = []
    for s in seeds:
        a_pred = a_logits_per_seed[s].argmax(1)
        single_accs = {t: float((L(t, s).argmax(1) == labels).mean()) for t in b_tags}
        b_tag = max(single_accs, key=single_accs.get)
        b_acc = float((L(b_tag, s).argmax(1) == labels).mean())
        a_acc = float((a_pred == labels).mean())
        gains.append(a_acc - b_acc)
    return gains, b_tag


# Build cached ensemble logits per seed (g1v3+g1v4 w=0.5)
g1_ensemble_logits = {s: 0.5 * L("g1v3", s) + 0.5 * L("g1v4", s) for s in (42, 43, 44)}

# Per-seed McNemar vs per-seed best single between (g1v3, g1v4)
mcn_p = []
mcn_disc = []
mcn_best_tag = []
for s in (42, 43, 44):
    a_pred = g1_ensemble_logits[s].argmax(1)
    acc3 = float((L("g1v3", s).argmax(1) == labels).mean())
    acc4 = float((L("g1v4", s).argmax(1) == labels).mean())
    if acc3 >= acc4:
        b_pred = L("g1v3", s).argmax(1)
        which = "g1v3"
    else:
        b_pred = L("g1v4", s).argmax(1)
        which = "g1v4"
    p, n10, n01 = mcnemar_two_sided(a_pred, b_pred, labels)
    mcn_p.append(p)
    mcn_disc.append([n10, n01])
    mcn_best_tag.append(which)
mcn_holm = holm(mcn_p)

# Per-seed gain vs per-seed best single (the more honest baseline)
gains, picked_best = per_seed_gain_vs_best_single(g1_ensemble_logits, ("g1v3", "g1v4"))

# Bootstrap 95% CI on mean per-seed gain (vs per-seed best single)
rng = np.random.default_rng(0)
n = labels.size
boot = []
for _ in range(2000):
    idx = rng.integers(0, n, size=n)
    g_per_seed = []
    for s in (42, 43, 44):
        a_pred = g1_ensemble_logits[s].argmax(1)
        acc_b3 = (L("g1v3", s).argmax(1) == labels)[idx].mean()
        acc_b4 = (L("g1v4", s).argmax(1) == labels)[idx].mean()
        if acc_b3 >= acc_b4:
            best_b = (L("g1v3", s).argmax(1) == labels)[idx]
        else:
            best_b = (L("g1v4", s).argmax(1) == labels)[idx]
        g_per_seed.append((a_pred == labels)[idx].mean() - best_b.mean())
    boot.append(np.mean(g_per_seed))
boot = np.asarray(boot)
ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

stats = {
    "note": "G1 matched-reg pair statistics. Ensemble = G1 v3 + G1 v4 at w=0.5 "
            "raw logit mean per seed. Baseline = per-seed best single between "
            "G1 v3 and G1 v4.",
    "ensemble": "g1v3+g1v4 (matched reg)",
    "mcnemar_per_seed_vs_per_seed_best_single": {
        str(s): {"p": p, "discordant": d, "best_single_tag": t}
        for s, p, d, t in zip((42, 43, 44), mcn_p, mcn_disc, mcn_best_tag)
    },
    "holm_adjusted_mcnemar": dict(zip(("42", "43", "44"), mcn_holm)),
    "per_seed_gain_vs_per_seed_best_single_pt": [float(g * 100) for g in gains],
    "mean_per_seed_gain_pt": float(np.mean(gains) * 100),
    "bootstrap_95_ci_gain_pt": [float(ci_low * 100), float(ci_high * 100)],
}
(RESULTS / "g1_stats.json").write_text(json.dumps(stats, indent=2))
print("wrote results/g1_stats.json")
print(json.dumps({
    "mcnemar_per_seed": {s: round(stats["mcnemar_per_seed_vs_per_seed_best_single"][s]["p"], 6)
                         for s in ("42", "43", "44")},
    "holm_adjusted": {s: round(v, 6) for s, v in stats["holm_adjusted_mcnemar"].items()},
    "per_seed_gain_pt": [round(g, 3) for g in stats["per_seed_gain_vs_per_seed_best_single_pt"]],
    "mean_gain_pt": round(stats["mean_per_seed_gain_pt"], 3),
    "bootstrap_95ci_pt": [round(x, 3) for x in stats["bootstrap_95_ci_gain_pt"]],
}, indent=2))
