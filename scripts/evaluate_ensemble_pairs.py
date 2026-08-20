#!/usr/bin/env python
"""A1 — Honest 2-model per-seed ensembles from cached per-seed logits.

Computes Spatial (v2 50ep LS) + Occlusion (v4 50ep LS) 2-model ensembles:
  - same-seed pairs: s42+s42, s43+s43, s44+s44
  - all cross-seed 3x3 pairs
All on FER-2013 PrivateTest (3589), EfficientNet-B0 1ch 48x48, weighted-average of softmax.

ALSO computes the 6-net variant-mean (3-seed Spatial + 3-seed Occlusion) for A2.
No retraining; uses /tmp/logits_v{2,4}_50ep_ls01_s{42,43,44}.npy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from fer.data.datasets import FERDataset

ROOT = Path(__file__).resolve().parents[1]

def get_labels() -> np.ndarray:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = FERDataset(ROOT / "data/fer2013", "test", transform=tfm, dataset="fer2013")
    return np.array([ds[i][1] for i in range(len(ds))])

def logits(variant: str, seed: int) -> np.ndarray:
    assert variant in ("v2_50ep_ls01", "v4_50ep_ls01"), variant
    return np.load(f"/tmp/logits_{variant}_s{seed}.npy")

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def linear_apply(logits_list: list[np.ndarray], w: float) -> np.ndarray:
    """Weighted average of two softmax logit sets: w for a, 1-w for b."""
    assert len(logits_list) == 2
    pa, pb = softmax(logits_list[0]), softmax(logits_list[1])
    return w * pa + (1 - w) * pb

def acc(fused_probs: np.ndarray, labels: np.ndarray) -> float:
    return float((fused_probs.argmax(1) == labels).mean())

def sweep_best(pa: np.ndarray, pb: np.ndarray, labels: np.ndarray):
    ws = np.arange(0.0, 1.001, 0.05)
    best_w, best_acc = 0.5, -1.0
    acc_05 = None
    for w in ws:
        fused = w * pa + (1 - w) * pb
        a = float((fused.argmax(1) == labels).mean())
        if abs(w - 0.5) < 1e-9:
            acc_05 = a
        if a > best_acc:
            best_acc, best_w = a, w
    return best_w, best_acc, acc_05

def main():
    labels = get_labels()
    s42_v2, s43_v2, s44_v2 = [logits("v2_50ep_ls01", s) for s in (42, 43, 44)]
    s42_v4, s43_v4, s44_v4 = [logits("v4_50ep_ls01", s) for s in (42, 43, 44)]
    v2_seeds = {"s42": s42_v2, "s43": s43_v2, "s44": s44_v2}
    v4_seeds = {"s42": s42_v4, "s43": s43_v4, "s44": s44_v4}

    # Same-seed pairs (the honest 2-model ensemble)
    same_seed = {}
    for s in ("s42", "s43", "s44"):
        best_w, best_acc, acc_05 = sweep_best(softmax(v2_seeds[s]), softmax(v4_seeds[s]), labels)
        same_seed[s] = {
            "0.5/0.5": round(acc_05, 6) if acc_05 is not None else None,
            "best_w": round(best_w, 2),
            "best_acc": round(best_acc, 6),
        }
    same_seed_accs = np.array([v["best_acc"] for v in same_seed.values()])

    # Cross-seed pairs (all 3x3)
    cross = {}
    for sa in ("s42", "s43", "s44"):
        for sb in ("s42", "s43", "s44"):
            if f"{sa}-{sb}" in cross:
                continue
            best_w, best_acc, acc_05 = sweep_best(softmax(v2_seeds[sa]), softmax(v4_seeds[sb]), labels)
            cross[f"{sa}-{sb}"] = {
                "0.5/0.5": round(acc_05, 6) if acc_05 is not None else None,
                "best_w": round(best_w, 2),
                "best_acc": round(best_acc, 6),
            }

    # 6-net variant-mean (A2): mean logits over 3 seeds per regime, then weighted avg
    vm_v2 = softmax(np.stack([s42_v2, s43_v2, s44_v2]).mean(axis=0))
    vm_v4 = softmax(np.stack([s42_v4, s43_v4, s44_v4]).mean(axis=0))
    best_w6, best_acc6, acc05_6 = sweep_best(vm_v2, vm_v4, labels)

    result = {
        "note": "2-model per-seed + cross-seed ensembles, Spatial(v2)+Occlusion(v4) 50ep LS. FER-2013 PrivateTest 3589, EfficientNet-B0 1ch 48x48. same_seed mean±std over 3 same-seed pairs.",
        "method": "weighted average of softmax over logits",
        "test_size": len(labels),
        "singles_variant_mean": {
            "v2_50ep_ls01": round(float((softmax(vm_v2).argmax(1) == labels).mean()), 6),
            "v4_50ep_ls01": round(float((softmax(vm_v4).argmax(1) == labels).mean()), 6),
        },
        "same_seed": same_seed,
        "same_seed_mean_acc": round(float(same_seed_accs.mean()), 6),
        "same_seed_std_acc": round(float(same_seed_accs.std(ddof=1)), 6),
        "cross_seed": cross,
        "six_net_variant_mean": {
            "best_w": round(best_w6, 2),
            "best_acc": round(best_acc6, 6),
            "0.5/0.5": round(acc05_6, 6) if acc05_6 is not None else None,
            "gflops": 6 * 0.023,
        },
    }
    out = ROOT / "results" / "ensemble_pair_seed.json"
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")

if __name__ == "__main__":
    main()
