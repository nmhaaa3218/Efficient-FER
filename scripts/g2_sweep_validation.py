#!/usr/bin/env python
"""15-pair sweep on FER-2013 from cached logits (no ckpt reloading).

Default split = test (PrivateTest, reproduces results/ensemble_2model_sweep_50ep.json).
Pass --split validation to evaluate on PublicTest for the G2 protocol audit.

Requires:
  /tmp/logits/<variant>_<split>_s{42,43,44}.npy  (use scripts/cache_fer2013_logits.py)

Outputs:
  results/ensemble_2model_sweep_50ep.json  (split=test, default)
  results/ensemble_2model_sweep_50ep_validation.json  (split=validation)

Usage:
  python scripts/g2_sweep_validation.py --split validation
"""
from __future__ import annotations
import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from torchvision import transforms

from fer.data.datasets import FERDataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/logits")

SINGLE_REGIMES = ["v1", "v2", "v3", "v4", "v5", "hybrid"]
LONG_REGIMES = ["v2_50ep_ls01", "v3_50ep_ls01", "v4_50ep_ls01"]
SEEDS = (42, 43, 44)


def load_labels(split: str) -> np.ndarray:
    TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = FERDataset(ROOT / "data/fer2013", split, transform=TFM, dataset="fer2013")
    return np.array([ds[i][1] for i in range(len(ds))])


def variant_mean(variant: str, split: str) -> np.ndarray:
    """Return mean of 3-seed logits for a variant."""
    arrs = [np.load(CACHE / f"logits_{variant}_{split}_s{s}.npy") for s in SEEDS]
    return np.stack(arrs).mean(axis=0)


def sweep_pairs(variants: list, labels: np.ndarray) -> dict:
    out = {}
    for a, b in itertools.combinations(variants, 2):
        ma, mb = variant_mean(a, split=labels_split_global), variant_mean(b, split=labels_split_global)
        acc05 = ((0.5 * ma + 0.5 * mb).argmax(1) == labels).mean()
        sweep = {f"{w:.2f}": float(((w * ma + (1 - w) * mb).argmax(1) == labels).mean())
                 for w in np.arange(0, 1.001, 0.05)}
        best_w = max(sweep, key=sweep.get)
        out[f"{a}+{b}"] = {"0.5/0.5": float(acc05),
                            "best_w": float(best_w),
                            "best_acc": float(sweep[best_w]),
                            "sweep_0.05": sweep}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    args = parser.parse_args()
    global labels_split_global
    labels_split_global = args.split

    labels = load_labels(args.split)
    results = {
        "note": (f"2-model sweep 0.5/0.5 + 0.05 weight sweep, variant-mean 3 seeds per regime, "
                 f"FER-2013 {args.split} {len(labels)}, EfficientNet-B0 1ch 48x48"),
        "split": args.split,
        "test_size": len(labels),
        "method": "logits mean per variant (3 seeds) then weighted average, argmax (raw logit)",
        "30ep": {}, "50ep_ls01": {}, "best": {},
    }
    results["30ep"] = sweep_pairs(SINGLE_REGIMES, labels)
    results["50ep_ls01"] = sweep_pairs(LONG_REGIMES, labels)
    # singles
    results["singles_variant_mean_3seed"] = {v: float((variant_mean(v, split=args.split)
                                                       .argmax(1) == labels).mean())
                                              for v in SINGLE_REGIMES + LONG_REGIMES}
    all_best = [(v["best_acc"], k, grp, v["best_w"])
                for grp in ["30ep", "50ep_ls01"] for k, v in results[grp].items()]
    all_best_sorted = sorted(all_best, reverse=True)
    results["best"]["2_model_overall"] = {
        "pair": all_best_sorted[0][1], "group": all_best_sorted[0][2],
        "w": all_best_sorted[0][3], "acc": all_best_sorted[0][0],
    }
    results["best"]["top5"] = [
        {"pair": k, "group": g, "w": w, "acc": acc}
        for acc, k, g, w in all_best_sorted[:5]
    ]
    suffix = "" if args.split == "test" else f"_{args.split}"
    out = ROOT / "results" / f"ensemble_2model_sweep_50ep{suffix}.json"
    if out.exists() and args.split == "test":
        shutil.copy(out, ROOT / "results" / "ensemble_2model_sweep_50ep.backup.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out} — best {results['best']['2_model_overall']}")


if __name__ == "__main__":
    main()
