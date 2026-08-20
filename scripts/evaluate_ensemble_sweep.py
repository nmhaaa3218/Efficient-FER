#!/usr/bin/env python
"""Reproduce 15-pair 0.5/0.5 + 0.05 weight sweep on FER-2013 split.

Default split is 'test' (PrivateTest, 3589), matching the original sweep.
Pass --split validation to evaluate on PublicTest (3589) for the G2 protocol
audit (paper line 98: 15-pair pair-selection was claimed to be on PublicTest).

Outputs:
  results/ensemble_2model_sweep_50ep.json
  results/ensemble_2model_sweep_50ep_validation.json  (when --split validation)

Usage:
  python scripts/evaluate_ensemble_sweep.py                      # PrivateTest (default)
  python scripts/evaluate_ensemble_sweep.py --split validation   # PublicTest
"""
from __future__ import annotations
import argparse
import json
import itertools
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from fer.data.datasets import FERDataset
from fer.models.factory import get_model

ROOT = Path(__file__).resolve().parents[1]
TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def logits_for(variant: str, suffix: str, seed: int, split: str = "test") -> np.ndarray:
    cache = ROOT / f"runs/efficientnet_b0_{variant}_fer2013{suffix}/seed_{seed}.pth"
    if not cache.exists():
        raise FileNotFoundError(f"Missing checkpoint {cache} — run ./scripts/run_sweep.sh first")
    ckpt = torch.load(str(cache), map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    m = get_model("efficientnet_b0", 7, 1, False)
    m.load_state_dict(sd)
    m.eval()
    ds = FERDataset(ROOT / "data/fer2013", split, transform=TFM, dataset="fer2013")
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(m(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def variant_mean_logits(variant: str, split: str = "test") -> np.ndarray:
    # variant like "v2" or "v2_50ep_ls01" → split
    if "_50ep" in variant:
        base = variant.split("_50ep")[0]
        suffix = "_50ep_ls01"
    else:
        base, suffix = variant, ""
    arrs = [logits_for(base, suffix, s, split=split) for s in (42, 43, 44)]
    return np.stack(arrs).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["test", "validation"], default="test",
        help="FER-2013 split: 'test' (PrivateTest, default) or 'validation' (PublicTest).",
    )
    args = parser.parse_args()
    split = args.split

    ds = FERDataset(ROOT / "data/fer2013", split, transform=TFM, dataset="fer2013")
    labels = np.array([ds[i][1] for i in range(len(ds))])
    variants_30 = ["v1", "v2", "v3", "v4", "v5", "hybrid"]
    variants_50 = ["v2_50ep_ls01", "v3_50ep_ls01", "v4_50ep_ls01"]
    results = {
        "note": (f"2-model sweep 0.5/0.5 + 0.05 weight sweep, variant-mean 3 seeds per regime, "
                 f"FER-2013 {split} {len(ds)}, EfficientNet-B0 1ch 48x48"),
        "split": split,
        "test_size": len(ds),
        "method": "logits mean per variant (3 seeds) then weighted average, argmax (raw logit)",
        "30ep": {}, "50ep_ls01": {}, "best": {},
    }
    for a, b in itertools.combinations(variants_30, 2):
        ma, mb = variant_mean_logits(a, split=split), variant_mean_logits(b, split=split)
        acc05 = ((0.5 * ma + 0.5 * mb).argmax(1) == labels).mean()
        sweep = {f"{w:.2f}": float(((w * ma + (1 - w) * mb).argmax(1) == labels).mean())
                 for w in np.arange(0, 1.001, 0.05)}
        best_w = max(sweep, key=sweep.get)
        results["30ep"][f"{a}+{b}"] = {
            "0.5/0.5": float(acc05), "best_w": float(best_w),
            "best_acc": float(sweep[best_w]), "sweep_0.05": sweep,
        }
        print(f"{a}+{b} 30ep 0.5 {acc05*100:.2f}% best {sweep[best_w]*100:.2f}% w{best_w}")
    for a, b in itertools.combinations(variants_50, 2):
        ma, mb = variant_mean_logits(a, split=split), variant_mean_logits(b, split=split)
        acc05 = ((0.5 * ma + 0.5 * mb).argmax(1) == labels).mean()
        sweep = {f"{w:.2f}": float(((w * ma + (1 - w) * mb).argmax(1) == labels).mean())
                 for w in np.arange(0, 1.001, 0.05)}
        best_w = max(sweep, key=sweep.get)
        results["50ep_ls01"][f"{a}+{b}"] = {
            "0.5/0.5": float(acc05), "best_w": float(best_w),
            "best_acc": float(sweep[best_w]), "sweep_0.05": sweep,
        }
        print(f"{a}+{b} 50ep 0.5 {acc05*100:.2f}% best {sweep[best_w]*100:.2f}% w{best_w}")
    singles = {v: float((variant_mean_logits(v, split=split).argmax(1) == labels).mean())
               for v in variants_30 + variants_50}
    results["singles_variant_mean_3seed"] = singles
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
    suffix = "" if split == "test" else f"_{split}"
    out = ROOT / "results" / f"ensemble_2model_sweep_50ep{suffix}.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out} — best {results['best']['2_model_overall']}")


if __name__ == "__main__":
    main()
