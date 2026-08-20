from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from fer.config import Config
from fer.data.datasets import FERDataset
from fer.data.transforms import eval_transform
from fer.eval.ensemble import collect_probs, weighted_average, reciprocal_rank_fusion
from fer.models.factory import get_model
from fer.training.metrics import compute_metrics
from fer.utils.device import get_device, to_device
from torch.utils.data import DataLoader


def load_probs_for_variant(variant: str, dataset: str, device, split: str = "validation"):
    """Load probs for a variant by finding its latest checkpoint."""
    # Find checkpoint: runs/efficientnet_b0_<variant>_<dataset>/seed_42.pth (best seed)
    ckpt = Path(f"runs/efficientnet_b0_{variant}_{dataset}/seed_42.pth")
    if not ckpt.exists():
        # Try any seed
        cands = sorted(Path(f"runs/efficientnet_b0_{variant}_{dataset}").glob("seed_*.pth"))
        if not cands:
            raise FileNotFoundError(f"No checkpoint for {variant} {dataset}")
        ckpt = cands[0]
    cfg = Config.from_yaml(f"configs/train/efficientnet_b0_{variant}_{dataset}.yaml")
    model = get_model(cfg.model.name, cfg.model.num_classes, cfg.model.in_channels, pretrained=False, eca=cfg.model.eca)
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = to_device(model, device)
    ds = FERDataset(Path(cfg.data.root) / cfg.data.name, split, transform=eval_transform(48), dataset=cfg.data.name)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    probs, labels = collect_probs([model], loader, device)
    return probs[0], labels  # (N,7), (N,)


def main():
    p = argparse.ArgumentParser(description="5-model ensemble weight optimization (grid search + RRF)")
    p.add_argument("--dataset", default="fer2013", help="fer2013 or ferplus")
    p.add_argument("--split", default="validation", help="validation for tuning, test for final")
    p.add_argument("--variants", nargs="+", default=["v1", "v2", "v3", "v4", "v5"], help="5 variants")
    p.add_argument("--step", type=float, default=0.1, help="Grid step for weight search (0.1 = 11^5 combos, use 0.2 for faster)")
    args = p.parse_args()

    device = get_device("auto")
    print(f"Loading {len(args.variants)} variants for {args.dataset} {args.split}...")
    all_probs = []
    labels = None
    for v in args.variants:
        probs, labs = load_probs_for_variant(v, args.dataset, device, args.split)
        all_probs.append(probs)
        if labels is None:
            labels = labs
        print(f"  {v}: probs {probs.shape}, acc {compute_metrics(labs, probs.argmax(1), 7)['accuracy']:.4f}")
    all_probs = np.stack(all_probs)  # (M,N,7)
    print(f"Stacked probs: {all_probs.shape}")

    # Grid search (coarse, step 0.2 → 6^5=7776 combos, feasible)
    best = {"acc": 0, "weights": None}
    n = len(args.variants)
    steps = np.arange(0, 1.0001, args.step)
    # Use simplex: weights sum to 1, step discretization
    # For step 0.1, use integer compositions
    from itertools import product as iproduct

    total = 0
    for w in iproduct(steps, repeat=n):
        if abs(sum(w) - 1.0) > 1e-6:
            continue
        fused = weighted_average(all_probs, weights=list(w))
        preds = fused.argmax(1)
        acc = (preds == labels).mean()
        total += 1
        if acc > best["acc"]:
            best = {"acc": float(acc), "weights": list(w), "macro_f1": float(compute_metrics(labels, preds, 7)["macro_f1"])}
    print(f"Grid search ({total} combos, step {args.step}): best acc {best['acc']:.4f} weights {best['weights']} macro_f1 {best['macro_f1']:.4f}")

    # RRF
    rrf_probs = reciprocal_rank_fusion(all_probs, k=60)
    rrf_preds = rrf_probs.argmax(1)
    rrf_acc = (rrf_preds == labels).mean()
    print(f"RRF k=60: acc {rrf_acc:.4f}")

    # Save
    out = Path(f"results/ensemble_5model_{args.dataset}_{args.split}_opt.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"grid": best, "rrf": {"acc": float(rrf_acc)}, "variants": args.variants}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
