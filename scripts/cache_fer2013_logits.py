#!/usr/bin/env python
"""Cache logits for all FER-2013 sweep ckpts to /tmp/logits for fast re-runs.

Run once (slow, ~5 min on MPS, ~5 min on CPU). Idempotent — skips already-cached
seeds. Cached files are required by scripts/g2_sweep_validation.py and any
subsequent sweep re-run.

Usage:
  python scripts/cache_fer2013_logits.py [--split test|validation|all]
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from fer.data.datasets import FERDataset
from fer.models.factory import get_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/logits")
CACHE.mkdir(parents=True, exist_ok=True)
TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
device = torch.device("mps" if torch.backends.mps.is_available()
                     else "cuda" if torch.cuda.is_available() else "cpu")

# (variant_tag, suffix_tag, runs_subdir) → ckpt path
SINGLE_REGIMES = ["v1", "v2", "v3", "v4", "v5", "hybrid"]    # 30 ep
LONG_REGIMES = ["v2_50ep_ls01", "v3_50ep_ls01", "v4_50ep_ls01"]  # 50 ep LS
SEEDS = (42, 43, 44)

# cache file naming convention:
#   /tmp/logits/<tag>_<split>_s<seed>.npy
# where tag like 'v1', 'hybrid', 'v2_50ep_ls01'

def variant_to_run(variant: str) -> str:
    """Resolve run dir name for a variant tag (matches sweep convention)."""
    if variant == "hybrid":
        return "efficientnet_b0_hybrid_fer2013"
    if "_50ep" in variant:
        base = variant.split("_50ep")[0]
        suffix = "_50ep_ls01"
        return f"efficientnet_b0_{base}_fer2013{suffix}"
    return f"efficientnet_b0_{variant}_fer2013"


def cache_one(variant: str, seed: int, split: str) -> None:
    # tag for filename is the variant name itself (v1, hybrid, v2_50ep_ls01)
    tag = variant
    out = CACHE / f"logits_{tag}_{split}_s{seed}.npy"
    if out.exists():
        return
    run_dir = ROOT / "runs" / variant_to_run(variant)
    ckpt = run_dir / f"seed_{seed}.pth"
    if not ckpt.exists():
        print(f"  MISSING: {ckpt}")
        return
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    m = get_model("efficientnet_b0", 7, 1, False).to(device).eval()
    m.load_state_dict(sd)
    ds = FERDataset(ROOT / "data/fer2013", split, transform=TFM, dataset="fer2013")
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    logits = []
    with torch.no_grad():
        for x, _ in dl:
            logits.append(m(x.to(device)).cpu().numpy())
    arr = np.concatenate(logits)
    np.save(out, arr)
    print(f"  wrote {out.name}  shape={arr.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "validation", "all"], default="all")
    args = parser.parse_args()
    splits = ["test", "validation"] if args.split == "all" else [args.split]

    variants = SINGLE_REGIMES + LONG_REGIMES
    t0 = time.time()
    for split in splits:
        print(f"=== split={split} ===")
        for v in variants:
            for s in SEEDS:
                cache_one(v, s, split)
    elapsed = (time.time() - t0) / 60
    print(f"total {elapsed:.1f} min")


if __name__ == "__main__":
    main()
