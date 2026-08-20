#!/usr/bin/env python
"""G1 ablation: cache logits from all relevant ckpts to /tmp/g1_logits/.

Run after v3/v4 50ep LS checkpoints land under the wd=1e-4 / no-clip
convention. Idempotent: skips ckpt whose cache file already exists.

Output:
  /tmp/g1_logits/logits_<tag>_s<n>.npy  for tag in {v2, v3, v4, g1v3, g1v4}

Usage:
  python scripts/g1_cache_logits.py
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from fer.data.datasets import FERDataset
from fer.models.factory import get_model

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/g1_logits")
CACHE.mkdir(parents=True, exist_ok=True)

TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
ds = FERDataset(ROOT / "data" / "fer2013", "test", transform=TFM, dataset="fer2013")
dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIRS = {
    "v2":   "efficientnet_b0_v2_fer2013_50ep_ls01",
    "v3":   "efficientnet_b0_v3_fer2013_50ep_ls01",
    "v4":   "efficientnet_b0_v4_fer2013_50ep_ls01",
    "g1v3": "efficientnet_b0_v3_fer2013_50ep_ls01_1e4_noclip",
    "g1v4": "efficientnet_b0_v4_fer2013_50ep_ls01_1e4_noclip",
}
SEEDS = (42, 43, 44)

t0 = time.time()
for tag, dirname in CKPT_DIRS.items():
    for seed in SEEDS:
        out = CACHE / f"logits_{tag}_s{seed}.npy"
        if out.exists():
            continue
        ckpt = ROOT / "runs" / dirname / f"seed_{seed}.pth"
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd["model"] if "model" in sd else sd
        m = get_model("efficientnet_b0", 7, 1, False, eca=False).to(device).eval()
        m.load_state_dict(sd)
        logits = []
        with torch.no_grad():
            for x, _ in dl:
                logits.append(m(x.to(device)).cpu().numpy())
        arr = np.concatenate(logits)
        np.save(out, arr)
        elapsed = (time.time() - t0) / 60
        print(f"{tag} seed{seed} {arr.shape}  (elapsed {elapsed:.1f}min)")
print(f"total {((time.time() - t0) / 60):.2f} min")
