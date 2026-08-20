#!/usr/bin/env python
"""E1 — Quantitative evidence for the 'MixUp/CutMix low-pass' claim (MC6).

Compares how augmentation regimes alter the frequency content of 48x48 FER-2013 faces:
  - high-frequency energy fraction (2D FFT, Nyquist-normalized) before vs after augmentation
  - edge magnitude (Sobel/Laplacian) before vs after
  - per-regime: Spatial (v2), Mixing (v5); Mixing uses MixUp/CutMix (linear blend)
Both Geometric (Spatial) and manifold-mixing (MixUp blend) are computed on the SAME inputs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from fer.data.datasets import FERDataset
from fer.data.transforms import train_transform

ROOT = Path(__file__).resolve().parents[1]

def hf_fraction(x: np.ndarray, cutoff: float = 0.5) -> float:
    """Fraction of |FFT|^2 power above cutoff*fmax (radial), for a single grayscale image."""
    fx = np.fft.fftshift(np.fft.fft2(x))
    p = np.abs(fx) ** 2
    h, w = x.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    rmax = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    mask = r > cutoff * rmax
    return float(p[mask].sum() / (p.sum() + 1e-12))

def edge_magnitude(x: np.ndarray) -> float:
    gx = np.abs(np.diff(x, axis=1)).mean()
    gy = np.abs(np.diff(x, axis=0)).mean()
    return float((gx + gy) / 2)

def main():
    tf_eval = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = FERDataset(ROOT / "data/fer2013", "train", transform=tf_eval, dataset="fer2013")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=400, replace=False)

    geo = train_transform("v2")   # Spatial: crop/rotate/flip
    mix = train_transform("v5")   # Mixing: light spatial + batch-level MixUp/CutMix

    # (a) geometric-spatial effect on frequency content
    hf_geo, hf_mix, edge_geo, edge_mix = [], [], [], []
    for i in idx:
        img_t, _ = ds[i]
        img = img_t.squeeze(0).numpy()
        img01 = (img + 1) / 2
        img_u8 = (img01 * 255).astype(np.uint8)
        aug_geo_t = geo(Image.fromarray(img_u8))      # train_transform feeds PIL
        aug_geo = (aug_geo_t.squeeze(0).numpy() + 1) / 2  # back to [0,1]
        # MixUp blend with a second random image (alpha=0.4)
        j = int(rng.integers(0, len(ds)))
        img2_t, _ = ds[j]
        img2 = (img2_t.squeeze(0).numpy() + 1) / 2
        lam = float(np.random.beta(0.4, 0.4))
        blended = lam * img01 + (1 - lam) * img2
        hf_geo.append(hf_fraction(aug_geo))
        hf_mix.append(hf_fraction(blended))
        edge_geo.append(edge_magnitude(aug_geo))
        edge_mix.append(edge_magnitude(blended))
        if len(hf_geo) >= 400:
            break
    hf_geo, hf_mix = np.array(hf_geo), np.array(hf_mix)
    edge_geo, edge_mix = np.array(edge_geo), np.array(edge_mix)

    # original references
    hf_orig = np.mean([hf_fraction((ds[i][0].squeeze(0).numpy() + 1) / 2) for i in idx])
    edge_orig = np.mean([edge_magnitude((ds[i][0].squeeze(0).numpy() + 1) / 2) for i in idx])

    res = {
        "note": "400 FER-2013 train faces; Mixing = MixUp blend (beta 0.4) with a second image; Spatial = crop/rotate/flip.",
        "high_freq_fraction": {
            "original": round(float(hf_orig), 5),
            "spatial_mean": round(float(hf_geo.mean()), 5),
            "mixing_mean": round(float(hf_mix.mean()), 5),
            "spatial_delta": round(float(hf_geo.mean() - hf_orig), 5),
            "mixing_delta": round(float(hf_mix.mean() - hf_orig), 5),
        },
        "edge_magnitude": {
            "original": round(float(edge_orig), 5),
            "spatial_mean": round(float(edge_geo.mean()), 5),
            "mixing_mean": round(float(edge_mix.mean()), 5),
        },
    }
    out = ROOT / "results" / "freq_evidence.json"
    import json
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("wrote", out)

if __name__ == "__main__":
    main()