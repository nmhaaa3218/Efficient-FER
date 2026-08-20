#!/usr/bin/env python
"""D1 — MC10: isolate augmentation-induced diversity vs seed/initialization diversity.

Compares same-regime (different-seed) ensembles vs cross-regime ensembles at the SAME computational cost (2 networks, w=0.5 / best sweep). From cached 50ep LS logits.

Configurations (all 2-net, raw-logit fusion):
  A) Spatial(s42) + Spatial(s43)          - same regime, different seeds
  B) Occlusion(s42) + Occlusion(s43)      - same regime, different seeds
  C) Spatial(s42) + Occlusion(s42)        - different regimes, same seed
  D) Spatial(s43) + Occlusion(s44)        - different regimes, different seeds (cross)
  E) Spatial(s42) + Occlusion(s43)        - different regimes, cross seed
Reports acc@0.5, best acc, gain over best single, disagreement.
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
    return np.load(f"/tmp/logits_{v}_50ep_ls01_s{s}.npy")

def run(pa_l, pb_l, y, name):
    ya = pa_l.argmax(1); yb = pb_l.argmax(1)
    acc05 = float(((0.5 * pa_l + 0.5 * pb_l).argmax(1) == y).mean())
    best, bw = 0.0, 0.5
    for w in np.arange(0, 1.001, 0.05):
        a = float(((w * pa_l + (1 - w) * pb_l).argmax(1) == y).mean())
        if a > best:
            best, bw = a, w
    best_single = max(float((ya == y).mean()), float((yb == y).mean()))
    D = float((ya != yb).mean())
    return {"name": name, "acc_05": round(acc05, 5), "best_acc": round(best, 5), "best_w": round(bw, 2),
            "best_single": round(best_single, 5), "gain_over_best_single": round(best - best_single, 5),
            "disagreement": round(D, 4)}

def main():
    y = labels()
    s42v2, s43v2, s44v2 = L("v2", 42), L("v2", 43), L("v2", 44)
    s42v4, s43v4, s44v4 = L("v4", 42), L("v4", 43), L("v4", 44)
    cfgs = [
        run(s42v2, s43v2, y, "Spatial(s42)+Spatial(s43)_same-regime-diff-seed"),
        run(s42v4, s43v4, y, "Occlusion(s42)+Occlusion(s43)_same-regime-diff-seed"),
        run(s42v2, s42v4, y, "Spatial(s42)+Occlusion(s42)_diff-regime-same-seed"),
        run(s43v2, s44v4, y, "Spatial(s43)+Occlusion(s44)_diff-regime-cross-seed"),
        run(s42v2, s43v4, y, "Spatial(s42)+Occlusion(s43)_diff-regime-cross-seed"),
    ]
    out = {"note": "MC10: 2-net ensembles isolating augmentation diversity vs seed/init diversity (raw-logit fusion, 50ep LS, PrivateTest).",
           "test_size": len(y), "configs": cfgs}
    f = ROOT / "results" / "ensemble_isolation_ablation.json"
    json.dump(out, open(f, "w"), indent=2)
    for c in cfgs:
        print(f"{c['name']:<45} 0.5={c['acc_05']*100:.2f} best={c['best_acc']*100:.2f} (w{c['best_w']}) gain={c['gain_over_best_single']*100:.2f} D={c['disagreement']}")
    print("wrote", f)

if __name__ == "__main__":
    main()
