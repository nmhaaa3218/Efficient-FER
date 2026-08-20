#!/usr/bin/env python
"""C2 — Ensemble-diversity metrics (Q-statistic, double-fault, disagreement, error correlation) over all 15 regime pairs.

From cached per-seed logits (30ep + 50ep LS regimes, FER-2013 PrivateTest 3589).
For each pair of REGIMES (variant-mean 3-seed logits):
  - disagreement D_ab = P(yhat_a != yhat_b)
  - Q-statistic (Yule's Q) from concordant/discordant correct/incorrect counts
  - double-fault measure DF = P(both wrong)
  - pairwise error correlation
  - ensemble gain (best weighted) for reference
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
from torchvision import transforms
from fer.data.datasets import FERDataset

ROOT = Path(__file__).resolve().parents[1]

def labels():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = FERDataset(ROOT / "data/fer2013", "test", transform=tf, dataset="fer2013")
    return np.array([ds[i][1] for i in range(len(ds))])

def vm(logits_list):
    return np.stack(logits_list).mean(axis=0)  # logits mean (raw), matching paper method

REGS = ["v1", "v2", "v3", "v4", "v5", "hybrid", "v2_50ep_ls01", "v3_50ep_ls01", "v4_50ep_ls01"]

def load_rego(reg):
    if "_50ep" in reg:
        base = reg.split("_50ep")[0]; suffix = "_50ep_ls01"
    else:
        base, suffix = reg, ""
    arrs = [np.load(f"/tmp/logits_{base}{suffix}_s{s}.npy") for s in (42, 43, 44)]
    return vm(arrs)

def eff_metrics(ya, yb, y):
    ca = (ya == y); cb = (yb == y)
    n11 = int(((ca & cb)).sum())      # both correct
    n00 = int(((~ca) & (~cb)).sum())  # both wrong (double fault)
    n10 = int(((ca) & (~cb)).sum())
    n01 = int(((~ca) & (cb)).sum())
    D = float((ya != yb).mean())      # disagreement
    # Q-statistic
    if n11 * n00 == n10 * n01:
        Q = 0.0
    else:
        Q = float((n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01))
    DF = n00 / len(y)                 # double-fault
    # error correlation
    ea = (~ca).astype(float); eb = (~cb).astype(float)
    eac = ea - ea.mean(); ebc = eb - eb.mean()
    den = np.sqrt((eac**2).sum() * (ebc**2).sum())
    rho = float((eac * ebc).sum() / den) if den > 0 else 0.0
    return {"D": round(D, 4), "Q": round(Q, 4), "double_fault": round(DF, 4), "rho_err": round(rho, 4),
            "acc_a": round(float(ca.mean()), 4), "acc_b": round(float(cb.mean()), 4)}

def main():
    y = labels()
    logits = {r: load_rego(r) for r in REGS}
    out = {"note": "All-15 (and 50ep) regime-pair diversity metrics. variant-mean 3-seed raw logits, PrivateTest.",
           "pairs": {}}
    for a, b in itertools.combinations(REGS, 2):
        ya, yb = logits[a].argmax(1), logits[b].argmax(1)
        m = eff_metrics(ya, yb, y)
        # ensemble gain (best weighted raw-logit) for reference
        best = 0.0
        for w in np.arange(0, 1.001, 0.05):
            acc = float(((w * logits[a] + (1 - w) * logits[b]).argmax(1) == y).mean())
            best = max(best, acc)
        m["ens_gain"] = round(best - max(m["acc_a"], m["acc_b"]), 4)
        out["pairs"][f"{a}+{b}"] = m
    f = ROOT / "results" / "ensemble_diversity_metrics.json"
    json.dump(out, open(f, "w"), indent=2)
    print("wrote", f, "pairs:", len(out["pairs"]))
    # print the 30ep Spatial/Mixing/Occlusion highlights
    for k in ["v2+v3", "v2+v4", "v2+v5", "v3+v4"]:
        if k in out["pairs"]:
            print(k, out["pairs"][k])

if __name__ == "__main__":
    main()
