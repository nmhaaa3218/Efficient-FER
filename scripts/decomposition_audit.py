#!/usr/bin/env python
"""Decomposition of ensemble gain by source of diversity at α=0.5 fixed.

Three categories:
  A) Same regime, diff seed:    per-seed cross-seed pair within one regime
                                  (3 + 3 = 6 pairs total: v2 s42+s43, v2 s42+s44, v2 s43+s44,
                                   v4 s42+s43, v4 s42+s44, v4 s43+s44)
  B) Diff regime, same seed:    same-seed cross-regime pair
                                  (3 pairs total: v2+v4 at s42, s43, s44)
  C) Diff regime, cross seed:   cross-seed cross-regime pair
                                  (6 pairs total: v2+v4 at (s,t) for s≠t in {42,43,44})

For each pair we report:
  - Ensemble accuracy at α=0.5 fixed (raw logit, matches paper §5)
  - Best single (max of the two single accuracies)
  - Gain vs best single  (= ensemble - best single)
  - Gain vs mean single  (= ensemble - mean(singles))

For each category, mean ± std across pairs + bootstrap 95% CI for the
mean of per-pair gains (vs max single; 2000 resamples).

Schedules evaluated:
  - Mixed-reg (v3+v4 50ep_ls01): original Stage-2 paper schedule.
  - Matched-reg (g1v3+g1v4 50ep_ls01): G1 controlled ablation.

Note: matched-reg cannot decompose by regime (g1v3 and g1v4 are
themselves the constituents); we report only the matched-reg cross-regime
same-seed analysis (3 pairs).

Reproduces:
  python scripts/decomposition_audit.py
Writes:
  results/decomposition_audit.json
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from torchvision import transforms

from fer.data.datasets import FERDataset

ROOT = Path(__file__).resolve().parents[1]
TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

SEEDS = (42, 43, 44)


def load_labels(split: str = "test") -> np.ndarray:
    ds = FERDataset(ROOT / "data" / "fer2013", split, transform=TFM, dataset="fer2013")
    return np.array([ds[i][1] for i in range(len(ds))])


def L(tag: str, seed: int) -> np.ndarray:
    """Load cached logits for a given regime variant and seed.

    Cache conventions (in priority order):
      /tmp/logits/logits_<variant>_test_s<seed>.npy        (Stage-1/2 caches, scripts/cache_fer2013_logits.py)
      /tmp/g1_logits/logits_<variant>_s<seed>.npy           (G1 caches, scripts/g1_cache_logits.py)
    """
    p1 = Path(f"/tmp/logits/logits_{tag}_test_s{seed}.npy")
    if p1.exists():
        return np.load(p1)
    p2 = Path(f"/tmp/g1_logits/logits_{tag}_s{seed}.npy")
    if p2.exists():
        return np.load(p2)
    raise FileNotFoundError(
        f"Missing cached logits for variant={tag} seed={seed}. Run "
        f"scripts/cache_fer2013_logits.py (FER-2013 sweeps) and/or "
        f"scripts/g1_cache_logits.py (G1 ckpts)."
    )


def pair_metrics(la: np.ndarray, lb: np.ndarray, labels: np.ndarray):
    """Return per-pair metrics at α=0.5 fixed."""
    pred_e = (0.5 * la + 0.5 * lb).argmax(1)
    ens = float((pred_e == labels).mean())
    pred_a = la.argmax(1)
    pred_b = lb.argmax(1)
    a_acc = float((pred_a == labels).mean())
    b_acc = float((pred_b == labels).mean())
    best_single = max(a_acc, b_acc)
    mean_single = 0.5 * (a_acc + b_acc)
    return {
        "ensemble_alpha05": ens,
        "single_a": a_acc,
        "single_b": b_acc,
        "best_single": best_single,
        "mean_single": mean_single,
        "gain_vs_max": ens - best_single,
        "gain_vs_mean": ens - mean_single,
    }


def categorize(regime_a: str, regime_b: str) -> str:
    """Return one of {same-regime-diff-seed, diff-regime-same-seed,
    diff-regime-cross-seed} based on (regime_a, regime_b) (and seed combo).
    """
    raise NotImplementedError("Use specific generators below.")


def gen_same_regime_diff_seed(regime: str):
    """All (seed_i, seed_j) pairs with seed_i != seed_j."""
    return [((regime, regime), (s_i, s_j))
            for s_i, s_j in combinations(SEEDS, 2)]


def gen_cross_regime_same_seed(regime_a: str, regime_b: str):
    """All (regime_a, regime_b) at the same seed."""
    return [((regime_a, regime_b), (s, s)) for s in SEEDS]


def gen_cross_regime_cross_seed(regime_a: str, regime_b: str):
    """All (regime_a, regime_b) at (seed_i, seed_j) with seed_i != seed_j."""
    return [((regime_a, regime_b), (s_i, s_j))
            for s_i, s_j in combinations(SEEDS, 2)]


def bootstrap_ci(values: list, n_boot: int = 2000, seed: int = 0):
    """Bootstrap 95% CI on the mean of a list of values."""
    if not values:
        return None, None, None
    arr = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def schedule_pair_metrics(regime_a: str, regime_b: str):
    """Compute all 3 categories for a (regime_a, regime_b) pair."""
    labels = load_labels("test")
    out = {}
    for cat_name, pairs in [
        ("same-regime-diff-seed", []),  # filled below per regime
        ("diff-regime-same-seed", gen_cross_regime_same_seed(regime_a, regime_b)),
        ("diff-regime-cross-seed", gen_cross_regime_cross_seed(regime_a, regime_b)),
    ]:
        results = []
        for (ra, rb), (s_i, s_j) in pairs:
            la, lb = L(ra, s_i), L(rb, s_j)
            m = pair_metrics(la, lb, labels)
            m["regime_a"] = ra
            m["regime_b"] = rb
            m["seed_a"] = s_i
            m["seed_b"] = s_j
            m["pair_label"] = f"{ra}(s{s_i})+{rb}(s{s_j})"
            results.append(m)
        out[cat_name] = results

    # Same-regime diff-seed needs special handling (per regime):
    same_regime_results = []
    for regime in (regime_a, regime_b):
        if regime_a == regime_b:
            continue  # skip duplicates when regimes are the same
        for (ra, rb), (s_i, s_j) in gen_same_regime_diff_seed(regime):
            la, lb = L(ra, s_i), L(rb, s_j)
            m = pair_metrics(la, lb, labels)
            m["regime_a"] = ra
            m["regime_b"] = rb
            m["seed_a"] = s_i
            m["seed_b"] = s_j
            m["pair_label"] = f"{ra}(s{s_i})+{rb}(s{s_j})"
            same_regime_results.append(m)
    out["same-regime-diff-seed"] = same_regime_results

    return out


def aggregate(results: dict):
    """Aggregate per-pair gains into per-category mean/std/95% CI."""
    agg = {}
    for cat, pairs in results.items():
        if not pairs:
            agg[cat] = {"n_pairs": 0}
            continue
        gains_max = [p["gain_vs_max"] for p in pairs]
        gains_mean = [p["gain_vs_mean"] for p in pairs]
        m_max, ci_lo_max, ci_hi_max = bootstrap_ci(gains_max)
        m_mean, ci_lo_mean, ci_hi_mean = bootstrap_ci(gains_mean)
        agg[cat] = {
            "n_pairs": len(pairs),
            "gain_vs_max_single_pt_mean": float(np.mean(gains_max)) * 100,
            "gain_vs_max_single_pt_std": float(np.std(gains_max, ddof=1)) * 100,
            "gain_vs_max_single_95ci_pt": (ci_lo_max * 100, ci_hi_max * 100),
            "gain_vs_mean_single_pt_mean": float(np.mean(gains_mean)) * 100,
            "gain_vs_mean_single_pt_std": float(np.std(gains_mean, ddof=1)) * 100,
            "gain_vs_mean_single_95ci_pt": (ci_lo_mean * 100, ci_hi_mean * 100),
        }
    return agg


def main():
    schedules = {
        "mixed-reg (Stage-2 50ep_ls01, headline)": ("v3", "v4"),
        "matched-reg (G1 controlled ablation)": ("g1v3", "g1v4"),
    }
    out = {
        "note": "Decomposition of ensemble gain (α=0.5 fixed, raw logit fusion) "
                "into three categories at 50ep_ls01 schedule. "
                "Per-pair gains are computed vs both max single (utility) and "
                "mean single (overall improvement); bootstrap 95% CI is over "
                "the per-pair gains (2000 resamples, seed=0).",
        "alpha": 0.5,
        "fusion": "raw logit mean",
        "schedules": {},
    }
    for schedule_name, (ra, rb) in schedules.items():
        per_pair = schedule_pair_metrics(ra, rb)
        agg = aggregate(per_pair)
        out["schedules"][schedule_name] = {
            "regime_a": ra,
            "regime_b": rb,
            "aggregate": agg,
            "per_pair": per_pair,
        }
    (ROOT / "results" / "decomposition_audit.json").write_text(json.dumps(out, indent=2))
    print("wrote results/decomposition_audit.json")
    for schedule_name, (ra, rb) in schedules.items():
        print(f"\n=== {schedule_name} ({ra} + {rb}) ===")
        agg = out["schedules"][schedule_name]["aggregate"]
        for cat in ("same-regime-diff-seed", "diff-regime-same-seed", "diff-regime-cross-seed"):
            d = agg.get(cat, {})
            if d.get("n_pairs", 0) == 0:
                print(f"  {cat:<30} no pairs")
                continue
            print(f"  {cat:<30} n={d['n_pairs']}")
            print(f"    gain vs max single (pt): mean={d['gain_vs_max_single_pt_mean']:+.2f} ± "
                  f"{d['gain_vs_max_single_pt_std']:.2f}; 95% CI {d['gain_vs_max_single_95ci_pt']}")
            print(f"    gain vs mean single (pt): mean={d['gain_vs_mean_single_pt_mean']:+.2f} ± "
                  f"{d['gain_vs_mean_single_pt_std']:.2f}; 95% CI {d['gain_vs_mean_single_95ci_pt']}")


if __name__ == "__main__":
    main()
