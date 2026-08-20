"""Statistical rigor audit for Q1 Table 1 — 3-seed mean±std, significance."""
import json
import pathlib
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
results_dir = ROOT / "results"

# Load all 3-seed results
models = [
    "efficientnet_b0_v1_fer2013",
    "efficientnet_b0_v2_fer2013",
    "efficientnet_b0_v3_fer2013",
    "efficientnet_b0_v4_fer2013",
    "efficientnet_b0_v5_fer2013",
    "efficientnet_b0_hybrid_fer2013",
    "shufflenetv2_fer2013",
    "mobilenetv3_small_fer2013",
    "efficientnet_b0_v5_ferplus",
    "efficientnet_b0_v2_ferplus",
]

print("=== Statistical Rigor Audit (3-seed mean±std) ===")
print(f"{'Model':30s} | {'Test Acc':20s} | {'Val Acc':20s} | n")
print("-" * 90)
for name in models:
    p = results_dir / f"{name}.json"
    if not p.exists():
        # Try alternative name
        alt = list(results_dir.glob(f"{name}*.json"))
        if alt:
            p = alt[0]
        else:
            continue
    data = json.loads(p.read_text())
    test = data.get("test", {}).get("accuracy", {})
    val = data.get("val", {}).get("accuracy", {})
    if isinstance(test, dict) and "mean" in test:
        print(f"{name:30s} | {test['mean']:.4f}±{test['std']:.4f} {test['values']} | {val.get('mean', 0):.4f}±{val.get('std', 0):.4f} | {len(test['values'])}")
    else:
        print(f"{name:30s} | {test} | {val} | ?")

# Significance: V3+V4 69.96% vs V2+V5 68.29% +1.67% — check non-overlapping CI and t-test
print("\n=== Significance: V3+V4 69.96% vs V2+V5 68.29% ===")
# Load ensemble results
ens_v2v5 = json.loads((results_dir / "ensemble_v2_v5_fer2013.json").read_text())
ens_5model = json.loads((results_dir / "ensemble_5model_v1v2v3v4hybrid_fer2013.json").read_text())
# For V3+V4, we need to compute from seed42 only? Actually V3+V4 69.96% is seed42 0.6996, but we need 3-seed mean for V3+V4
# Let's compute 3-seed for V3+V4 from individual seeds
import torch
from fer.config import Config
from fer.models.factory import get_model
from fer.utils.device import get_device, to_device
from fer.eval.ensemble import collect_probs, weighted_average
from fer.data.datasets import FERDataset
from fer.data.transforms import eval_transform
from torch.utils.data import DataLoader

# Quick 3-seed V3+V4
device = get_device("auto")
cfg = Config.from_yaml("configs/train/efficientnet_b0_v2_fer2013.yaml")
accs_v3v4 = []
for seed in [42,43,44]:
    m_v3 = get_model("efficientnet_b0", 7, 1, False)
    m_v3.load_state_dict(torch.load(f"runs/efficientnet_b0_v3_fer2013/seed_{seed}.pth", map_location="cpu"))
    m_v3 = to_device(m_v3, device)
    m_v4 = get_model("efficientnet_b0", 7, 1, False)
    m_v4.load_state_dict(torch.load(f"runs/efficientnet_b0_v4_fer2013/seed_{seed}.pth", map_location="cpu"))
    m_v4 = to_device(m_v4, device)
    ds = FERDataset(pathlib.Path("data/fer2013"), "test", transform=eval_transform(48), dataset="fer2013")
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    probs, labels = collect_probs([m_v3, m_v4], loader, device)
    fused = weighted_average(probs, weights=[0.4, 0.6])
    acc = (fused.argmax(1) == labels).mean()
    accs_v3v4.append(acc)
    print(f"V3+V4 seed {seed}: {acc:.4f}")

print(f"V3+V4 3-seed mean {np.mean(accs_v3v4):.4f} ± {np.std(accs_v3v4, ddof=1):.4f} (seed42 0.6996, TTA 0.7042)")

# V2+V5 3-seed
accs_v2v5 = []
for seed in [42,43,44]:
    m_v2 = get_model("efficientnet_b0", 7, 1, False)
    m_v2.load_state_dict(torch.load(f"runs/efficientnet_b0_v2_fer2013/seed_{seed}.pth", map_location="cpu"))
    m_v2 = to_device(m_v2, device)
    m_v5 = get_model("efficientnet_b0", 7, 1, False)
    m_v5.load_state_dict(torch.load(f"runs/efficientnet_b0_v5_fer2013/seed_{seed}.pth", map_location="cpu"))
    m_v5 = to_device(m_v5, device)
    ds = FERDataset(pathlib.Path("data/fer2013"), "test", transform=eval_transform(48), dataset="fer2013")
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    probs, labels = collect_probs([m_v2, m_v5], loader, device)
    fused = weighted_average(probs, weights=[0.45, 0.55])
    acc = (fused.argmax(1) == labels).mean()
    accs_v2v5.append(acc)

print(f"V2+V5 3-seed mean {np.mean(accs_v2v5):.4f} ± {np.std(accs_v2v5, ddof=1):.4f} (reported 68.29±0.20%)")

# T-test V3+V4 vs V2+V5
t, p = stats.ttest_ind(accs_v3v4, accs_v2v5, equal_var=False)
print(f"\nT-test V3+V4 vs V2+V5: t={t:.3f}, p={p:.4f} ({'significant' if p<0.05 else 'not significant'} at p<0.05)")
print(f"Non-overlapping CI? V3+V4 {np.mean(accs_v3v4):.4f}±{np.std(accs_v3v4, ddof=1):.4f} vs V2+V5 {np.mean(accs_v2v5):.4f}±{np.std(accs_v2v5, ddof=1):.4f} — {'no overlap, significant' if abs(np.mean(accs_v3v4)-np.mean(accs_v2v5)) > np.std(accs_v3v4, ddof=1)+np.std(accs_v2v5, ddof=1) else 'overlap, check p'}")

# Check all single models have 3 seeds
print("\n=== All singles have 3 seeds? ===")
for name in ["efficientnet_b0_v1_fer2013", "efficientnet_b0_v2_fer2013", "shufflenetv2_fer2013"]:
    p = results_dir / f"{name}.json"
    data = json.loads(p.read_text())
    n = len(data["test"]["accuracy"]["values"])
    print(f"{name}: {n}/3 {'OK' if n==3 else 'FAIL'}")

print("\n=== Audit complete — ready for Table 1 ===")
