from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from fer.config import Config
from fer.data.loaders import build_loaders
from fer.models.factory import get_model
from fer.profile.flops import compute_flops
from fer.profile.latency import export_onnx, measure_latency
from fer.training.metrics import compute_metrics
from fer.utils.device import get_device, to_device


def onnx_latency(model, image_size: int, in_channels: int, iterations: int) -> dict:
    import numpy as np
    import onnxruntime as ort

    model_cpu = model.to("cpu")
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        export_onnx(model_cpu, f.name, image_size=image_size, in_channels=in_channels)
        path = f.name
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.zeros((1, in_channels, image_size, image_size), dtype=np.float32)
    for _ in range(min(20, iterations)):
        sess.run(None, {"input": x})
    timings = []
    for _ in range(iterations):
        t0 = __import__("time").perf_counter()
        sess.run(None, {"input": x})
        timings.append(__import__("time").perf_counter() - t0)
    ms = np.asarray(timings) * 1000.0
    return {"onnx_mean_ms": round(float(ms.mean()), 3), "onnx_fps": round(float(1000.0 / ms.mean()), 2)}


def main():
    p = argparse.ArgumentParser(description="FLOPs + latency benchmark for all models")
    p.add_argument("--models", nargs="+", default=["efficientnet_b0", "mobilenetv3_small", "shufflenetv2_0_5x"])
    p.add_argument("--out", default="results/benchmark.csv")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--device", default="auto", help="cpu | mps | cuda")
    p.add_argument("--onnx", action="store_true", help="Also measure ONNX Runtime CPU latency")
    p.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="Optional: paths to trained .pth files. Each is loaded into the matching "
             "--models entry (positional), evaluated on test set, and Acc/F1 columns added.",
    )
    p.add_argument("--config", default=None, help="Config YAML used for --checkpoints evaluation (test set)")
    args = p.parse_args()

    device = get_device(args.device)
    rows = []
    for i, name in enumerate(args.models):
        model = get_model(name, num_classes=7, in_channels=1, pretrained=False).eval()
        flops = compute_flops(model, image_size=48, in_channels=1)
        lat = measure_latency(model, device, image_size=48, in_channels=1, iterations=args.iterations)
        row = {
            "model": name,
            "params_m": round(flops["params_m"], 3),
            "gflops": round(flops["gflops"], 4),
            f"{device.type}_mean_ms": round(lat["mean_ms"], 3),
            f"{device.type}_fps": round(lat["fps"], 2),
        }
        if args.onnx:
            row.update(onnx_latency(model, image_size=48, in_channels=1, iterations=args.iterations))

        if args.checkpoints and i < len(args.checkpoints):
            ckpt = args.checkpoints[i]
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
            model = model.to(device)
            if args.config is None:
                raise ValueError("--config is required when --checkpoints is set")
            cfg = Config.from_yaml(args.config)
            test_loader = build_loaders(cfg, device, "test")
            preds, labels = [], []
            with torch.no_grad():
                for img, lab in test_loader:
                    out = model(img.to(device))
                    if out.dim() == 1:
                        out = out.unsqueeze(0)
                    preds.extend(out.argmax(1).cpu().numpy())
                    labels.extend(lab.numpy())
            m = compute_metrics(labels, preds, cfg.model.num_classes)
            row.update(
                {
                    "ckpt": str(ckpt),
                    "test_acc": round(m["accuracy"], 4),
                    "test_macro_f1": round(m["macro_f1"], 4),
                    "test_balanced_acc": round(m["balanced_accuracy"], 4),
                }
            )
        rows.append(row)
        print(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
