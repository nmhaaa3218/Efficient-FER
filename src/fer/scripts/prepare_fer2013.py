from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def regenerate_fer2013_folders(csv_path: str | Path, out_root: str | Path) -> None:
    """Rebuild data/fer2013/{train,validation,test}/<class>/*.png losslessly from CSV.

    Split mapping matches the legacy project / Kaggle convention:
      Training -> train, PublicTest -> validation, PrivateTest -> test.

    WARNING: the previous data/fer2013 folders contained LOSSY .jpg re-encodings
    that corrupted accuracy (~0.67 -> ~0.40). This script regenerates exact
    48x48 grayscale PNGs from the original fer2013.csv pixels.
    """
    df = pd.read_csv(csv_path)
    out_root = Path(out_root)
    mapping = {"Training": "train", "PublicTest": "validation", "PrivateTest": "test"}
    counts = {}

    for _, row in df.iterrows():
        split = mapping[row["Usage"]]
        label = int(row["emotion"])
        pixels = np.asarray(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, mode="L")

        d = out_root / split / str(label)
        d.mkdir(parents=True, exist_ok=True)
        idx = counts.get((split, label), 0)
        img.save(d / f"{idx:06d}.png")
        counts[(split, label)] = idx + 1

    total = sum(counts.values())
    print(f"Regenerated {total} lossless PNGs into {out_root}")
    for (split, label), n in sorted(counts.items()):
        print(f"  {split}/{label}: {n}")


def main():
    p = argparse.ArgumentParser(description="Regenerate FER-2013 folder splits losslessly from CSV")
    p.add_argument("--csv", required=True, help="Path to fer2013.csv")
    p.add_argument("--out", default="data/fer2013")
    args = p.parse_args()
    regenerate_fer2013_folders(args.csv, args.out)


if __name__ == "__main__":
    main()
