"""RAF-DB 48x48 1ch lossless preparation — mirrors fer2013.py.

RAF-DB original: 12,271 train / 3,068 test, 100x100 RGB face crops, 7 classes (same as FER-2013).
We convert to 48x48 L (grayscale) lossless PNG, same pipeline as FER-2013 for fair 48 1ch comparison.
No AffectNet pre-training, from scratch.

Label remap (RAF-DB 1..7) -> FER-2013 7-class index (matches fer.utils.constants.EMOTION_LABELS):
    1:Surprise -> 5,  2:Fear -> 2,  3:Disgust -> 1,  4:Happiness -> 3,
    5:Sadness -> 4,    6:Anger -> 0,   7:Neutral -> 6

Validation split: 10% of train (1227 samples) carved with fixed seed=42.
Val indices persisted to <out>/val_split_indices.txt for audit/repro.

Output layout (matches FER-2013 ImageFolder; FERDataset works as-is):
    <out>/{train,validation,test}/{0..6}/NNNNNN.png  (48x48 L PNG, compress_level=0)

Usage:
    python -m fer.scripts.prepare_rafdb --in data/rafdb --out data/rafdb
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# RAF-DB raw labels 1..7 -> FER-2013 7-class index
RAFDB_TO_FER7: dict[int, int] = {
    1: 5,  # Surprise
    2: 2,  # Fear
    3: 1,  # Disgust
    4: 3,  # Happiness
    5: 4,  # Sadness
    6: 0,  # Anger
    7: 6,  # Neutral
}

VAL_FRAC = 0.10
VAL_SEED = 42


def parse_list_file(list_path: Path, aligned_dir: Path) -> pd.DataFrame:
    """Parse list_patition_label.txt -> DataFrame [aligned_name, rafdb_label, fer7_label].

    The label file references original names (train_00001.jpg), but aligned/
    contains the _aligned.jpg versions. Resolve to aligned path here.
    """
    rows: list[dict] = []
    with list_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, lab = line.split()
            raf_lab = int(lab)
            if raf_lab not in RAFDB_TO_FER7:
                raise ValueError(f"Unknown RAF-DB label {raf_lab} for {name}")
            fer_lab = RAFDB_TO_FER7[raf_lab]
            stem = name[:-4]  # strip .jpg
            aligned_name = f"{stem}_aligned.jpg"
            if not (aligned_dir / aligned_name).exists():
                raise FileNotFoundError(f"Missing aligned image: {aligned_dir / aligned_name}")
            rows.append({"aligned_name": aligned_name, "rafdb_label": raf_lab, "fer7_label": fer_lab})
    return pd.DataFrame(rows)


def prepare_rafdb(in_dir: Path, out_dir: Path, size: int = 48) -> dict:
    """Build lossless RAF-DB {size}x{size} L folder splits (7-class).

    Reads in_dir/aligned/*.jpg + in_dir/list_patition_label.txt.
    Writes out_dir/{train,validation,test}/{0..6}/NNNNNN.png.
    Validation = 10% of train (1227), fixed seed=42.

    Idempotent: deterministic naming + resampling overwrites identical bytes on re-run.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    aligned_dir = in_dir / "aligned"
    list_path = in_dir / "list_patition_label.txt"

    if not aligned_dir.is_dir():
        raise FileNotFoundError(f"aligned/ not found at {aligned_dir}")
    if not list_path.exists():
        raise FileNotFoundError(f"list_patition_label.txt not found at {list_path}")

    df = parse_list_file(list_path, aligned_dir)

    # Native split via filename prefix (train_*.jpg vs test_*.jpg)
    is_train = df["aligned_name"].str.startswith("train_")
    df_train = df[is_train].reset_index(drop=True)
    df_test = df[~is_train].reset_index(drop=True)
    if len(df_train) != 12271:
        raise RuntimeError(f"Expected 12271 train, got {len(df_train)}")
    if len(df_test) != 3068:
        raise RuntimeError(f"Expected 3068 test, got {len(df_test)}")

    # Carve 10% from train as validation (fixed seed for reproducibility)
    rng = np.random.RandomState(VAL_SEED)
    perm = rng.permutation(len(df_train))
    n_val = int(round(len(df_train) * VAL_FRAC))  # 1227
    val_idx = sorted(perm[:n_val].tolist())
    val_set = set(val_idx)
    is_val = np.array([i in val_set for i in range(len(df_train))])
    df_val = df_train[is_val].reset_index(drop=True)
    df_train = df_train[~is_val].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_split_indices.txt").write_text(
        "# RAF-DB val carve indices into original train list (0-indexed, sorted)\n"
        f"# seed={VAL_SEED}, frac={VAL_FRAC}, n_val={n_val}\n"
        + "\n".join(str(i) for i in val_idx)
        + "\n"
    )

    counts: Counter = Counter()

    def save_split(split_df: pd.DataFrame, split_name: str) -> None:
        for _, row in split_df.iterrows():
            fer_lab = int(row["fer7_label"])
            src = aligned_dir / row["aligned_name"]
            d = out_dir / split_name / str(fer_lab)
            d.mkdir(parents=True, exist_ok=True)
            idx = counts[(split_name, fer_lab)]
            img = Image.open(src).convert("L").resize((size, size), Image.Resampling.LANCZOS)
            img.save(d / f"{idx:06d}.png", compress_level=0)
            counts[(split_name, fer_lab)] = idx + 1

    save_split(df_train, "train")
    save_split(df_val, "validation")
    save_split(df_test, "test")

    total = sum(counts.values())
    print(f"Built RAF-DB dataset: {total} images into {out_dir}")
    for (split, label), n in sorted(counts.items()):
        print(f"  {split}/{label}: {n}")
    return {"total": total, "train": len(df_train), "validation": len(df_val), "test": len(df_test)}