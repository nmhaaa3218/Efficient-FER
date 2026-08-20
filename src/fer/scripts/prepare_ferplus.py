from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# FERPlus 10-annotator columns (fer2013new.csv) -> 8 semantic emotions
FERPLUS_EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

# Map FERPlus 8-emotion probabilities down to the 7-class FER-2013 schema
FERPLUS_TO_7 = {
    "neutral": 6,
    "happiness": 3,
    "surprise": 5,
    "sadness": 4,
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "contempt": 6,
}

SPLIT_MAP = {"Training": "train", "PublicTest": "validation", "PrivateTest": "test"}


def build_ferplus_dataset(fer_csv: str | Path, ferplus_csv: str | Path, out_root: str | Path) -> dict:
    """Build lossless FER+ folder splits (7-class) from fer2013.csv + fer2013new.csv.

    Row order in both CSVs is aligned (Microsoft's merge convention):
      - fer2013.csv provides image pixels + original usage split
      - fer2013new.csv provides 10-annotator vote labels for the SAME images

    Output layout: out_root/{train,validation,test}/{0..6}/<file>.png (lossless)
    Labels: 8 emotion votes (contempt folds into neutral) -> 7-class hard label.

    Soft labels are saved per-split in LOAD ORDER (sorted class folders,
    sequential filenames) so FERDataset can directly index them.
    """
    fer = pd.read_csv(fer_csv)
    ferplus = pd.read_csv(ferplus_csv)
    assert len(fer) == len(ferplus), (
        f"Row count mismatch: fer2013.csv={len(fer)} vs fer2013new.csv={len(ferplus)}"
    )

    votes = ferplus[FERPLUS_EMOTIONS].fillna(0.0).to_numpy(dtype=float)
    out_root = Path(out_root)
    counts: Counter = Counter()
    per_split_class_soft: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    for i, (_, row) in enumerate(fer.iterrows()):
        split = SPLIT_MAP[row["Usage"]]
        pixels = np.asarray(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, mode="L")

        vote8 = votes[i]
        hard8 = int(vote8.argmax())
        hard7 = FERPLUS_TO_7[FERPLUS_EMOTIONS[hard8]]

        soft7 = np.zeros(7)
        for e, v in zip(FERPLUS_EMOTIONS, vote8):
            soft7[FERPLUS_TO_7[e]] += v
        s = soft7.sum()
        if s > 0:
            soft7 = soft7 / s

        d = out_root / split / str(hard7)
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / f"{counts[(split, hard7)]:06d}.png", compress_level=0)

        counts[(split, hard7)] += 1
        per_split_class_soft[(split, hard7)].append(soft7)

    # Save soft labels per-split in LOAD ORDER (sorted class, sequential files)
    # so FERDataset soft mode can directly index without CSV reordering.
    soft_per_split: dict[str, np.ndarray] = {}
    for split in ["train", "validation", "test"]:
        ordered: list[np.ndarray] = []
        for hard in sorted(range(7)):
            ordered.extend(per_split_class_soft[(split, hard)])
        soft_per_split[split] = np.stack(ordered) if ordered else np.zeros((0, 7))

    label_npz = out_root / "labels.npz"
    np.savez_compressed(
        label_npz,
        train_soft_7=soft_per_split["train"],
        validation_soft_7=soft_per_split["validation"],
        test_soft_7=soft_per_split["test"],
    )

    total = sum(counts.values())
    print(f"Built FER+ dataset: {total} images into {out_root}")
    print(f"Saved label arrays -> {label_npz}")
    for (split, label), n in sorted(counts.items()):
        print(f"  {split}/{label}: {n}")
    return {"total": total}


def main():
    p = argparse.ArgumentParser(
        description="Build lossless FER+ folder splits (7-class) from fer2013.csv + fer2013new.csv"
    )
    p.add_argument("--fer", required=True, help="Path to fer2013.csv (images)")
    p.add_argument("--ferplus", required=True, help="Path to fer2013new.csv (FER+ labels)")
    p.add_argument("--out", default="data/ferplus")
    args = p.parse_args()
    build_ferplus_dataset(args.fer, args.ferplus, args.out)


if __name__ == "__main__":
    main()
