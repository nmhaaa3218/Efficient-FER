from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fer.data.fer2013 import parse_fer2013_pixels


class FERDataset(Dataset):
    """Folder-split FER-2013 / FERPlus dataset.

    Expects data/<dataset>/<train|validation|test>/<class>/image.png
    (legacy ImageFolder layout) OR a csv with 'pixels' column.

    FERPlus soft labels: if data/<dataset>/labels.npz exists, rows are aligned
    with fer2013.csv order (Training -> PublicTest -> PrivateTest) and soft_7
    probabilities are returned when label_mode='soft'.
    """

    def __init__(self, root: str | Path, split: str, label_mode: str = "hard", transform=None, dataset: str = "fer2013"):
        self.root = Path(root)
        self.split = split
        self.label_mode = label_mode
        self.transform = transform
        self.dataset = dataset
        self.samples: list[tuple[str, int]] = []
        self.soft_labels: np.ndarray | None = None

        split_dir = self.root / split
        if split_dir.is_dir():
            self._load_from_folders(split_dir)
        else:
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        if self.label_mode == "soft":
            self._load_soft_labels()

    def _load_from_folders(self, split_dir: Path):
        classes = sorted(p for p in split_dir.iterdir() if p.is_dir())
        for label, class_dir in enumerate(classes):
            for img_path in sorted(class_dir.iterdir()):
                self.samples.append((str(img_path), label))

    def _load_soft_labels(self):
        npz_path = self.root / "labels.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Soft labels requested but {npz_path} missing")
        data = np.load(npz_path)
        # New format: per-split keys (train_soft_7, validation_soft_7, test_soft_7) in load order
        # Fallback to legacy single soft_7 with SPLIT_RANGES for backwards compat
        key = f"{self.split}_soft_7"
        if key in data:
            self.soft_labels = data[key].astype(np.float32)
        else:
            # Legacy: single soft_7 in CSV order
            legacy_ranges = {
                "ferplus": {"train": (0, 28709), "validation": (28709, 32298), "test": (32298, 35887)},
            }
            start, end = legacy_ranges[self.dataset][self.split]
            self.soft_labels = data["soft_7"][start:end].astype(np.float32)
        assert len(self.soft_labels) == len(self.samples), (
            f"Soft label count mismatch: {len(self.soft_labels)} vs {len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        from PIL import Image

        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        if self.label_mode == "soft" and self.soft_labels is not None:
            return image, torch.from_numpy(self.soft_labels[idx])
        return image, torch.tensor(label, dtype=torch.long)
