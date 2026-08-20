from __future__ import annotations

import random

import torchvision.transforms as T
from PIL import Image

from fer.data.transforms import train_transform
from fer.config import AugConfig


class HybridTransform:
    """Per-sample hybrid of V2 (CRP) and V5-light (flip+resize).

    For each __call__, randomly picks V2 (p=0.5) or V5-light (p=0.5).
    Batch-level MixUp/CutMix is handled by Trainer (if aug.variant == 'hybrid',
    Trainer will apply CutMix to the whole batch as for V5).
    """

    def __init__(self, aug: AugConfig | None = None, p_v2: float = 0.5):
        self.p_v2 = p_v2
        aug = aug or AugConfig()
        # V2 branch: CRP
        self.v2 = train_transform("v2", aug)
        # V5-light branch: flip + resize (no CRP, MixUp at batch level)
        self.v5 = train_transform("v5", aug)

    def __call__(self, img: Image.Image):
        if random.random() < self.p_v2:
            return self.v2(img)
        return self.v5(img)
