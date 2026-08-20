from __future__ import annotations

import torchvision.transforms as T

from fer.config import AugConfig

# FER-2013 grayscale normalization (matches legacy project: [-1, 1] range)
MEAN = (0.5,)
STD = (0.5,)


def _base(variant: str, aug: AugConfig) -> list:
    """Geometric + photometric transform list, mirroring legacy V1-V4."""

    def pil_list():
        return [T.Grayscale(num_output_channels=1)]

    if variant == "v1":
        t = pil_list()
        t.append(T.Resize((aug.image_size, aug.image_size)))
    elif variant == "v2":  # Crop + Rotate + Flip (CRP)
        t = pil_list()
        t += [
            T.RandomHorizontalFlip(p=aug.flip_prob),
            T.RandomCrop(aug.image_size, padding=aug.crop_padding),
            T.RandomRotation(degrees=aug.rotation_deg),
        ]
    elif variant == "v3":  # CRP + brightness/contrast
        t = pil_list()
        t += [
            T.RandomHorizontalFlip(p=aug.flip_prob),
            T.RandomCrop(aug.image_size, padding=aug.crop_padding),
            T.RandomRotation(degrees=aug.rotation_deg),
            T.ColorJitter(brightness=aug.color_brightness, contrast=aug.color_contrast),
        ]
    elif variant == "v4":  # CRP + cutout
        t = pil_list()
        t += [
            T.RandomHorizontalFlip(p=aug.flip_prob),
            T.RandomCrop(aug.image_size, padding=aug.crop_padding),
            T.RandomRotation(degrees=aug.rotation_deg),
        ]
        # RandomErasing is tensor-level, must be after ToTensor
        t += [T.ToTensor(), T.RandomErasing(p=0.5, scale=(0.02, 0.15)), T.Normalize(MEAN, STD)]
        return t
    elif variant == "v5":  # MixUp/CutMix handled at batch level; spatial augs kept light
        t = pil_list()
        t += [T.RandomHorizontalFlip(p=aug.flip_prob), T.Resize((aug.image_size, aug.image_size))]
    else:
        raise ValueError(f"Unknown augmentation variant '{variant}'")
    t += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return t


def train_transform(variant: str, aug: AugConfig | None = None) -> T.Compose:
    aug = aug or AugConfig(variant=variant)
    return T.Compose(_base(variant, aug))


def eval_transform(image_size: int = 48) -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
