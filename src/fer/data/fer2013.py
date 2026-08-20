from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def read_fer2013_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read Kaggle FER-2013 fer2013.csv into a DataFrame."""
    return pd.read_csv(csv_path)


def parse_fer2013_pixels(pixel_str: str) -> np.ndarray:
    """'231 212 ...' -> 48x48 uint8 array."""
    arr = np.asarray(pixel_str.split(), dtype=np.uint8)
    return arr.reshape(48, 48)


def load_image_from_df(row: pd.Series) -> Image.Image:
    img = parse_fer2013_pixels(row["pixels"])
    return Image.fromarray(img, mode="L")
