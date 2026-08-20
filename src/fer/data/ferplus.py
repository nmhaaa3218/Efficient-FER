from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from fer.utils.constants import FERPLUS_EMOTIONS, FERPLUS_TO_7

SPLIT_MAP = {"Training": "train", "PublicTest": "validation", "PrivateTest": "test"}


def hard_soft_7(vote8: np.ndarray) -> tuple[int, np.ndarray]:
    """Convert 8-class FER+ vote vector to (7-class hard label, 7-class soft prob)."""
    hard8 = int(vote8.argmax())
    hard7 = FERPLUS_TO_7[FERPLUS_EMOTIONS[hard8]]
    soft7 = np.zeros(7, dtype=float)
    for e, v in zip(FERPLUS_EMOTIONS, vote8):
        soft7[FERPLUS_TO_7[e]] += v
    s = soft7.sum()
    if s > 0:
        soft7 = soft7 / s
    return hard7, soft7


def read_votes(ferplus_csv: str | Path) -> pd.DataFrame:
    """Read fer2013new.csv, returning just the 8 emotion vote columns."""
    df = pd.read_csv(ferplus_csv)
    return df[FERPLUS_EMOTIONS].fillna(0.0)
