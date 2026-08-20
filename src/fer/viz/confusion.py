from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fer.utils.constants import EMOTION_LABELS


def confusion_matrix_figure(
    cm: np.ndarray,
    save_path: str | None = None,
    title: str = "Confusion Matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(EMOTION_LABELS.values()),
                yticklabels=list(EMOTION_LABELS.values()), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
