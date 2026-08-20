from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from fer.utils.constants import EMOTION_LABELS


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def save_fig(fig, name: str, out_dir: str | Path = "figures", eps: bool = True, tiff: bool = False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    if tiff:
        fig.savefig(out / f"{name}.tiff", bbox_inches="tight", format="tiff", dpi=300)
    if eps:
        fig.savefig(out / f"{name}.eps", bbox_inches="tight", format="eps")
    plt.close(fig)
