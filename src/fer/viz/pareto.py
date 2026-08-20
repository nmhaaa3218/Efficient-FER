from __future__ import annotations

import matplotlib.pyplot as plt


def pareto_figure(
    x: list[float],
    y: list[float],
    labels: list[str],
    xlabel: str,
    ylabel: str,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=80)
    for xi, yi, l in zip(x, y, labels):
        ax.annotate(l, (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Pareto Efficiency Frontier")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
