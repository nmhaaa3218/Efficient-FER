from __future__ import annotations

import argparse

from fer.viz.style import apply_style


def main():
    p = argparse.ArgumentParser(description="Regenerate all publication figures")
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    apply_style()
    print(f"Figures will be written to {args.out} (see fer/scripts/figures.py for per-figure scripts)")


if __name__ == "__main__":
    main()
