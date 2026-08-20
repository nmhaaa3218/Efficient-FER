"""CLI: python -m fer.scripts.prepare_rafdb --in data/_sources/rafdb --out data/rafdb"""
from __future__ import annotations
import argparse
from pathlib import Path
from fer.data.rafdb import prepare_rafdb

def main():
    p = argparse.ArgumentParser(description="Prepare RAF-DB 48 1ch lossless")
    p.add_argument("--in", dest="inp", required=True, help="Input RAF-DB root with 100x100 aligns")
    p.add_argument("--out", required=True, help="Output data/rafdb")
    p.add_argument("--size", type=int, default=48)
    args = p.parse_args()
    prepare_rafdb(Path(args.inp), Path(args.out), size=args.size)
    print(f"RAF-DB prepared at {args.out} 48x48 L lossless")

if __name__ == "__main__":
    main()
