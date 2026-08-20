#!/usr/bin/env bash
# fetch_datasets.sh
# Download official FER-2013 / FER+ sources.
#
# FER-2013: original Kaggle CSV (needs manual download from Kaggle):
#   https://www.kaggle.com/datasets/deadskull7/fer2013
#   Place at: data/_sources/fer2013.csv
#
# FER+ labels: official Microsoft GitHub (downloads automatically):
set -euo pipefail

mkdir -p data/_sources

echo "== FER+ labels (official microsoft/FERPlus) =="
if [ ! -f data/_sources/fer2013new.csv ]; then
  curl -sL -o data/_sources/fer2013new.csv \
    "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
  echo "Downloaded fer2013new.csv"
else
  echo "Already present"
fi

if [ ! -f data/_sources/fer2013.csv ]; then
  echo
  echo "== FER-2013 (manual) =="
  echo "Download fer2013.csv from Kaggle (requires login):"
  echo "  https://www.kaggle.com/datasets/deadskull7/fer2013"
  echo "Then place it at: data/_sources/fer2013.csv"
else
  echo "fer2013.csv already present"
fi

echo
echo "Next steps:"
echo "  python -m fer.scripts.prepare_fer2013 --csv data/_sources/fer2013.csv --out data/fer2013"
echo "  python -m fer.scripts.prepare_ferplus --fer data/_sources/fer2013.csv --ferplus data/_sources/fer2013new.csv --out data/ferplus"
