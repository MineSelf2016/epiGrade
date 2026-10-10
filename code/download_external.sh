#!/bin/bash
set -u
RAW="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
GEO=https://ftp.ncbi.nlm.nih.gov/geo
get(){ mkdir -p "$2"; f="$2/$(basename "$1")"
  if [ -s "$f" ]; then echo "[skip] $(basename "$1")"; return 0; fi
  echo "[get ] $(basename "$1")"
  curl -fsSL --retry 5 --retry-delay 5 -C - -o "$f" "$1" || { echo "[FAIL] $1"; return 1; }; }

S=$GEO/series/GSE155nnn/GSE155673/suppl
get $S/GSE155673_features.tsv.gz $RAW/covid_pbmc
for d in cov01 cov02 cov03 cov04 cov07 cov08 cov09 cov10 cov11 cov12 cov17 cov18; do
  get $S/GSE155673_${d}_barcodes.tsv.gz $RAW/covid_pbmc
  get $S/GSE155673_${d}_matrix.mtx.gz   $RAW/covid_pbmc
done

get $GEO/samples/GSM3681nnn/GSM3681518/suppl/GSM3681518_MNC_RNA_counts.tsv.gz $RAW/bonemarrow
get $GEO/samples/GSM3681nnn/GSM3681519/suppl/GSM3681519_MNC_ADT_counts.tsv.gz $RAW/bonemarrow

echo "=== DONE ==="; du -sh $RAW/* 2>/dev/null
