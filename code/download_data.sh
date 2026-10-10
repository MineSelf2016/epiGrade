#!/bin/bash
set -u
RAW="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
GEO=https://ftp.ncbi.nlm.nih.gov/geo
get(){
  mkdir -p "$2"; f="$2/$(basename "$1")"
  if [ -s "$f" ]; then echo "[skip] $(basename "$1")"; return 0; fi
  echo "[get ] $(basename "$1")"
  curl -fsSL --retry 5 --retry-delay 5 -C - -o "$f" "$1" || { echo "[FAIL] $1"; return 1; }
}

S=$GEO/series/GSE100nnn/GSE100866/suppl
get $S/GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz  $RAW/cbmc
get $S/GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz  $RAW/cbmc

S=$GEO/samples/GSM5008nnn
for x in barcodes.tsv features.tsv matrix.mtx; do
  get $S/GSM5008740/suppl/GSM5008740_RNA_5P-$x.gz $RAW/eccite_5p
  get $S/GSM5008741/suppl/GSM5008741_ADT_5P-$x.gz $RAW/eccite_5p
done

for x in barcodes.tsv features.tsv matrix.mtx; do
  get $S/GSM5008737/suppl/GSM5008737_RNA_3P-$x.gz $RAW/gse3p_3p
  get $S/GSM5008738/suppl/GSM5008738_ADT_3P-$x.gz $RAW/gse3p_3p
done

S=$GEO/series/GSE164nnn/GSE164378/suppl
get $S/GSE164378_sc.meta.data_3P.csv.gz $RAW/meta
get $S/GSE164378_sc.meta.data_5P.csv.gz $RAW/meta

echo "=== DONE ==="; du -sh $RAW/* 2>/dev/null
