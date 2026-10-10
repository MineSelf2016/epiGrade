#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs
for pair in "eccite_5p ECCITE" "gse3p_3p GSE3P"; do
  set -- $pair
  OMP_NUM_THREADS=4 .venv/bin/python -u code/experiments/run_baselines.py --data data/processed/$1.h5ad \
    --dataset-name $2 --reference-run results_fused/donor5/$2_donor_full --output results_fused/donor5/knn30/$2_donor \
    --config code/configs/fused.yaml --only knn_expression knn_pca > logs/knn30_donor5_$2.out 2>&1
done
