#!/bin/bash
#SBATCH --job-name=epi_annf
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ann_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/m_annot.txt)
read -r FILE NAME SPLIT <<< "$LINE"
echo "host=$(hostname) [$LINE] start=$(date)"
.venv/bin/python -u code/experiments/run_annotation.py \
  --data "data/processed/${FILE}.h5ad" \
  --run "results_fused/${NAME}_${SPLIT}_full" \
  --output "results_fused/${NAME}_${SPLIT}_annotation"
echo "end=$(date)"
