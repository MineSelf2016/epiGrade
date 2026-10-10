#!/bin/bash
#SBATCH --job-name=epi_xcf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/xcf_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
read -r ATLAS ANAME QUERY QNAME SUB <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/manifest_xc.txt)"
echo "host=$(hostname) [$ANAME -> $QNAME] start=$(date)"
.venv/bin/python -u code/experiments/run_cross_cohort_epigrade.py \
  --atlas "data/processed/${ATLAS}.h5ad" --atlas-name "$ANAME" \
  --query "data/processed/${QUERY}.h5ad" --query-name "$QNAME" \
  --output "results_fused/xc_${ANAME}_to_${QNAME}" --atlas-subsample "$SUB"
echo "end=$(date)"
