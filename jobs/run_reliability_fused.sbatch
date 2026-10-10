#!/bin/bash
#SBATCH --job-name=epi_rel
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rel_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
.venv/bin/python -u code/experiments/protein_reliability.py
