#!/bin/bash
#SBATCH --job-name=epi_fused
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fusedq_%A_%a.out
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
jobs/run_fused_one.sh $(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/m_fused.txt)
