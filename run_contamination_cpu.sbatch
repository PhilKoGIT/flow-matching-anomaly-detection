#!/bin/bash
#SBATCH -J cont_vp
#SBATCH -p cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH -o slurm/%x-%j.out
#SBATCH -e slurm/%x-%j.err

source ~/.bashrc
conda deactivate

cd /pfs/work9/workspace/scratch/ma_pkonz-thesis/ForestDiffusion/tests
source ../.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "== Job Start: $(date) =="
echo "Python: $(which python)"
echo "Imports dauern ~2-3 Minuten, bitte warten..."

python -u contamination.py

echo "== Job Ende: $(date) =="