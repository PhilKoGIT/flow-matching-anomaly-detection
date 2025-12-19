#!/bin/bash
#SBATCH -J cont_vp
#SBATCH -p cpu_il
#SBATCH --cpus-per-task=50    
#SBATCH --mem=64G         
#SBATCH --time=70:00:00
#SBATCH -o slurm/%x-%j.out
#SBATCH -e slurm/%x-%j.err

source ~/.bashrc
conda deactivate

cd /pfs/work9/workspace/scratch/ma_pkonz-thesis/ForestDiffusion/tests
source ../.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGBOOST_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_AFFINITY=granularity=fine,compact,1,0

echo "== Job Start: $(date) =="
echo "Python: $(which python)"
echo "Imports dauern ~2-3 Minuten, bitte warten..."

python -u contamination.py

echo "== Job Ende: $(date) =="