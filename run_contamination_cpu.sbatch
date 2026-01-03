#!/bin/bash
#SBATCH -J cont_vp
#SBATCH -p cpu_il
#SBATCH --cpus-per-task=50    
#SBATCH --mem=64G         
#SBATCH --time=70:00:00
#SBATCH -o slurm/%x-%j.out
#SBATCH -e slurm/%x-%j.err

source ~/.bashrc
conda deactivate 2>/dev/null

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd tests
echo "== Job Start: $(date) =="
python -u contamination.py
echo "== Job Ende: $(date) =="