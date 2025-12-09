#!/bin/bash
#SBATCH -J contamination
#SBATCH -p gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o slurm/%x-%j.out
#SBATCH -e slurm/%x-%j.err


# --- Environment laden ---
source ~/.bashrc
source .venv/bin/activate

# --- In dein Projektverzeichnis wechseln ---
cd /pfs/work9/workspace/scratch/ma_pkonz-thesis/ForestDiffusion/tests

# --- Startmeldung ---0=/*66
echo "== Job start =="

# --- Python Script starten (ForestDiffusion Training) ---
python contamination.py

# --- Ende ---
echo "== Job abgeschlossen =="
