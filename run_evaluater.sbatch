#!/bin/bash
#SBATCH -J evaluater
#SBATCH -p gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -o slurm/%x-%j.out
#SBATCH -e slurm/%x-%j.err


# --- Environment laden ---
source ~/.bashrc
source .venv/bin/activate

# --- In dein Projektverzeichnis wechseln ---
cd /pfs/work9/workspace/scratch/ma_pkonz-thesis/ForestDiffusion/tests

# --- Startmeldung ---
echo "== Job start =="

# --- Python Script starten (ForestDiffusion Training) ---
python evaluater.py

# --- Ende ---
echo "== Job abgeschlossen =="
