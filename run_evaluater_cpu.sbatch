#!/bin/bash
#
# --- Slurm Konfiguration für CPU-Job ---
#
#SBATCH -J evaluater_cpu   # Name des Jobs (angepasst)
#SBATCH -p cpu_il              # NEUE PARTITION: CPU Ice Lake (72h Limit)
#SBATCH --cpus-per-task=16      # CPUs anfordern (z.B. 4)
#SBATCH --mem=32G              # Speicher anfordern (z.B. 16GB, oft reicht weniger ohne GPU)
#SBATCH --time=10:00:00        # Zeit anpassen (z.B. 2 Stunden, um auf der sicheren Seite zu sein)
#SBATCH -o slurm/%x-%j.out
#SBATCH -e slurm/%x-%j.err


# --- Environment laden ---
source ~/.bashrc
source .venv/bin/activate

# --- In dein Projektverzeichnis wechseln ---
cd /pfs/work9/workspace/scratch/ma_pkonz-thesis/ForestDiffusion/tests

# --- Startmeldung ---
echo "== Job start (CPU-Partition: cpu_il) =="
echo "Python Script wird mit $(nproc) Kernen gestartet..."

# --- Python Script starten ---
python evaluater.py

# --- Ende ---
echo "== Job abgeschlossen =="