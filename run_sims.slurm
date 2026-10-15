#!/bin/bash

#SBATCH --job-name=fisher_sim
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output=slurm_fisher_sim_%j.out
#SBATCH --error=slurm_fisher_sim_%j.err

set -euo pipefail

# Environnement logici
source ~/miniforge3/etc/profile.d/conda.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jaxds

# Racine du projet : dossier depuis lequel sbatch est lance
cd "$SLURM_SUBMIT_DIR"

# Permet a Python de trouver le package src/
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=============================================="
echo "TEST SLURM"
echo "Job ID              : $SLURM_JOB_ID"
echo "Machine             : $(hostname)"
echo "Dossier             : $(pwd)"
echo "CPUs demandes       : $SLURM_CPUS_PER_TASK"
echo "Python              : $(which python)"
echo "=============================================="

python -c "import jax; print('JAX:', jax.__version__); print('Devices:', jax.devices())"

python run_simus.py
