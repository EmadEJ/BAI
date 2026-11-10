#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/slurm_%x_%j.out

module load gcc python
source .venv/bin/activate
JOBS=$SLURM_CPUS_PER_TASK ./run_all.sh "$@"
