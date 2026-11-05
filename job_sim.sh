#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/slurm_%x_%j.out

module load gcc python
source .venv/bin/activate
python -u simulate.py --instance_index 32 33 34 62 63 64 92 93 94 122 123 124 --runs 20 --jobs $SLURM_CPUS_PER_TASK "$@"
