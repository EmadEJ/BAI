#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/slurm_%x_%j.out

module load gcc python
source .venv/bin/activate
python -u simulate.py --instance_index 42 44 46 48 82 84 86 88 122 124 126 128 162 164 166 168 202 204 206 208 --runs 20 --jobs $SLURM_CPUS_PER_TASK "$@"
