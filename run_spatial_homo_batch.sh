#!/bin/bash

#SBATCH --job-name=spatial_sim
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mlu@zib.de

#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --time=60:00:00

## 25 parallel trajectories (indices 0-24)
#SBATCH --array=0-50

#SBATCH --output=logs/spatial_%A_%a.out
#SBATCH --error=logs/spatial_%A_%a.err

# --- Environment ---
source ~/schloegl_env/bin/activate

# --- Run ---
cd ~/bistable_model_simulation
mkdir -p logs

echo "Task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"

srun python3 -m scripts.runners.run_spatial_homo

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
