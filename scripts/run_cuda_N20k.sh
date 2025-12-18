#!/bin/bash
#SBATCH --job-name=CUDA_N20k
#SBATCH --output=logs/cuda_N20k_%j.log
#SBATCH --error=logs/cuda_N20k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

# --- SETTINGS ---
N=20000
STEPS=1000
BENCH="true"
BIN="./bin/nbody_cuda"

# Load modules
module load cuda/12.0

echo "=============================================================================="
echo "                       HPC N-BODY SIMULATION REPORT                       "
echo "=============================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Description : CUDA (GPU Acceleration)"
echo " Start Time  : $(date)"
echo " Node        : $(hostname)"
echo "=============================================================================="
echo "                       GPU ARCHITECTURE SNAPSHOT                              "
echo "=============================================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "=============================================================================="
echo "                           SIMULATION EXECUTION                               "
echo "=============================================================================="

if [ -f "$BIN" ]; then
    echo "Running: $BIN $N $STEPS $BENCH"
    time $BIN $N $STEPS $BENCH
else
    echo "ERROR: Binary not found at $BIN"
fi

echo "=============================================================================="
echo " End Time    : $(date)"
echo "=============================================================================="
