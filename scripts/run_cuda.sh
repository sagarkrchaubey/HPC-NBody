#!/bin/bash
#SBATCH --job-name=run_cuda
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/run_cuda_%j.log
#SBATCH --error=logs/errors/run_cuda_%j.err

N=${1:-20000}
STEPS=${2:-1000}
MODE=${3:-bench}
SAVE=${4:-1}
BIN="./bin/nbody_cuda"

module purge
module load spack
. /home/apps/spack/share/spack/setup-env.sh
spack find gcc
spack load gcc@11.2.0
module spider cuda
module load cuda/12.0 

echo "=============================================================================="
echo "                       HPC N-BODY SIMULATION REPORT                       "
echo "=============================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Description : CUDA Standard (V100 GPU)"
echo " Start Time  : $(date)"
echo " Node        : $(hostname)"
echo "=============================================================================="
echo "                       HARDWARE ARCHITECTURE SNAPSHOT                         "
echo "=============================================================================="
lscpu | grep -E "Model name|Socket|Thread|NUMA|MHz"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "=============================================================================="
echo "                           SIMULATION EXECUTION                               "
echo "=============================================================================="

if [ -f "$BIN" ]; then
    echo "Running: $BIN $N $STEPS $MODE $SAVE"
    time $BIN $N $STEPS $MODE $SAVE
else
    echo "ERROR: Binary not found at $BIN"
fi

echo "=============================================================================="
echo " End Time    : $(date)"
echo "=============================================================================="
