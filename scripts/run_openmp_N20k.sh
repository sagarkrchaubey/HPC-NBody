#!/bin/bash
#SBATCH --job-name=OpenMP_N20k
#SBATCH --output=logs/openmp_N20k_%j.log
#SBATCH --error=logs/openmp_N20k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

# --- SETTINGS ---
N=20000
STEPS=1000
BENCH="true"
BIN="./bin/nbody_openmp"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=============================================================================="
echo "                       HPC N-BODY SIMULATION REPORT                       "
echo "=============================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Description : OpenMP (Shared Memory)"
echo " Resources   : 1 Node x $SLURM_CPUS_PER_TASK Cores"
echo " Start Time  : $(date)"
echo " Node        : $(hostname)"
echo "=============================================================================="
echo "                       HARDWARE ARCHITECTURE SNAPSHOT                         "
echo "=============================================================================="
lscpu | grep -E "Model name|Socket|Thread|NUMA|MHz"
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
