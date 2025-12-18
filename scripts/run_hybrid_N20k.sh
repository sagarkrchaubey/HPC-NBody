#!/bin/bash
#SBATCH --job-name=Hybrid_N20k
#SBATCH --output=logs/hybrid_N20k_%j.log
#SBATCH --error=logs/hybrid_N20k_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

# --- SETTINGS ---
N=20000
STEPS=1000
BENCH="true"
BIN="./bin/nbody_hybrid"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Load modules
module load openmpi/4.1.1

echo "=============================================================================="
echo "                       HPC N-BODY SIMULATION REPORT                       "
echo "=============================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Description : Hybrid (MPI + OpenMP)"
echo " Resources   : $SLURM_JOB_NUM_NODES Nodes x $SLURM_CPUS_PER_TASK Threads = $(($SLURM_JOB_NUM_NODES * $SLURM_CPUS_PER_TASK)) Total Cores"
echo " Start Time  : $(date)"
echo " Master Node : $(hostname)"
echo "=============================================================================="
echo "                           SIMULATION EXECUTION                               "
echo "=============================================================================="

if [ -f "$BIN" ]; then
    echo "Command: mpirun -np $SLURM_NTASKS $BIN $N $STEPS $BENCH"
    time mpirun -np $SLURM_NTASKS $BIN $N $STEPS $BENCH
else
    echo "ERROR: Binary not found at $BIN"
fi

echo "=============================================================================="
echo " End Time    : $(date)"
echo "=============================================================================="
