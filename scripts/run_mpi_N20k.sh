#!/bin/bash
#SBATCH --job-name=MPI_N20k
#SBATCH --output=logs/mpi_N20k_%j.log
#SBATCH --error=logs/mpi_N20k_%j.err
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

# --- SETTINGS ---
N=20000
STEPS=1000
BENCH="true"
BIN="./bin/nbody_mpi"

# Load modules if strictly required at runtime
module load openmpi/4.1.1

echo "=============================================================================="
echo "                       HPC N-BODY SIMULATION REPORT                       "
echo "=============================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Description : MPI (Distributed Memory)"
echo " Resources   : $SLURM_JOB_NUM_NODES Nodes x $SLURM_NTASKS_PER_NODE Ranks = $SLURM_NTASKS Total Ranks"
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
