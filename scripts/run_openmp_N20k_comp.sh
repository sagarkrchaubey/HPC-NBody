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

# Paths
SRC_FILE="src/openmp/nbody_openmp.cpp"
BIN_FILE="/tmp/nbody_openmp_$SLURM_JOB_ID"  # Compile to a temporary fast location

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=============================================================================="
echo "                       HPC N-BODY SIMULATION REPORT                       "
echo "=============================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Node        : $(hostname)"
echo " CPU Model   : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
echo "=============================================================================="
echo "                           COMPILING ON NODE                                  "
echo "=============================================================================="

# Compile with strict architecture optimization for THIS node
g++ -g -fopenmp -march=native -ffast-math -std=c++11 \
    -fno-omit-frame-pointer \
    $SRC_FILE -o $BIN_FILE

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "Success: Compiled optimized binary at $BIN_FILE"

echo "=============================================================================="
echo "                           SIMULATION EXECUTION                               "
echo "=============================================================================="

# Run using the fresh binary
numactl --interleave=all time $BIN_FILE $N $STEPS $BENCH

# Cleanup
#rm -f $BIN_FILE

echo "=============================================================================="
echo " End Time    : $(date)"
echo "=============================================================================="
