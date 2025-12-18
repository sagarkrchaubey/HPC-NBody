#!/bin/bash
#SBATCH --job-name=Serial_N20k
#SBATCH --output=logs/serial_N20k_%j.log
#SBATCH --error=logs/serial_N20k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --partition=cpu

# --- SETTINGS ---
N=20000
STEPS=1000
BENCH="true"
BIN="./bin/nbody_serial"

echo "=============================================================================="
echo "                       HPC N-BODY SIMULATION REPORT                       "
echo "=============================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Description : Serial Baseline (Single Core)"
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
