#!/bin/bash

#SBATCH --exclusive
#SBATCH --job-name=r_h_u
#SBATCH --partition=cpu
#SBATCH --nodes=16
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=24
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j_run_hybrid_ultra.log
#SBATCH --error=logs/errors/%j_run_hybrid_ultra.err

N=${1:-100000}
STEPS=${2:-1000}
MODE=${3:-bench}
SAVE=${4:-1}
BIN="./bin/nbody_hybrid_ultra"

module purge
module load spack
. /home/apps/spack/share/spack/setup-env.sh
spack find gcc
spack load gcc@13.1.0%gcc@13.1.0
module spider openmpi
module load openmpi/4.1.1

export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "=============================================================================="
echo "                       HPC N-BODY SIMULATION REPORT                       "
echo "=============================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Description : Hybrid Ultra ($SLURM_NTASKS Ranks, 24 Threads/Rank)"
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
    echo "Running: mpirun -np 32 $BIN $N $STEPS $MODE $SAVE"
    time mpirun -np $SLURM_NTASKS --mca btl tcp,self,vader --map-by socket:PE=24 $BIN $N $STEPS $MODE $SAVE
else
    echo "ERROR: Binary not found at $BIN"
fi

echo "=============================================================================="
echo " End Time    : $(date)"
echo "=============================================================================="
