#!/bin/bash
#SBATCH --job-name=run_mpi
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_mpi_%j.log
#SBATCH --error=logs/errors/run_mpi_%j.err
#SBATCH --exclusive

N=${1:-5000}
STEPS=${2:-1000}
MODE=${3:-bench}
SAVE=${4:-1}
BIN="./bin/nbody_mpi"

module purge
module load spack
. /home/apps/spack/share/spack/setup-env.sh
spack find gcc
spack load gcc@13.1.0%gcc@13.1.0
module spider openmpi
module load openmpi/4.1.1

echo "=============================================================================="
echo "                       HPC N-BODY SIMULATION REPORT                       "
echo "=============================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Description : MPI Standard (48 Ranks)"
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
    echo "Running: mpirun -np 48 $BIN $N $STEPS $MODE $SAVE"
    time mpirun -np 48 --mca btl tcp,self,vader $BIN $N $STEPS $MODE $SAVE
else
    echo "ERROR: Binary not found at $BIN"
fi

echo "=============================================================================="
echo " End Time    : $(date)"
echo "=============================================================================="
