#!/bin/bash
#SBATCH --job-name=prof_omp_u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output=logs/prof_openmp_ultra_%j.log
#SBATCH --error=logs/errors/prof_openmp_ultra_%j.err
#SBATCH --exclusive

# --- Configuration ---
N=${1:-5000}
STEPS=1000
MODE="bench"
RESULT_DIR="vtune_reports/openmp_ultra_N${N}_ID${SLURM_JOB_ID}"

# --- Modules ---
module purge
module load gcc/8.2.0
module load oneapi/vtune/latest

# --- Execution ---
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "============================================"
echo "Profiling OpenMP Ultra"
echo "Particles: $N | Threads: $OMP_NUM_THREADS"
echo "Output: $RESULT_DIR"
echo "============================================"

vtune -collect hotspots \
      -result-dir ${RESULT_DIR} \
      -- ./bin/nbody_openmp_ultra $N $STEPS $MODE
