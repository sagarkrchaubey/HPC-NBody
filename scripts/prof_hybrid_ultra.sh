#!/bin/bash
#SBATCH --job-name=prof_hyb_u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=logs/prof_hybrid_ultra_%j.log
#SBATCH --error=logs/errors/prof_hybrid_ultra_%j.err
#SBATCH --exclusive

# --- Configuration ---
N=${1:-5000}
STEPS=1000
MODE="bench"
RESULT_DIR="vtune_reports/hybrid_ultra_N${N}_ID${SLURM_JOB_ID}"

# --- Modules ---
module purge
module load gcc/8.2.0
module load openmpi/4.1.1
module load oneapi/vtune/latest

# --- Hybrid Environment ---
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "============================================"
echo "Profiling Hybrid Ultra"
echo "Particles: $N"
echo "Ranks: 2 | Threads/Rank: 24"
echo "Output: $RESULT_DIR"
echo "============================================"

# Using map-by socket to ensure the Ultra NUMA optimizations work during profiling
mpirun -np 2 \
    --map-by socket:PE=24 \
    --mca btl tcp,self,vader \
    vtune -collect hotspots \
    -result-dir ${RESULT_DIR} \
    -- ./bin/nbody_hybrid_ultra $N $STEPS $MODE
