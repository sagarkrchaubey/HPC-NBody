#!/bin/bash
#SBATCH --job-name=prof_mpi_u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/prof_mpi_ultra_%j.log
#SBATCH --error=logs/errors/prof_mpi_ultra_%j.err
#SBATCH --exclusive

# --- Configuration ---
N=${1:-5000}
STEPS=1000
MODE="bench"
RESULT_DIR="vtune_reports/mpi_ultra_N${N}_ID${SLURM_JOB_ID}"

# --- Modules ---
module purge
module load gcc/8.2.0
module load openmpi/4.1.1
module load oneapi/vtune/latest

echo "============================================"
echo "Profiling MPI Ultra"
echo "Particles: $N | Ranks: 48"
echo "Output: $RESULT_DIR"
echo "============================================"

mpirun -np 48 \
	--mca btl tcp,self,vader \
	 vtune -collect hotspots \
	-result-dir ${RESULT_DIR} \
	-- ./bin/nbody_mpi_ultra $N $STEPS $MODE
