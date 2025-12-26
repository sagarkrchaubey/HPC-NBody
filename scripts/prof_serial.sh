#!/bin/bash
#SBATCH --job-name=prof_serial
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/prof_serial_%j.log
#SBATCH --error=logs/errors/prof_serial_%j.err
#SBATCH --exclusive

# --- Configuration ---
# 1. Get N from command line arg, default to 5000 if not provided
N=${1:-5000}
STEPS=1000
MODE="bench"

# 2. Define Output Directory with N and Job ID
RESULT_DIR="vtune_reports/serial_N${N}_ID${SLURM_JOB_ID}"

# 3. Load Modules
module load oneapi/vtune/latest
module load gcc/8.2.0
echo "============================================"
echo "Profiling Standard Serial Implementation"
echo "Particles (N) : $N"
echo "Steps         : $STEPS"
echo "Output Dir    : $RESULT_DIR"
echo "Date          : $(date)"
echo "============================================"

# 4. Run VTune (Hotspots Analysis)
# Removed "-knob sampling-mode=hw" to avoid permission errors
vtune -collect hotspots \
      -result-dir ${RESULT_DIR} \
      -- ./bin/nbody_serial $N $STEPS $MODE

echo "Profiling finished. Result saved to: $RESULT_DIR"
