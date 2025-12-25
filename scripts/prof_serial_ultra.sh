#!/bin/bash
#SBATCH --job-name=prof_ultra
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/prof_serial_ultra_%j.log
#SBATCH --error=logs/errors/prof_serial_ultra_%j.err
#SBATCH --exclusive

# --- Configuration ---
# 1. Get N from command line arg, default to 5000 if not provided
N=${1:-5000}
STEPS=1000
MODE="bench"

# 2. Define Output Directory with N and Job ID to prevent collisions
RESULT_DIR="vtune_reports/serial_ultra_N${N}_ID${SLURM_JOB_ID}"

# 3. Load Modules (Match what you used for compiling if needed)
# If you used a specific gcc for compiling, load it here just in case libraries are needed
# module load gcc/8.2.0 
module load oneapi/vtune/latest

echo "============================================"
echo "Profiling Serial Ultra Implementation"
echo "Particles (N) : $N"
echo "Steps         : $STEPS"
echo "Output Dir    : $RESULT_DIR"
echo "Date          : $(date)"
echo "============================================"

# 4. Run VTune
# -collect hotspots : Good for finding CPU bottlenecks
# -knob sampling-mode=hw : Uses hardware counters (low overhead)
vtune -collect hotspots \
      -result-dir ${RESULT_DIR} \
      -- ./bin/nbody_serial_ultra $N $STEPS $MODE

echo "Profiling finished. Result saved to: $RESULT_DIR"
