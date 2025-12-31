#!/bin/bash
#SBATCH --job-name=prof_cuda
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/prof_cuda_ulra_%j.log
#SBATCH --error=logs/errors/prof_cuda_ultra_%j.err

# --- Configuration ---
# Default to 20k particles for a good GPU workload
N=${1:-20000}
STEPS=1000
MODE="bench"

# Report output file (nsys-rep)
REPORT_NAME="cuda_ultra_profile_N${N}_ID${SLURM_JOB_ID}"
REPORT_DIR="cuda_reports"

# --- Modules ---
module purge
module load spack
. /home/apps/spack/share/spack/setup-env.sh
spack find gcc
spack load gcc@11.2.0

module spider cuda
module load cuda/12.0

echo "Modules used:"
gcc --version
nvcc --version
nsys --version


echo "============================================"
echo "Profiling CUDA Implementation"
echo "Particles: $N"
echo "Profiler:  NVIDIA Nsight Systems (nsys)"
echo "Output:    $REPORT_DIR/$REPORT_NAME.nsys-rep"
echo "============================================"

# Ensure output directory exists
mkdir -p $REPORT_DIR

# --- Run Profiler ---
# --stats=true      : Prints a summary table to the log file (Text output)
# -t cuda,osrt,nvtx : Traces CUDA calls, OS runtime, and NVTX markers
# -o                : Output filename for the visual report
# --force-overwrite : Overwrite if file exists

nsys profile \
    --stats=true \
    --trace=cuda,osrt,nvtx \
    --force-overwrite=true \
    --output=${REPORT_DIR}/${REPORT_NAME} \
    ./bin/nbody_cuda_ultra $N $STEPS $MODE
