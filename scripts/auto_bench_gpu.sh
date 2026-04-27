#!/bin/bash
# FILE: scripts/auto_bench_gpu.sh

# ===========================
# CONTROL SWITCH (EDIT THIS)
# ===========================
RUN_STANDARD=0   # set to 1 to enable CUDA Standard
RUN_ULTRA=1      # set to 1 to enable CUDA Ultra

# Data scaling matrix
PARTICLES=(1000 2000 4000 6000 8000 10000 15000 20000 25000 30000 40000 50000 60000 75000 100000 125000 150000 175000 200000 250000)
STEPS=1000
MODE="bench"
SAVE=1

# Calculate limits for the job array
MAX_INDEX=$((${#PARTICLES[@]} - 1))
PARTICLES_STR="${PARTICLES[*]}" # Convert to space-separated string for SLURM

echo "=========================================="
echo " GPU Benchmark Matrix Launcher (ARRAY MODE)"
echo "=========================================="

# ===========================
# CUDA STANDARD
# ===========================
if [ "$RUN_STANDARD" -eq 1 ]; then
    echo "Submitting CUDA Standard Job Array (Tasks: 0 to $MAX_INDEX)..."
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gpu_std_arr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-${MAX_INDEX}
#SBATCH --output=logs/%A_%a_run_cuda_std.log
#SBATCH --error=logs/errors/%A_%a_run_cuda_std.err

# Rebuild array on compute node and extract N based on task ID
ARR=($PARTICLES_STR)
N=\${ARR[\$SLURM_ARRAY_TASK_ID]}

echo "Running Array Task \$SLURM_ARRAY_TASK_ID | N=\$N"
bash scripts/run_cuda.sh \$N $STEPS $MODE $SAVE
EOF
fi

# ===========================
# CUDA ULTRA
# ===========================
if [ "$RUN_ULTRA" -eq 1 ]; then
    echo "Submitting CUDA Ultra Job Array (Tasks: 0 to $MAX_INDEX)..."
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gpu_ult_arr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-${MAX_INDEX}
#SBATCH --output=logs/%A_%a_run_cuda_ultra.log
#SBATCH --error=logs/errors/%A_%a_run_cuda_ultra.err

# Rebuild array on compute node and extract N based on task ID
ARR=($PARTICLES_STR)
N=\${ARR[\$SLURM_ARRAY_TASK_ID]}

echo "Running Array Task \$SLURM_ARRAY_TASK_ID | N=\$N"
bash scripts/run_cuda_ultra.sh \$N $STEPS $MODE $SAVE
EOF
fi

echo ""
echo "Array jobs submitted successfully!"
echo "Use 'squeue -u \$USER' to monitor."
