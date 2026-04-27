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

echo "=========================================="
echo " GPU Benchmark Matrix Launcher"
echo "=========================================="

# ===========================
# CUDA STANDARD
# ===========================
if [ "$RUN_STANDARD" -eq 1 ]; then
    echo "Launching CUDA Standard Benchmark Matrix"
    echo "------------------------------------------"
    for N in "${PARTICLES[@]}"; do
        echo "Submitting CUDA Standard | N = $N"
        sbatch scripts/run_cuda.sh $N $STEPS $MODE $SAVE
        sleep 0.5
    done
fi

# ===========================
# CUDA ULTRA
# ===========================
if [ "$RUN_ULTRA" -eq 1 ]; then
    echo ""
    echo "Launching CUDA Ultra Benchmark Matrix"
    echo "------------------------------------------"
    for N in "${PARTICLES[@]}"; do
        echo "Submitting CUDA Ultra | N = $N"
        sbatch scripts/run_cuda_ultra.sh $N $STEPS $MODE $SAVE
        sleep 0.5
    done
fi

echo ""
echo "All selected GPU benchmarking jobs submitted!"
echo "Use 'squeue -u \$USER' to monitor progress."
echo "Once complete, run 'make summary' to view the results."
