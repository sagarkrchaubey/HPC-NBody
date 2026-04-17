#!/bin/bash
# FILE: scripts/auto_bench_gpu.sh

# The exact data scaling matrix we discussed
PARTICLES=(1000 2000 4000 6000 8000 10000 15000 20000 25000 30000 40000 50000 60000 75000 100000 125000 150000 175000 200000 250000)
STEPS=1000
MODE="bench"
SAVE=1

echo "=========================================="
echo " Launching CUDA Standard Benchmark Matrix"
echo "=========================================="
for N in "${PARTICLES[@]}"; do
    echo "Submitting CUDA Standard | N = $N"
    sbatch scripts/run_cuda.sh $N $STEPS $MODE $SAVE
    sleep 0.5 # A tiny pause so we don't spam the SLURM controller all at once
done

echo ""
echo "=========================================="
echo " Launching CUDA Ultra Benchmark Matrix"
echo "=========================================="
for N in "${PARTICLES[@]}"; do
    echo "Submitting CUDA Ultra | N = $N"
    sbatch scripts/run_cuda_ultra.sh $N $STEPS $MODE $SAVE
    sleep 0.5
done

echo ""
echo "All GPU benchmarking jobs submitted!"
echo "Use 'squeue -u \$USER' to monitor progress."
echo "Once complete, run 'make summary' to view the results."
