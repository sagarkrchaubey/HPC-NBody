#!/bin/bash
# FILE: scripts/auto_bench_openmp.sh

STEPS=1000
PARTITION="cpu"

# High-Resolution Data Scaling N Values (37 Data Points)
DATA_N=(1000 2000 4000 6000 8000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000 110000 120000 130000 140000 150000 160000 170000 180000 200000 250000 300000 400000 500000)

# Strong Scaling Thread Counts (Notice the 24 to 25 jump for NUMA testing)
STRONG_THREADS=(1 2 4 6 8 12 16 20 24 25 28 32 36 40 44 48)
STRONG_N=50000

# Weak Scaling Pairs: (Threads N_calculated) based on N=5000 at 1 Thread
# N_new = 5000 * sqrt(Threads)
WEAK_PAIRS=(
    "1 5000"
    "2 7071"
    "4 10000"
    "8 14142"
    "12 17320"
    "16 20000"
    "24 24494"
    "32 28284"
    "48 34641"
)

SCRIPTS=("scripts/run_openmp.sh" "scripts/run_openmp_ultra.sh")

echo "========================================================="
echo " Submitting Massive OpenMP Benchmarking Matrix"
echo "========================================================="

for SCRIPT in "${SCRIPTS[@]}"; do
    
    # 1. DATA SCALING (Fixed at 48 Threads)
    for N in "${DATA_N[@]}"; do
        sbatch --nodes=1 --cpus-per-task=48 \
               --job-name="omp_data" \
               $SCRIPT $N $STEPS
    done

    # 2. STRONG SCALING (Fixed N)
    for THREADS in "${STRONG_THREADS[@]}"; do
        sbatch --nodes=1 --cpus-per-task=$THREADS \
               --job-name="omp_strong" \
               $SCRIPT $STRONG_N $STEPS
    done

    # 3. WEAK SCALING (Scaled N and Threads)
    for PAIR in "${WEAK_PAIRS[@]}"; do
        THREADS=$(echo $PAIR | awk '{print $1}')
        N=$(echo $PAIR | awk '{print $2}')
        
        sbatch --nodes=1 --cpus-per-task=$THREADS \
               --job-name="omp_weak" \
               $SCRIPT $N $STEPS
    done

done

echo "========================================================="
echo " All OpenMP jobs successfully injected into the queue!"
echo " Monitor with: squeue --me"
echo "========================================================="
