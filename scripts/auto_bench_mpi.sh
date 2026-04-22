#!/bin/bash
# FILE: scripts/auto_bench_pure_mpi.sh

STEPS=1000
MODE="bench"
SAVE=0

# Both Standard and Ultra Executables
SCRIPTS=("scripts/run_mpi.sh" "scripts/run_mpi_ultra.sh")

# Capped safely at the cluster's 8-node limit
NODES_LIST=(1 2 3 4 5 6 7 8)

echo "=========================================="
echo "    PURE MPI: HORIZONTAL STRONG SCALING   "
echo "=========================================="
# Fixed workload matching Hybrid script
STRONG_N=100000

for SCRIPT in "${SCRIPTS[@]}"; do
    NAME=$(basename $SCRIPT .sh)
    echo "--- Queuing $NAME ---"

    for NODES in "${NODES_LIST[@]}"; do
        TASKS=$((NODES * 48)) # 48 MPI Ranks per Node
        echo "Submitting Strong | Nodes = $NODES | N = $STRONG_N"
        sbatch --nodes=$NODES --ntasks=$TASKS --cpus-per-task=1 \
               --job-name="${NAME}_s" \
               $SCRIPT $STRONG_N $STEPS $MODE $SAVE
        sleep 0.2
    done
    echo ""
done

echo "=========================================="
echo "       PURE MPI: O(N^2) WEAK SCALING      "
echo "=========================================="
# N = 20000 * sqrt(Nodes) (Matching Hybrid exact array)
WEAK_N=(
    20000  # 1 Node
    28284  # 2 Nodes
    34641  # 3 Nodes
    40000  # 4 Nodes
    44721  # 5 Nodes
    48990  # 6 Nodes
    52915  # 7 Nodes
    56568  # 8 Nodes
)

for SCRIPT in "${SCRIPTS[@]}"; do
    NAME=$(basename $SCRIPT .sh)
    echo "--- Queuing $NAME ---"

    for i in "${!NODES_LIST[@]}"; do
        NODES=${NODES_LIST[$i]}
        N=${WEAK_N[$i]}
        TASKS=$((NODES * 48)) # 48 MPI Ranks per Node

        echo "Submitting Weak | Nodes = $NODES | N = $N"
        sbatch --nodes=$NODES --ntasks=$TASKS --cpus-per-task=1 \
               --job-name="${NAME}_w" \
               $SCRIPT $N $STEPS $MODE $SAVE
        sleep 0.2
    done
    echo ""
done

echo "=========================================="
echo "    PURE MPI: DATA SCALING (SATURATION)   "
echo "=========================================="
# Fixed Hardware: 4 Nodes (Matching Hybrid script)
FIXED_NODES=4
TASKS=$((FIXED_NODES * 48)) # 192 Total Ranks

# Exact same 35 Data points as Hybrid script
DATA_N=(
    1000 2000 4000 6000 8000 10000 15000 20000 25000 30000
    35000 40000 45000 50000 55000 60000 65000 70000 75000 80000
    85000 90000 95000 100000 110000 120000 130000 140000 150000
    160000 170000 180000 200000 225000 250000
)

for SCRIPT in "${SCRIPTS[@]}"; do
    NAME=$(basename $SCRIPT .sh)
    echo "--- Queuing $NAME ---"

    for N in "${DATA_N[@]}"; do
        echo "Submitting Data Scaling | Nodes = $FIXED_NODES | N = $N"
        sbatch --nodes=$FIXED_NODES --ntasks=$TASKS --cpus-per-task=1 \
               --job-name="${NAME}_d" \
               $SCRIPT $N $STEPS $MODE $SAVE
        sleep 0.2
    done
    echo ""
done

echo "All Pure MPI benchmarking jobs submitted! (Mirrors Hybrid parameters perfectly)"
