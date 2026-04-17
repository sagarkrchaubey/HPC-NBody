#!/bin/bash
# FILE: scripts/auto_bench_hybrid.sh

STEPS=1000
MODE="bench"
SAVE=1

# 20 Data points scaling from 1 to 40 nodes
NODES_LIST=(1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 24 28 32 36 40)

echo "=========================================="
echo "  HYBRID ULTRA: HORIZONTAL STRONG SCALING "
echo "=========================================="
# Fixed workload (massive enough to span 40 nodes), increasing resources
STRONG_N=100000

for NODES in "${NODES_LIST[@]}"; do
    TASKS=$((NODES * 2)) # 2 MPI Ranks per Node (1 per Socket)
    echo "Submitting Strong Scaling | Nodes = $NODES | Ranks = $TASKS | N = $STRONG_N"
    
    sbatch --nodes=$NODES --ntasks=$TASKS scripts/run_hybrid_ultra.sh $STRONG_N $STEPS $MODE $SAVE
    sleep 0.5
done

echo ""
echo "=========================================="
echo "   HYBRID ULTRA: O(N^2) WEAK SCALING      "
echo "=========================================="
# Workload scaled quadratically: N = 20000 * sqrt(Nodes)
WEAK_N=(
    20000  # 1 Node
    28284  # 2 Nodes
    34641  # 3 Nodes
    40000  # 4 Nodes
    44721  # 5 Nodes
    48990  # 6 Nodes
    52915  # 7 Nodes
    56568  # 8 Nodes
    60000  # 9 Nodes
    63245  # 10 Nodes
    69282  # 12 Nodes
    74833  # 14 Nodes
    80000  # 16 Nodes
    84852  # 18 Nodes
    89442  # 20 Nodes
    97979  # 24 Nodes
    105830 # 28 Nodes
    113137 # 32 Nodes
    120000 # 36 Nodes
    126491 # 40 Nodes
)

for i in "${!NODES_LIST[@]}"; do
    NODES=${NODES_LIST[$i]}
    N=${WEAK_N[$i]}
    TASKS=$((NODES * 2))
    
    echo "Submitting Weak Scaling | Nodes = $NODES | Ranks = $TASKS | N = $N"
    sbatch --nodes=$NODES --ntasks=$TASKS scripts/run_hybrid_ultra.sh $N $STEPS $MODE $SAVE
    sleep 0.5
done

echo ""
echo "All 40 Hybrid benchmarking jobs submitted!"
echo "Use 'squeue -u \$USER' to monitor progress."
