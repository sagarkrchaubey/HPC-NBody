#!/bin/bash

echo "========================================================================="
echo "        AUTO BENCHMARK: SERIAL STANDARD vs SERIAL ULTRA                  "
echo "========================================================================="

# Fixed parameters
STEPS=1000
MODE="bench"
SAVE=0  # Turn off file saving for pure performance benchmarks

# WARNING: O(N^2) complexity on a single core! 
# Do NOT increase N beyond 100,000 unless you drastically reduce STEPS.
DATA_N=(1000 5000 10000 25000 50000 75000 100000)

# ==========================================
# 1. Submit Serial Standard
# ==========================================
echo "[1/2] Submitting Serial Standard jobs..."
for n in "${DATA_N[@]}"; do
    echo "  -> Submitting N=$n"
    sbatch scripts/run_serial.sh $n $STEPS $MODE $SAVE
    sleep 0.5 # Brief pause to be kind to the SLURM scheduler
done

echo ""

# ==========================================
# 2. Submit Serial Ultra
# ==========================================
echo "[2/2] Submitting Serial Ultra jobs..."
for n in "${DATA_N[@]}"; do
    echo "  -> Submitting N=$n"
    sbatch scripts/run_serial_ultra.sh $n $STEPS $MODE $SAVE
    sleep 0.5
done

echo "========================================================================="
echo " All Serial benchmark jobs submitted!                                    "
echo " Monitor your queue with: squeue -u \$USER                                "
echo "========================================================================="
