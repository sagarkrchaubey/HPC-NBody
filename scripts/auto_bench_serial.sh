#!/bin/bash
# FILE: scripts/auto_bench_serial.sh

# ===========================
# CONTROL SWITCHES
# ===========================
RUN_STANDARD=1   # Set to 1 to run Serial Standard
RUN_ULTRA=1      # Set to 1 to run Serial Ultra

STEPS=1000
MODE="bench"
SAVE=1

# ===========================
# SCALING PARAMETERS
# ===========================
# Increased data points for a smooth curve up to 100,000
DATA_N=(1000 2000 4000 6000 8000 10000 15000 20000 25000 30000 35000 40000 50000 60000 70000 80000 90000 100000 120000 140000 )

MAX_DATA_INDEX=$((${#DATA_N[@]} - 1))
DATA_STR="${DATA_N[*]}"

# Determine which codes to run
TARGETS=()
if [ "$RUN_STANDARD" -eq 1 ]; then TARGETS+=("std"); fi
if [ "$RUN_ULTRA" -eq 1 ]; then TARGETS+=("ultra"); fi

echo "========================================================="
echo " Launching Master Serial Benchmarking Matrix (Array Mode)"
echo "========================================================="

for TARGET in "${TARGETS[@]}"; do

    if [ "$TARGET" == "std" ]; then
        SCRIPT="scripts/run_serial.sh"
        PREFIX="std"
        echo -e "\n---> Processing Serial STANDARD..."
    else
        SCRIPT="scripts/run_serial_ultra.sh"
        PREFIX="ultra"
        echo -e "\n---> Processing Serial ULTRA..."
    fi

    echo "  Submitting Data Scaling Array (Tasks: 0 to $MAX_DATA_INDEX)..."
    
    # Notice the --exclusive and --time=24:00:00 to protect the benchmarks
    sbatch <<EOF >/dev/null
#!/bin/bash
#SBATCH --job-name=ser_${PREFIX}_dat
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --exclusive
#SBATCH --array=0-${MAX_DATA_INDEX}
#SBATCH --output=logs/%A_%a_run_serial_${PREFIX}_data.log
#SBATCH --error=logs/errors/%A_%a_run_serial_${PREFIX}_data.err

ARR=($DATA_STR)
N=\${ARR[\$SLURM_ARRAY_TASK_ID]}

echo "Running Serial Task \$SLURM_ARRAY_TASK_ID | N=\$N"

bash $SCRIPT \$N $STEPS $MODE $SAVE
EOF
    sleep 0.5

done

echo ""
echo "========================================================="
echo " All Serial Array Jobs successfully injected into the queue!"
echo " Use 'squeue --me' to monitor."
echo "========================================================="
