#!/bin/bash
# FILE: scripts/auto_bench_openmp.sh

# ===========================
# CONTROL SWITCHES
# ===========================
RUN_STANDARD=1   # Set to 1 to run OpenMP Standard
RUN_ULTRA=1      # Set to 1 to run OpenMP Ultra

STEPS=1000
MODE="bench"
SAVE=1

# ===========================
# SCALING PARAMETERS
# ===========================
# 1. Data Scaling (Fixed at 48 Threads)
DATA_N=(1000 2000 4000 6000 8000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000 110000 120000 130000 140000 150000 160000 170000 180000 200000 250000 300000 400000 500000)

# 2. Strong Scaling (Fixed N)
STRONG_THREADS=(1 2 4 6 8 12 16 20 24 25 28 32 36 40 44 48)
STRONG_N=50000

# 3. Weak Scaling Pairs: (Threads N_calculated)
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

# ===========================
# ARRAY LOGIC PREPARATION
# ===========================
MAX_DATA_INDEX=$((${#DATA_N[@]} - 1))
DATA_STR="${DATA_N[*]}"

MAX_STRONG_INDEX=$((${#STRONG_THREADS[@]} - 1))
STRONG_STR="${STRONG_THREADS[*]}"

# Deconstruct Weak Scaling Pairs into separate parallel arrays for easier SLURM handling
WEAK_T=()
WEAK_N=()
for PAIR in "${WEAK_PAIRS[@]}"; do
    WEAK_T+=($(echo $PAIR | awk '{print $1}'))
    WEAK_N+=($(echo $PAIR | awk '{print $2}'))
done
MAX_WEAK_INDEX=$((${#WEAK_T[@]} - 1))
WEAK_T_STR="${WEAK_T[*]}"
WEAK_N_STR="${WEAK_N[*]}"

# Determine which codes to run
TARGETS=()
if [ "$RUN_STANDARD" -eq 1 ]; then TARGETS+=("std"); fi
if [ "$RUN_ULTRA" -eq 1 ]; then TARGETS+=("ultra"); fi

echo "========================================================="
echo " Launching Master OpenMP Benchmarking Matrix (Array Mode)"
echo "========================================================="

for TARGET in "${TARGETS[@]}"; do

    if [ "$TARGET" == "std" ]; then
        SCRIPT="scripts/run_openmp.sh"
        PREFIX="std"
        echo -e "\n---> Processing OpenMP STANDARD..."
    else
        SCRIPT="scripts/run_openmp_ultra.sh"
        PREFIX="ultra"
        echo -e "\n---> Processing OpenMP ULTRA..."
    fi

    # ---------------------------------------------------------
    # 1. DATA SCALING (Array Mode)
    # ---------------------------------------------------------
    echo "  [1/3] Submitting Data Scaling Array (Tasks: 0 to $MAX_DATA_INDEX)..."
    sbatch <<EOF >/dev/null
#!/bin/bash
#SBATCH --job-name=omp_${PREFIX}_dat
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:30:00
#SBATCH --exclusive
#SBATCH --array=0-${MAX_DATA_INDEX}
#SBATCH --output=logs/%A_%a_run_openmp_${PREFIX}_data.log
#SBATCH --error=logs/errors/%A_%a_run_openmp_${PREFIX}_data.err

ARR=($DATA_STR)
N=\${ARR[\$SLURM_ARRAY_TASK_ID]}

echo "Running Data Scaling Task \$SLURM_ARRAY_TASK_ID | N=\$N"
bash $SCRIPT \$N $STEPS $MODE $SAVE
EOF
    sleep 0.5

    # ---------------------------------------------------------
    # 2. STRONG SCALING (Array Mode)
    # ---------------------------------------------------------
    echo "  [2/3] Submitting Strong Scaling Array (Tasks: 0 to $MAX_STRONG_INDEX)..."
    sbatch <<EOF >/dev/null
#!/bin/bash
#SBATCH --job-name=omp_${PREFIX}_str
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:30:00
#SBATCH --exclusive
#SBATCH --array=0-${MAX_STRONG_INDEX}
#SBATCH --output=logs/%A_%a_run_openmp_${PREFIX}_strong.log
#SBATCH --error=logs/errors/%A_%a_run_openmp_${PREFIX}_strong.err

ARR=($STRONG_STR)
T=\${ARR[\$SLURM_ARRAY_TASK_ID]}

echo "Running Strong Scaling Task \$SLURM_ARRAY_TASK_ID: \$T Threads | N=$STRONG_N"

# Override thread count for the specific array task securely 
export OMP_NUM_THREADS=\$T
export SLURM_CPUS_PER_TASK=\$T

bash $SCRIPT $STRONG_N $STEPS $MODE $SAVE
EOF
    sleep 0.5

    # ---------------------------------------------------------
    # 3. WEAK SCALING (Array Mode)
    # ---------------------------------------------------------
    echo "  [3/3] Submitting Weak Scaling Array (Tasks: 0 to $MAX_WEAK_INDEX)..."
    sbatch <<EOF >/dev/null
#!/bin/bash
#SBATCH --job-name=omp_${PREFIX}_wk
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:30:00
#SBATCH --exclusive
#SBATCH --array=0-${MAX_WEAK_INDEX}
#SBATCH --output=logs/%A_%a_run_openmp_${PREFIX}_weak.log
#SBATCH --error=logs/errors/%A_%a_run_openmp_${PREFIX}_weak.err

ARR_T=($WEAK_T_STR)
ARR_N=($WEAK_N_STR)

T=\${ARR_T[\$SLURM_ARRAY_TASK_ID]}
N=\${ARR_N[\$SLURM_ARRAY_TASK_ID]}

echo "Running Weak Scaling Task \$SLURM_ARRAY_TASK_ID: \$T Threads | N=\$N"

# Override thread count for the specific array task securely 
export OMP_NUM_THREADS=\$T
export SLURM_CPUS_PER_TASK=\$T

bash $SCRIPT \$N $STEPS $MODE $SAVE
EOF
    sleep 0.5

done

echo ""
echo "========================================================="
echo " All OpenMP Array Jobs successfully injected into the queue!"
echo " Use 'squeue --me' to monitor."
echo "========================================================="
