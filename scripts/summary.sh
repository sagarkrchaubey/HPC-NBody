#!/bin/bash
# FILE: scripts/summary.sh

LOG_DIR="logs"

echo "======================================================================================================="
printf "%-8s | %-16s | %-26s | %-10s | %-10s | %-10s | %s\n" "Job ID" "Paradigm" "Hardware" "N (Bodies)" "Time (s)" "GFLOPs" "NodeList"
echo "======================================================================================================="

# Extract Job IDs for batch SLURM querying (Fixed glob pattern and safer extraction)
JOB_IDS=$(ls ${LOG_DIR}/*run_*.log 2>/dev/null | awk -F'/' '{print $NF}' | grep -o '^[0-9]\+' | paste -sd "," -)

if [ -n "$JOB_IDS" ]; then
    sacct -j "$JOB_IDS" --format=JobID,AllocNodes,NodeList -X -n -P > /tmp/slurm_cache.txt 2>/dev/null
else
    touch /tmp/slurm_cache.txt
fi

# Fixed glob pattern to match the new file names
for log in ${LOG_DIR}/*run_*.log; do
    if [ -f "$log" ]; then
        # Safely extract job ID from filename (e.g., 60010_run_mpi.log -> 60010)
        jobid=$(basename "$log" | awk -F'_' '{print $1}')

        # Read hardware layout from cache
        slurm_data=$(grep "^${jobid}|" /tmp/slurm_cache.txt | head -n 1)
        nodes=$(echo "$slurm_data" | awk -F'|' '{print $2}')
        nodelist=$(echo "$slurm_data" | awk -F'|' '{print $3}')

        if [ -z "$nodes" ]; then nodes="1"; fi
        if [ -z "$nodelist" ]; then nodelist="-"; fi

        # Extract Paradigm string (Safe parsing fix)
        desc_full=$(grep "Description :" "$log" | awk -F':' '{print $2}' | xargs)

        if [[ "$desc_full" == *"("*")"* ]]; then
            paradigm=$(echo "$desc_full" | awk -F'[(]' '{print $1}' | xargs)
            config=$(echo "$desc_full" | awk -F'[(]' '{print $2}' | awk -F'[)]' '{print $1}' | xargs)
        else
            paradigm="$desc_full"
            config="-"
        fi

        # --- COMPACT HARDWARE COLUMN ---
        # Combines everything into one string like "8 Nodes, 8 Ranks, 24 Threads"
        hardware="${nodes} Nodes, ${config}"

        # Extract Workload and Execution metrics
        run_line=$(grep "Running:" "$log" | tail -n 1)
        N=$(echo "$run_line" | awk '{for(i=1;i<=NF;i++) if($i ~ /bin\/nbody_/) print $(i+1)}')

        time=$(grep "Time:" "$log" | tail -n 1 | awk '{print $2}')
        gflops=$(grep "GFLOPs:" "$log" | tail -n 1 | awk '{print $2}')

        # Print only if metrics exist
        if [ -n "$time" ] && [ -n "$gflops" ]; then
            printf "%-8s | %-16s | %-26s | %-10s | %-10s | %-10s | %s\n" "$jobid" "$paradigm" "$hardware" "$N" "$time" "$gflops" "$nodelist"
        fi
    fi
done | sort -k1,1n

echo "======================================================================================================="
rm -f /tmp/slurm_cache.txt
