#!/bin/bash
# FILE: scripts/summary.sh

LOG_DIR="logs"

echo "============================================================================================================="
printf "%-8s | %-16s | %-5s | %-15s | %-10s | %-10s | %-10s | %s\n" "Job ID" "Paradigm" "Nodes" "Configuration" "N (Bodies)" "Time (s)" "GFLOPs" "NodeList"
echo "============================================================================================================="

for log in ${LOG_DIR}/run_*.log; do
    if [ -f "$log" ]; then
        jobid=$(echo "$log" | grep -o '[0-9]\+')

        # Query SLURM for Allocation details (-X ensures we only get the master record)
        slurm_data=$(sacct -j $jobid --format=AllocNodes,NodeList -X -n -P 2>/dev/null | head -n 1)
        nodes=$(echo "$slurm_data" | awk -F'|' '{print $1}')
        nodelist=$(echo "$slurm_data" | awk -F'|' '{print $2}')

        if [ -z "$nodes" ]; then nodes="1"; fi
        if [ -z "$nodelist" ]; then nodelist="-"; fi

        # Extract Paradigm and exact Configuration (Threads/Ranks/GPU) from the log
        desc_full=$(grep "Description :" "$log" | awk -F':' '{print $2}' | xargs)
        
        if [[ "$desc_full" == *"("*")"* ]]; then
            # Splits "OpenMP Ultra (16 Threads)" into Paradigm="OpenMP Ultra" and Config="16 Threads"
            paradigm=$(echo "$desc_full" | awk -F' \\(' '{print $1}' | xargs)
            config=$(echo "$desc_full" | awk -F'\\(' '{print $2}' | tr -d ')' | xargs)
        else
            paradigm="$desc_full"
            config="-"
        fi

        # Extract Parameters (N) and Metrics (Time, GFLOPs)
        run_line=$(grep "Running:" "$log" | tail -n 1)
        N=$(echo "$run_line" | awk '{for(i=1;i<=NF;i++) if($i ~ /bin\/nbody_/) print $(i+1)}')

        time=$(grep "Time:" "$log" | tail -n 1 | awk '{print $2}')
        gflops=$(grep "GFLOPs:" "$log" | tail -n 1 | awk '{print $2}')

        # Print the formatted row only if the run successfully outputted metrics
        if [ -n "$time" ] && [ -n "$gflops" ]; then
            printf "%-8s | %-16s | %-5s | %-15s | %-10s | %-10s | %-10s | %s\n" "$jobid" "$paradigm" "$nodes" "$config" "$N" "$time" "$gflops" "$nodelist"
        fi
    fi
# Sort numerically by Job ID to ensure perfect chronological order
done | sort -k1,1n
echo "============================================================================================================="
