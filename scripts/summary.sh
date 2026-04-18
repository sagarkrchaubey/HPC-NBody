#!/bin/bash
# FILE: scripts/summary.sh

LOG_DIR="logs"

echo "======================================================================================================================="
printf "%-8s | %-15s | %-5s | %-5s | %-10s | %-10s | %-10s | %-9s | %s\n" "Job ID" "Paradigm" "Nodes" "Cores" "N (Bodies)" "Elapsed" "Internal(s)" "GFLOPs" "NodeList"
echo "======================================================================================================================="

for log in ${LOG_DIR}/run_*.log; do
    if [ -f "$log" ]; then
        jobid=$(echo "$log" | grep -o '[0-9]\+')
        
        # Query SLURM for Allocation details AND Elapsed Time (-X ensures we only get the master record, ignoring .batch/.0)
        slurm_data=$(sacct -j $jobid --format=AllocNodes,AllocCPUs,Elapsed,NodeList -X -n -P 2>/dev/null | head -n 1)
        nodes=$(echo "$slurm_data" | awk -F'|' '{print $1}')
        cores=$(echo "$slurm_data" | awk -F'|' '{print $2}')
        elapsed=$(echo "$slurm_data" | awk -F'|' '{print $3}')
        nodelist=$(echo "$slurm_data" | awk -F'|' '{print $4}')

        if [ -z "$nodes" ]; then nodes="1"; fi
        if [ -z "$cores" ]; then cores="-"; fi
        if [ -z "$elapsed" ]; then elapsed="-"; fi
        if [ -z "$nodelist" ]; then nodelist="-"; fi

        # Extract Paradigm Description
        desc_full=$(grep "Description :" "$log" | awk -F':' '{print $2}' | xargs)
        if [[ "$desc_full" == *"("*")"* ]]; then
            paradigm=$(echo "$desc_full" | awk -F' \\(' '{print $1}' | xargs)
        else
            paradigm="$desc_full"
        fi
        
        # Extract Parameters and Metrics
        run_line=$(grep "Running:" "$log" | tail -n 1)
        N=$(echo "$run_line" | awk '{for(i=1;i<=NF;i++) if($i ~ /bin\/nbody_/) print $(i+1)}')
        
        time=$(grep "Time:" "$log" | tail -n 1 | awk '{print $2}')
        gflops=$(grep "GFLOPs:" "$log" | tail -n 1 | awk '{print $2}')
        
        # Print the formatted row
        if [ -n "$time" ] && [ -n "$gflops" ]; then
            printf "%-8s | %-15s | %-5s | %-5s | %-10s | %-10s | %-10s | %-9s | %s\n" "$jobid" "$paradigm" "$nodes" "$cores" "$N" "$elapsed" "$time" "$gflops" "$nodelist"
        fi
    fi
done
echo "======================================================================================================================="
