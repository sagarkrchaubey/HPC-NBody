#!/bin/bash
# FILE: scripts/summary.sh

LOG_DIR="logs"

echo "========================================================================================================="
printf "%-8s | %-16s | %-20s | %-10s | %-6s | %-10s | %-10s\n" "Job ID" "Paradigm" "Config" "N (Bodies)" "Steps" "Time (s)" "GFLOPs"
echo "========================================================================================================="

# Loop through all run logs (ignoring profiling logs)
for log in ${LOG_DIR}/run_*.log; do
    if [ -f "$log" ]; then
        jobid=$(echo "$log" | grep -o '[0-9]\+')
        
        # 1. Extract and Split the Description (e.g., "MPI Ultra (48 Ranks)")
        desc_full=$(grep "Description :" "$log" | awk -F':' '{print $2}' | xargs)
        if [[ "$desc_full" == *"("*")"* ]]; then
            # Split into Paradigm Name and the Config inside the parentheses
            paradigm=$(echo "$desc_full" | awk -F' \\(' '{print $1}' | xargs)
            config=$(echo "$desc_full" | awk -F'\\(' '{print $2}' | tr -d ')' | xargs)
        else
            paradigm="$desc_full"
            config="-"
        fi
        
        # 2. Extract N and STEPS dynamically from the "Running:" echo
        run_line=$(grep "Running:" "$log" | tail -n 1)
        # Finds the executable (contains /bin/nbody_) and grabs the next two arguments
        N=$(echo "$run_line" | awk '{for(i=1;i<=NF;i++) if($i ~ /bin\/nbody_/) print $(i+1)}')
        STEPS=$(echo "$run_line" | awk '{for(i=1;i<=NF;i++) if($i ~ /bin\/nbody_/) print $(i+2)}')
        
        # 3. Extract Performance Metrics
        time=$(grep "Time:" "$log" | tail -n 1 | awk '{print $2}')
        gflops=$(grep "GFLOPs:" "$log" | tail -n 1 | awk '{print $2}')
        
        # 4. Print row if the job finished successfully
        if [ -n "$time" ] && [ -n "$gflops" ]; then
            printf "%-8s | %-16s | %-20s | %-10s | %-6s | %-10s | %-10s\n" "$jobid" "$paradigm" "$config" "$N" "$STEPS" "$time" "$gflops"
        fi
    fi
done
echo "========================================================================================================="
