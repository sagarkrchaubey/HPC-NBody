#!/bin/bash
# FILE: scripts/summary.sh

LOG_DIR="logs"

echo "======================================================================================================================================="
printf "%-8s | %-15s | %-5s | %-6s | %-18s | %-8s | %-8s | %-10s | %-7s | %-6s\n" "Job ID" "Paradigm" "Nodes" "Cores" "NodeList" "N" "Time (s)" "Actual" "Peak" "Eff(%)"
printf "%-8s | %-15s | %-5s | %-6s | %-18s | %-8s | %-8s | %-10s | %-7s | %-6s\n" "" "" "" "" "" "(Bodies)" "" "(GFLOPs)" "(GFLOPs)" ""
echo "======================================================================================================================================="

for log in ${LOG_DIR}/run_*.log; do
    if [ -f "$log" ]; then
        jobid=$(echo "$log" | grep -o '[0-9]\+')
        
        # 1. Query SLURM Database for exact hardware footprint using sacct
        # Format: AllocNodes|AllocCPUs|NodeList
        slurm_data=$(sacct -j $jobid --format=AllocNodes,AllocCPUs,NodeList -X -n -P 2>/dev/null | head -n 1)
        nodes=$(echo "$slurm_data" | awk -F'|' '{print $1}')
        cores=$(echo "$slurm_data" | awk -F'|' '{print $2}')
        nodelist=$(echo "$slurm_data" | awk -F'|' '{print $3}')

        # Fallbacks just in case the job is too old or sacct is slow
        if [ -z "$nodes" ]; then nodes="1"; fi
        if [ -z "$cores" ]; then cores="-"; fi
        if [ -z "$nodelist" ]; then nodelist="-"; fi

        # 2. Extract Paradigm Description
        desc_full=$(grep "Description :" "$log" | awk -F':' '{print $2}' | xargs)
        if [[ "$desc_full" == *"("*")"* ]]; then
            paradigm=$(echo "$desc_full" | awk -F' \\(' '{print $1}' | xargs)
        else
            paradigm="$desc_full"
        fi
        
        # 3. Extract Parameters and Performance Metrics
        run_line=$(grep "Running:" "$log" | tail -n 1)
        N=$(echo "$run_line" | awk '{for(i=1;i<=NF;i++) if($i ~ /bin\/nbody_/) print $(i+1)}')
        
        time=$(grep "Time:" "$log" | tail -n 1 | awk '{print $2}')
        actual_gflops=$(grep "GFLOPs:" "$log" | tail -n 1 | awk '{print $2}')
        
        # 4. Calculate Theoretical Peak and Hardware Efficiency
        peak_gflops="-"
        eff="-"
        
        if [ -n "$actual_gflops" ]; then
            if [[ "$paradigm" == *"CUDA"* ]]; then
                # V100 Peak FP64
                peak_gflops=7833
            else
                # CPU Peak: ~3686 GFLOPs per Node * Number of Nodes
                peak_gflops=$(echo "$nodes * 3686" | bc)
            fi
            
            # Calculate Efficiency percentage (Actual / Peak * 100)
            eff=$(echo "scale=2; ($actual_gflops / $peak_gflops) * 100" | bc 2>/dev/null)
            
            # Add a leading zero if bash bc strips it (e.g., .45 -> 0.45)
            if [[ "$eff" == .* ]]; then eff="0$eff"; fi
        fi

        # 5. Print the formatted row
        if [ -n "$time" ] && [ -n "$actual_gflops" ]; then
            printf "%-8s | %-15s | %-5s | %-6s | %-18s | %-8s | %-8s | %-10s | %-7s | %-6s\n" "$jobid" "$paradigm" "$nodes" "$cores" "$nodelist" "$N" "$time" "$actual_gflops" "$peak_gflops" "$eff"
        fi
    fi
done
echo "======================================================================================================================================="
