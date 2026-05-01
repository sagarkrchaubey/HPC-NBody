#!/usr/bin/env bash
# summary.sh — fast, fixed-width, compact hardware

set -Eeuo pipefail
shopt -s nullglob
LOG_DIR="${LOG_DIR:-logs}"

if [[ -t 1 ]]; then
  BOLD=$(tput bold); RESET=$(tput sgr0)
  C_HEAD=$(tput setaf 6); C_TIME=$(tput setaf 3); C_FLOP=$(tput setaf 2)
else
  BOLD=""; RESET=""; C_HEAD=""; C_TIME=""; C_FLOP=""
fi

logs=( "$LOG_DIR"/*run_*.log )
(( ${#logs[@]} )) || { echo "No logs in $LOG_DIR"; exit 0; }

# --- sacct cache ---
jobids=()
for f in "${logs[@]}"; do jobids+=( "${f##*/}" ); done
jobids=( "${jobids[@]%%_run*}" )
job_csv=$(IFS=,; echo "${jobids[*]}")

declare -A NODES NLIST
if command -v sacct >/dev/null; then
  while IFS='|' read -r jid alloc nlist; do
    NODES["$jid"]=${alloc:-1}
    NLIST["$jid"]=${nlist:--}
  done < <(sacct -j "$job_csv" --format=JobID,AllocNodes,NodeList -X -n -P 2>/dev/null)
fi

# --- header ---
printf "${BOLD}${C_HEAD}%-8s | %-16s | %-12s | %12s | %10s | %10s | %s${RESET}\n" \
  "Job ID" "Paradigm" "Hardware" "N Bodies" "Time(s)" "GFLOPs" "NodeList"
printf '%*s\n' 100 '' | tr ' ' '-'

# --- process ---
for log in "${logs[@]}"; do
  jobid="${log##*/}"; jobid="${jobid%%_run*}"
  nodes=${NODES[$jobid]:-1}
  nlist=${NLIST[$jobid]:--}

  IFS=$'\t' read -r paradigm config N time gflops <<< "$(
    awk '
      /Description :/ { sub(/.*Description : */,""); d=$0; gsub(/^[ \t]+|[ \t]+$/,"",d) }
      /Running:.*bin\/nbody_/ { for(i=1;i<=NF;i++) if($i~/bin\/nbody_/) N=$(i+1) }
      /^Time:/ { t=$2 }
      /^GFLOPs:/ { g=$2 }
      END {
        p=d; c="-";
        if (match(d, /\(([^)]+)\)/)) { p=substr(d,1,RSTART-1); c=substr(d,RSTART+1,RLENGTH-2) }
        gsub(/^[ \t]+|[ \t]+$/,"",p); gsub(/^[ \t]+|[ \t]+$/,"",c)
        printf "%s\t%s\t%s\t%s\t%s", p,c,N,t,g
      }' "$log"
  )"

  [[ -z "$time" ]] && continue

  # --- compact hardware ---
  short=""
  if [[ $config =~ (V100|A100|H100|MI250|MI300) ]]; then
    short="${BASH_REMATCH[1]}"
  elif [[ $config =~ ([0-9]+)[^0-9]*Ranks? ]]; then
    r="${BASH_REMATCH[1]}"
    if [[ $config =~ ([0-9]+)[^0-9]*Threads? ]]; then
      t="${BASH_REMATCH[1]}"
      short="${r}x${t}"
    else
      short="${r}R"
    fi
  elif [[ $config != "-" ]]; then
    short=$(echo "$config" | tr -d ' ' | cut -c1-10)
  fi

  hardware="${nodes}N"
  [[ -n "$short" ]] && hardware+=" $short"


  printf "%-8s | %-16s | %-12s | %12s | ${C_TIME}%10.4f${RESET} | ${C_FLOP}%10.4f${RESET} | %s\n" \
    "$jobid" "$paradigm" "$hardware" "${N:--}" "$time" "$gflops" "$nlist"
done | LC_ALL=C sort -V

printf '%*s\n' 92 '' | tr ' ' '-'
