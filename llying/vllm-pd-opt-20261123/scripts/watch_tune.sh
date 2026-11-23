#!/usr/bin/env bash
# Login-node watcher for a running tune_decode_gemm.sh: every 5 min one "TUNE <time> ..." line with the profiled-task count
# (one torch-profiler session per candidate), its rate and the rows of the tuned table so far; "TUNE_ROWS" when the table
# grows (the tuner writes it once per --batch shapes), "TUNE_END" when tune-summary.txt appears. Small greps only.
#   watch_tune.sh TUNE_DIR [LOG=tune-retry.log] [TOTAL_TASKS]
d=$1 log=${2:-tune-retry.log} total=${3:-0}
last=0 lastrows=0 t_last=$(date +%s)
while true; do
    n=$(grep -a -c "profiler_start" "$d/$log" 2>/dev/null); n=${n:-0}
    now=$(date +%s); rate=$(( (n - last) * 60 / (now - t_last + 1) )); last=$n; t_last=$now
    rows=$(( $(wc -l < "$d/glm52_decode_bf16_tuned_gemm.csv" 2>/dev/null || echo 1) - 1 ))
    eta=""; (( total > 0 && rate > 0 )) && eta=" eta $(( (total - n) / rate )) min"
    echo "TUNE $(date -u +%T) tasks $n rate ${rate}/min rows $rows$eta"
    (( rows > lastrows )) && { echo "TUNE_ROWS $(date -u +%T) $rows"; lastrows=$rows; }
    [ -f "$d/tune-summary.txt" ] && { echo "TUNE_END $(date -u +%T) $(cat "$d/tune-summary.txt")"; break; }
    sleep 300
done
