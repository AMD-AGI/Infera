#!/usr/bin/env bash
# Summary of one scoring point from its pd_perf_chain.sh log: the throughput row (summarize_points.py) and the PD
# bottleneck numbers over the AgentX measurement window (analyze_pd_logs.py: decode step time by the busiest rank's
# running requests, prefill beat). Usage: score_summary.sh LOG_ROOT/perf/LABEL/c104.log
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log="$1"
WORK_DIR="${WORK_DIR:?}"
run="$(grep -ao 'run=[^ ]*' "$log" | head -1 | cut -d= -f2)"
[[ -n "$run" ]] || { echo "no run dir in $log"; exit 1; }
b="$run/agentx/benchmark.log"
t0="$(grep -am1 'Phase profiling (profiling) started' "$b" | cut -c1-8)"
t1="$(grep -am1 'Phase profiling (profiling) sending complete' "$b" | cut -c1-8)"
day1="$(date -u -r "$b" +%F)" day="$day1"
[[ "$t1" < "$t0" ]] && day="$(date -u -d "$day1 - 1 day" +%F)"  # the window crossed midnight
echo "== run=$run window $day $t0 .. $day1 $t1"
tmp="$(mktemp -d)"
ln -s "$run"/agentx/agentx_conc*.json "$tmp/"; ln -s "$run/agentx/aiperf_artifacts/profile_export_aiperf.json" "$tmp/"
echo "-- tok/s/GPU | out tok/s/GPU | req/s | done / dropped | intvty p50 / p90 | ITL p50 / p90 | TTFT p50 / p90 / mean | E2E p90 | cache hit"
python3 "$here/summarize_points.py" "$tmp" 2>&1 | tail -1
rm -rf "$tmp"
echo "-- analyze_pd_logs"
python3 "$here/analyze_pd_logs.py" "$run" "${day}T$t0" "${day1}T$t1" 2>&1 | tail -6
