#!/usr/bin/env bash
# Take one torch trace of TRACE_S seconds once the engine is decode-only: every engine has requests running, none
# waiting, and the summed prompt-token counter has not moved for two consecutive polls (10 s apart). The engine needs
# --profiler-config. With several API servers the stats log lines are off, so this reads /metrics (multiprocess
# Prometheus, labelled per engine).
#   trace_when_decoding.sh PORT [TRACE_S=3] [TIMEOUT=3600]
set -uo pipefail
port=$1 secs=${2:-3} timeout=${3:-3600}
t0=$(date +%s) quiet=0 last=-1
sleep 30
while (( $(date +%s) - t0 < timeout )); do
  read -r prompt waiting running engines < <(curl -sf -m 10 "http://127.0.0.1:$port/metrics" | awk '
    /^vllm:prompt_tokens_total\{/ {p += $2}
    /^vllm:num_requests_waiting\{/ {w += $2}
    /^vllm:num_requests_running\{/ {n++; if ($2 > 0) r++}
    END {printf "%d %d %d %d\n", p, w, r, n}' || echo "-1 -1 0 0")
  if (( prompt >= 0 && waiting == 0 && running > 0 && running == engines && prompt == last )); then
    quiet=$((quiet + 1))
  else
    quiet=0
  fi
  last=$prompt
  if (( quiet >= 2 )); then
    echo "decode-only at $(date -u +%T) (prompt tokens $prompt, engines running $running/$engines)"
    curl -sf -m 60 -X POST "http://127.0.0.1:$port/start_profile" && echo "start_profile $(date -u +%T)"
    sleep "$secs"
    curl -sf -m 900 -X POST "http://127.0.0.1:$port/stop_profile" && echo "stop_profile $(date -u +%T)"
    exit 0
  fi
  sleep 10
done
echo "never decode-only within $timeout s"
exit 0
