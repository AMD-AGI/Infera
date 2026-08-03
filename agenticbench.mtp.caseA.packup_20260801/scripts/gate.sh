#!/bin/bash
# The leg gate. Every row, before any measured window.
#
# Two things this does that a naive grep does not:
#   * reads the logs through `strings` -- they contain binary bytes, and a plain
#     `grep -c` returns 0, which reads exactly like "the bad thing never
#     happened";
#   * checks `Errno 98` AFTER the ready line. A --kv-snapshot-port collision
#     lets a leg log "ready to roll" and THEN die during etcd registration, so
#     it looks healthy and simply never appears in /v1/workers.
#
# Usage: gate.sh [tag]
set -u
TAG="${1:-g1}"
W=/shared_nfs/yihou_agbench_mtp
PJOB="${PJOB:-24300}"; PIP="${PIP:-10.245.157.89}"
DJOB="${DJOB:-24301}"; DIP="${DIP:-10.245.146.87}"

row() { printf '  %-46s %s\n' "$1" "$2"; }

for pair in "prefill $PJOB" "decode $DJOB"; do
  set -- $pair; ROLE=$1; JOB=$2
  LOG=$W/logs/${TAG}_${ROLE}.log
  echo "===== $ROLE ($LOG) ====="
  if [ ! -s "$LOG" ]; then echo "  LOG MISSING OR EMPTY"; continue; fi
  S=$(mktemp); strings "$LOG" > "$S"

  row "ready to roll"              "$(grep -c 'ready to roll' "$S")"
  row "Memory access fault  (want 0)" "$(grep -c 'Memory access fault' "$S")"
  row "Scheduler hit an exception (want 0)" "$(grep -c 'Scheduler hit an exception' "$S")"
  row "Traceback (want 0)"         "$(grep -c 'Traceback' "$S")"
  row "infera-kvd adapter"         "$(grep -ci 'infera-kvd' "$S")"
  row "context_length=262144"      "$(grep -c 'context_length=262144' "$S")"
  row "dp_size=8"                  "$(grep -c 'dp_size=8' "$S")"
  # server_args echoes it QUOTED: speculative_algorithm='EAGLE'. Matching a bare
  # '=EAGLE' misses it and reports MTP absent on a leg that is running it.
  row "speculative EAGLE (prefill 0 / decode 1)" "$(grep -c "speculative_algorithm='EAGLE'" "$S")"
  row "disable_custom_all_reduce (decode want True)" "$(grep -o 'disable_custom_all_reduce=[A-Za-z]*' "$S" | head -1)"
  row "enable_hierarchical_cache (prefill True/decode False)" "$(grep -o 'enable_hierarchical_cache=[A-Za-z]*' "$S" | head -1)"
  row "MC_FORCE_TCP (want 0)"      "$(grep -c 'MC_FORCE_TCP' "$S")"
  row "mlx5_0 mentions"            "$(grep -c 'mlx5_0' "$S")"

  # Errno 98 strictly AFTER the ready line -- see the header.
  RL=$(grep -n 'ready to roll' "$S" | head -1 | cut -d: -f1)
  if [ -n "$RL" ]; then
    row "Errno 98 AFTER ready (want 0)" "$(tail -n +"$RL" "$S" | grep -c 'Errno 98')"
  else
    row "Errno 98 AFTER ready" "n/a (not ready yet)"
  fi

  row "accept len (last 3)" "$(grep -ao 'accept len: [0-9.]*' "$S" | tail -3 | tr '\n' ' ')"
  rm -f "$S"

  echo "  --- live processes ---"
  spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec agbench_mtp bash -c 'echo -n \"    scheduler_DP procs: \"; ps aux | grep -c \"[s]glang::scheduler_DP\"'" 2>&1 | grep -v libtinfow
done

echo "===== router view ====="
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec agbench_mtp bash -c 'curl -sf -m5 http://$PIP:8190/v1/workers | head -c 2000; echo'" 2>&1 | grep -v libtinfow
