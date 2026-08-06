#!/bin/bash
# Poll one arm's two legs to readiness.
#
# Polls the HTTP endpoint from INSIDE each container rather than grepping the log
# for "ready to roll": the logs are appended to across runs, so a grep matches a
# PREVIOUS run's line within seconds and reports a leg ready that is still
# loading weights.
#
# Never curl a PD leg's port from outside its own container -- that hangs.
# /health on the leg itself is fine from inside; the ROUTER is what clients use.
#
# Usage: PJOB=..(prefill) PIP=.. DJOB=..(decode) DIP=.. ab_wait.sh [timeout_s]
set -u
TMO="${1:-2400}"
PJOB="${PJOB:?}"; PIP="${PIP:?}"
DJOB="${DJOB:?}"; DIP="${DIP:?}"
CTR="${CTR:-agbench_mtp}"

probe() {  # job ip port -> prints 1 if healthy
  spur exec "$1" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec $CTR bash -c 'curl -sf -m5 http://$2:$3/health >/dev/null 2>&1 && echo 1 || echo 0'" 2>/dev/null \
    | grep -v libtinfow | tr -d '[:space:]'
}

t0=$(date +%s)
while :; do
  p=$(probe "$PJOB" "$PIP" 30000)
  d=$(probe "$DJOB" "$DIP" 30001)
  el=$(( $(date +%s) - t0 ))
  echo "  [${el}s] prefill=${p:-?} decode=${d:-?}"
  [ "$p" = "1" ] && [ "$d" = "1" ] && { echo "BOTH READY after ${el}s"; exit 0; }
  [ "$el" -gt "$TMO" ] && { echo "TIMEOUT after ${el}s"; exit 1; }
  sleep 30
done
