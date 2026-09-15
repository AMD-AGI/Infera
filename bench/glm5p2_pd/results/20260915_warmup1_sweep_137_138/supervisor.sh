#!/usr/bin/env bash
# Keeps the sweep going unattended.
#
# sweep.sh reads its plan into an array at startup, so points appended to
# points.tsv while it runs are not picked up.  Re-invoking it is safe and cheap:
# any point that already has a VERDICT is skipped, so each pass only runs what is
# still missing.  That makes "append a point, it gets run" work without having to
# babysit the process.
set -uo pipefail

CAMPAIGN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CAMPAIGN"
MAX_PASSES=12

log() { printf '[supervisor %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Never run two sweeps at once.  An earlier version matched on the absolute
# path, which missed a sweep started as ./sweep.sh; two sweeps then ran together
# and the second one rm -rf'd the point the first was still writing, destroying a
# 63-minute C64 run.  Match the bare script name, and also refuse to start while
# any topology container is still up.
sweep_running() { pgrep -f "sweep[.]sh" | grep -qv "^$$\$"; }
nodes_busy() {
  local n
  for n in crsuse2-m2m-137 crsuse2-m2m-138; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" \
      'docker ps --format "{{.Names}}" | grep -q glm52-pd' </dev/null 2>/dev/null && return 0
  done
  return 1
}
while sweep_running || nodes_busy; do
  log "a sweep is still running or the nodes are busy; waiting"
  sleep 60
done

missing_points() {
  local n=0 phase topo conc mem extra
  while IFS=$'\t' read -r phase topo conc mem extra; do
    [[ "$phase" =~ ^#|^$ ]] && continue
    [[ -z "${topo:-}" || -z "${conc:-}" ]] && continue
    [[ -f "$CAMPAIGN/$topo/c$conc/VERDICT" ]] || n=$((n + 1))
  done <points.tsv
  echo "$n"
}

for pass in $(seq 1 "$MAX_PASSES"); do
  left="$(missing_points)"
  if [[ "$left" -eq 0 ]]; then
    log "every planned point has a verdict; done"
    break
  fi
  log "pass $pass: $left point(s) still missing a verdict"
  ./sweep.sh >>"$CAMPAIGN/sweep_supervised.log" 2>&1 </dev/null
  after="$(missing_points)"
  if [[ "$after" -eq "$left" ]]; then
    # A pass that completes nothing means the points are failing before they can
    # even record a verdict; looping would just burn the remaining hours.
    log "pass $pass made no progress ($left still missing); stopping"
    break
  fi
  sleep 20
done

log "final index:"
cat "$CAMPAIGN/index.tsv"
