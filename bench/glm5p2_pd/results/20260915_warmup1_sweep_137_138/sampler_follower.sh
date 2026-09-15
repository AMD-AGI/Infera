#!/usr/bin/env bash
# Keeps exactly one sample_metrics.py attached to whichever sweep point is
# currently running, writing into that point's own directory.
#
# Why a follower instead of wiring the sampler into sweep.sh: sweep.sh is
# re-spawned by supervisor.sh and one instance is usually mid-run, and bash
# reads a script incrementally while executing it, so editing sweep.sh in place
# can corrupt the running pass.  This watches from the outside instead.
#
# A point is "active" when its directory exists and it has no VERDICT yet;
# sweep.sh writes VERDICT exactly once, at the end of the point.
set -uo pipefail

CAMPAIGN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFILL=10.245.153.247:19001
DECODE=10.245.157.237:19002
INTERVAL=15

log() { printf '[follower %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

current_point() {
  # Newest point directory that has not been given a verdict yet.
  local d
  for d in $(ls -dt "$CAMPAIGN"/p*d*/c[0-9]* 2>/dev/null); do
    [[ -d "$d" ]] || continue
    [[ -f "$d/VERDICT" ]] && continue
    printf '%s\n' "$d"
    return 0
  done
  return 1
}

sampler_pid=""
sampler_point=""

cleanup() {
  [[ -n "$sampler_pid" ]] && kill "$sampler_pid" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

while true; do
  point="$(current_point || true)"

  # The attached sampler's point finished (or vanished): detach.
  if [[ -n "$sampler_pid" ]] && { [[ "$point" != "$sampler_point" ]] || ! kill -0 "$sampler_pid" 2>/dev/null; }; then
    kill "$sampler_pid" 2>/dev/null
    log "detached from $(basename "$(dirname "$sampler_point")")/$(basename "$sampler_point")"
    sampler_pid=""
    sampler_point=""
  fi

  if [[ -n "$point" && -z "$sampler_pid" ]]; then
    python3 "$CAMPAIGN/sample_metrics.py" --prefill "$PREFILL" --decode "$DECODE" \
      --interval "$INTERVAL" --out "$point/metrics_sample.log" \
      >>"$CAMPAIGN/sampler_follower.err" 2>&1 &
    sampler_pid=$!
    sampler_point="$point"
    log "attached to $(basename "$(dirname "$point")")/$(basename "$point") (pid $sampler_pid)"
  fi

  sleep 20
done
