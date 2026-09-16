#!/usr/bin/env bash
# Keeps exactly one sample_metrics.py attached to whichever point is currently
# running, writing into that point's own directory.
#
# Why a follower instead of wiring the sampler into sweep.sh: bash reads a
# script incrementally while executing it, so editing sweep.sh in place can
# corrupt a running pass.  This watches from the outside instead.
#
# A point is "active" when its directory exists and it has no VERDICT yet;
# sweep.sh writes VERDICT exactly once, at the end of the point.
#
# Endpoints follow topology_rows: the engine port is ENGINE_PORT_BASE (19001)
# plus the row index in the topology file, so the mapping is per topology, not
# per role.
set -uo pipefail

CAMPAIGN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IP_137=10.245.153.247
IP_138=10.245.157.237
IP_140=10.245.159.30
IP_141=10.245.148.188
INTERVAL=15

log() { printf '[follower %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

endpoints_for() {  # $1 topology
  case "$1" in
    p8x2d8)   echo "p0=$IP_137:19001 p1=$IP_140:19002 d0=$IP_138:19003" ;;
    p8x3d8)   echo "p0=$IP_137:19001 p1=$IP_140:19002 p2=$IP_141:19003 d0=$IP_138:19004" ;;
    p8x2d8x2) echo "p0=$IP_137:19001 p1=$IP_140:19002 d0=$IP_138:19003 d1=$IP_141:19004" ;;
  esac
}

current_point() {
  local d
  for d in $(ls -dt "$CAMPAIGN"/p8x*/c[0-9]* 2>/dev/null); do
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

  if [[ -n "$sampler_pid" ]] && { [[ "$point" != "$sampler_point" ]] || ! kill -0 "$sampler_pid" 2>/dev/null; }; then
    kill "$sampler_pid" 2>/dev/null
    log "detached from $sampler_point"
    sampler_pid=""
    sampler_point=""
  fi

  if [[ -n "$point" && -z "$sampler_pid" ]]; then
    topo="$(basename "$(dirname "$point")")"
    args=()
    for ep in $(endpoints_for "$topo"); do args+=(--endpoint "$ep"); done
    if [[ ${#args[@]} -gt 0 ]]; then
      python3 "$CAMPAIGN/sample_metrics.py" "${args[@]}" \
        --interval "$INTERVAL" --out "$point/metrics_sample.log" \
        >>"$CAMPAIGN/sampler_follower.err" 2>&1 &
      sampler_pid=$!
      sampler_point="$point"
      log "attached to $topo/$(basename "$point") (pid $sampler_pid)"
    fi
  fi

  sleep 20
done
