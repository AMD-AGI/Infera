#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: cut a window out of a running load and capture one torch trace per PD role.
# why : prefill and decode overlap in wall-clock time, so a single trace cannot be split by
#       role after the fact. The router's role selector is what guarantees the prefill
#       directory holds prefill operators and nothing else.
# how : DO NOT run this directly — run it through cluster/<your-cluster>.sh, e.g.
#         PROFILE_DECODE=1 bash cluster/<your-cluster>.sh up
#         nohup bash cluster/<your-cluster>.sh bench 64 &        # load, in the background
#         PROFILE_DECODE=1 bash cluster/<your-cluster>.sh capture
#       The same PROFILE_* values must be given to `up` and to `capture`: `up` decides
#       whether the control plane exists at all, `capture` reads them to pick roles.
#
# Knobs (all optional):
#   WARMUP_S=60     seconds to let the load settle before opening the window
#   WINDOW_S=20     length of the sampled window
#   TRACE_OUT=DIR   shared host path mounted rw into both engine containers (default ../profiles)
#   REQUIRE_LOAD=0  skip the "is a bench running" check (you are driving load some other way)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

require_env PREFILL_NODE; require_env DECODE_NODE; require_env PREFILL_IP
SSH_CMD="${SSH_CMD:-ssh -o StrictHostKeyChecking=no}"
on(){ local h="$1"; shift; $SSH_CMD "$h" "$*" </dev/null; }

URL="http://$PREFILL_IP:$ROUTER_PORT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
TRACE_ROOT="${TRACE_OUT:-$(cd "$DIR/.." && pwd)/profiles}"
OUT="$TRACE_ROOT/$RUN_TS"
CTR_TRACE="$OUT"                                       # same bind-mounted path in both containers
WARMUP_S="${WARMUP_S:-60}"
WINDOW_S="${WINDOW_S:-20}"

ROLES=()
[ "${PROFILE_PREFILL:-0}" = "1" ] && ROLES+=(prefill)
[ "${PROFILE_DECODE:-0}"  = "1" ] && ROLES+=(decode)
[ "${#ROLES[@]}" -gt 0 ] || die "set PROFILE_PREFILL=1 and/or PROFILE_DECODE=1 — the same values the stack was brought up with"

node_of(){ [ "$1" = "prefill" ] && echo "$PREFILL_NODE" || echo "$DECODE_NODE"; }

# what: ship a script INTO the container by reading it from stdin.
# why : a capture step is several commands that have to run as one unit, and passing them as
#       a string means quoting them through four layers (host shell -> $SSH_CMD -> docker
#       exec -> bash -c). base64 is [A-Za-z0-9+/=] only, so nothing in the payload can be
#       re-interpreted on the way. Same reason up.sh stages /run_kvd.sh as a file.
put_script(){
  local h="$1" path="$2" b64
  b64="$(base64 -w0)" || die "base64 -w0 failed on this host"
  on "$h" "docker exec $CTR bash -c 'echo $b64 | base64 -d > $path && chmod +x $path'" \
    || die "could not stage $path inside $CTR on $h"
}

bench_running(){ on "$PREFILL_NODE" "docker exec $CTR pgrep -f 'sglang.bench_serving' >/dev/null 2>&1"; }

# ---- 1/6 preflight -------------------------------------------------------------------------
log "=== 1/6 preflight (roles: ${ROLES[*]}) ==="

# Direct output is safe only when the exact host path is a writable bind mount. Without this
# check Docker would create the directory in the container layer and capture would appear to
# succeed while the host sees nothing.
for r in "${ROLES[@]}"; do
  h="$(node_of "$r")"
  mounted="$(on "$h" "docker inspect -f '{{range .Mounts}}{{if eq .Destination \"$TRACE_ROOT\"}}{{.RW}}{{end}}{{end}}' $CTR" | tr -d '\r\n')"
  [ "$mounted" = "true" ] \
    || die "TRACE_OUT is not mounted rw in $CTR on $h: $TRACE_ROOT
Re-run 'bash cluster/<your-cluster>.sh down', then bring up the stack with the same TRACE_OUT."
done

# Probe the control plane with a role that cannot exist. In infera/server/app.py the handler
# checks the 403 gate FIRST and rejects the role BEFORE selecting or contacting any worker,
# so this tells us whether profiling is enabled without touching a running profile:
#   400 -> enabled (it got as far as validating the role)   403 -> not enabled
code="$(on "$PREFILL_NODE" "docker exec $CTR curl -s -o /dev/null -w '%{http_code}' -m 10 \
  -X POST '$URL/v1/admin/profile/start?role=__probe__'" | tr -d '\r\n')"
case "$code" in
  400) log "  control plane ON (probe -> 400 invalid role, as expected)" ;;
  403) die "the router has profiling disabled. Re-run bring-up with the switch set:
  PROFILE_PREFILL=1 PROFILE_DECODE=1 bash cluster/<your-cluster>.sh up" ;;
  000|"") die "no answer from the router at $URL — is the stack up?" ;;
  *)   warn "  unexpected probe status '$code'; continuing, but read /tmp/router.log if start fails" ;;
esac

# A role with no ACTIVE worker makes /v1/admin/profile/start 404. Catch it here rather than
# after the warm-up, which is a minute of waiting for nothing.
workers="$(on "$PREFILL_NODE" "docker exec $CTR curl -s -m10 $URL/v1/workers" | tr -d ' \r')"
for r in "${ROLES[@]}"; do
  n="$(printf '%s' "$workers" | grep -c "\"disagg_mode\":\"$r\"")"
  [ "${n:-0}" -ge 1 ] || die "no $r worker registered with the router — start?role=$r would 404"
  log "  $r workers registered: $n"
done

if [ "${REQUIRE_LOAD:-1}" = "1" ] && ! bench_running; then
  die "no load in flight. An idle window profiles an empty scheduler loop, not your model.
Start one first, in another shell or in the background:
  nohup bash cluster/<your-cluster>.sh bench 64 &
Give it enough requests to outlast WARMUP_S + WINDOW_S (${WARMUP_S}s + ${WINDOW_S}s here) — N=<count> raises it.
Set REQUIRE_LOAD=0 if you are driving load some other way."
fi

# ---- 2/6 warm-up ---------------------------------------------------------------------------
log "=== 2/6 warm-up ${WARMUP_S}s ==="
# Profile steady state, not ramp-up: the first requests hit a cold radix cache and a decode
# leg with nothing queued, and their step times are not the ones worth optimising.
sleep "$WARMUP_S"
if [ "${REQUIRE_LOAD:-1}" = "1" ] && ! bench_running; then
  warn "  the load finished during warm-up — the window will catch an idle engine."
  warn "  Send more requests (N=<count>) or shorten WARMUP_S."
fi

# ---- 3/6 output directories ----------------------------------------------------------------
log "=== 3/6 output directories ==="
# SGLang does not create output_dir. When it is missing the export fails inside the profiler
# callback long after /start_profile has already answered 200 — you find out at the end, with
# an empty directory and no error anywhere.
for r in "${ROLES[@]}"; do
  on "$(node_of "$r")" "docker exec $CTR mkdir -p $CTR_TRACE/$r" \
    || die "could not create $CTR_TRACE/$r in $CTR on $(node_of "$r")"
done

# ---- 4/6 start -----------------------------------------------------------------------------
log "=== 4/6 start (window ${WINDOW_S}s) ==="
# The request body. Three keys are not obvious:
#   with_stack=false    MUST be explicit. SGLang's default is True (`with_stack if
#                       with_stack is not None else True`), which adds millions of
#                       python_function events — measured 122 MB vs 14 MB per rank, ~80% of
#                       the file — that no downstream analysis reads.
#   record_shapes=true  operator input shapes; without them no FLOPs/bytes roofline is possible.
#   activities          spelled out rather than left null, so the engine does not choose.
body(){ printf '{"output_dir":"%s/%s","record_shapes":true,"with_stack":false,"activities":["CPU","GPU"]}' "$CTR_TRACE" "$1"; }

# Both starts are issued from inside ONE container, backgrounded, and waited on together.
# Issuing them from the host in sequence puts a full round-trip between the two roles'
# windows; from one shell the skew is sub-millisecond. Both go to the router regardless of
# role — it is the router that fans out to the worker on the other node.
{
  echo '#!/bin/bash'
  for r in "${ROLES[@]}"; do
    printf "curl -sS -m 60 -X POST -H 'Content-Type: application/json' -d '%s' '%s/v1/admin/profile/start?role=%s' > /tmp/capture_start_%s.out 2>&1 &\n" \
      "$(body "$r")" "$URL" "$r" "$r"
  done
  echo 'wait'
  for r in "${ROLES[@]}"; do
    printf 'printf "  %%-8s " %s; cat /tmp/capture_start_%s.out; echo\n' "$r" "$r"
  done
} | put_script "$PREFILL_NODE" /run_capture_start.sh
on "$PREFILL_NODE" "docker exec $CTR /run_capture_start.sh" \
  || die "profile start failed — see the output above and /tmp/router.log in $CTR"

sleep "$WINDOW_S"

# ---- 5/6 stop ------------------------------------------------------------------------------
log "=== 5/6 stop ==="
# One call, not one per role. Two sequential stops would leave the first role recording for
# however long the first round-trip took, which is exactly the skew step 4 avoided. With both
# roles selected that means a broadcast (no selector); with one, a selector, so the other leg
# is not sent a stop it never started.
sel=""; [ "${#ROLES[@]}" -eq 1 ] && sel="?role=${ROLES[0]}"
# The router's own HTTP client has a hardcoded 30s read timeout (infera/server/app.py). A stop
# that has to flush eight ranks can exceed it, and the engine keeps writing regardless — so a
# failure here is not evidence that the stop failed. The flush check below is what decides.
on "$PREFILL_NODE" "docker exec $CTR curl -sS -m 180 -X POST \
  -H 'Content-Type: application/json' -d '{}' '$URL/v1/admin/profile/stop$sel'" \
  || warn "  stop returned an error — this is often just the router's 30s read timeout; checking the files"
echo

# ---- 6/6 flush -------------------------------------------------------------------------------
log "=== 6/6 flush ==="
# A trace file appearing is not a trace file being finished: torch writes it from the profiler
# callback after stop has already returned. Wait for the byte count to stop moving rather than
# for a fixed sleep, which is either too short or wastes minutes.
wait_flush(){
  local h="$1" role="$2" path="$CTR_TRACE/$2" prev=-1 cur
  for _ in $(seq 1 "${FLUSH_TRIES:-30}"); do
    cur="$(on "$h" "docker exec $CTR bash -c 'du -sb $path 2>/dev/null | cut -f1'" | tr -d '\r\n')"
    cur="${cur:-0}"
    if [ "$cur" -gt 0 ] && [ "$cur" = "$prev" ]; then
      log "  $h/$role: $path stable at $cur bytes"; return 0
    fi
    prev="$cur"; sleep "${FLUSH_GAP:-10}"
  done
  warn "  $h/$role: $path never stopped growing (last: $prev bytes)"
  return 1
}

for r in "${ROLES[@]}"; do
  wait_flush "$(node_of "$r")" "$r"
done
log "traces available -> $OUT"

log "capture done -> $OUT"
exit 0
