#!/bin/bash
# PD mini-LB router (intra-node). 1 prefill + 1 decode over loopback.
# Usage (inside container): bash pd_router.sh
set -x
P_IPS="${P_IPS:-127.0.0.1:30000}"
D_IPS="${D_IPS:-127.0.0.1:30100}"
BOOTSTRAP="${BOOTSTRAP:-8998}"
PORT="${PORT:-8100}"
WORK="${WORK:-/home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round3_pd_disagg}"
LOG="${LOG:-$WORK/router.log}"
mkdir -p "$(dirname "$LOG")"
PRE_ARGS=()
for p in $P_IPS; do case "$p" in *:*) purl="http://$p";; *) purl="http://$p:30000";; esac; PRE_ARGS+=(--prefill "$purl" "$BOOTSTRAP"); done
DEC_ARGS=()
for d in $D_IPS; do case "$d" in *:*) durl="http://$d";; *) durl="http://$d:30100";; esac; DEC_ARGS+=(--decode "$durl"); done
exec python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  "${PRE_ARGS[@]}" \
  "${DEC_ARGS[@]}" \
  --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1
