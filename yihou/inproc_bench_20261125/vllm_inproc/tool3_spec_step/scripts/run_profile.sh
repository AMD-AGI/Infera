#!/usr/bin/env bash
# Run ONE profiled internal-decode point inside the owned container, snapshotting what ran.
# Port of the GLM-5.2 run_profile.sh (same knobs, guards and markers; delegates the launch to run_decode.sh so the
# digest/owner assertions, no-clobber, snapshot, launch_started.json and ARG CHECK are shared). The measurement knobs
# default to this port's reference point (llying decode leg: DP2 TP1, C448 global = 224 per DP rank, ISL 173000,
# OSL 1024, accept 3.51).
#
# A profiled run is NOT a performance measurement. TPOT / throughput from the emitted
# result_yihou.json must not be quoted as performance; use the non-profiled runs.
#
# Usage: bash run_profile.sh [--dry-run] ITERATION [extra benchmark args...]
# Environment overrides (all optional):
#   CONC=448 ISL=173000 OSL=1024 ACCEPT=3.51 MEMUTIL=0.95 WARMUP=10 TP_SIZE=1 DP_SIZE=2 EP_SIZE=1
#   MAX_STEPS=40 PROFILE_START=20 PROFILE_NUM=5 PROFILE_RANKS=0
#   GRAPH=on|off          off appends --disable-cuda-graph
#   GRAPH_PACKET_CAPTURE  when set, exported as DEBUG_CLR_GRAPH_PACKET_CAPTURE (uncharacterised; avoid)
#   CAPTURE_PROFILE=1     adds --profile-graph-capture-trace (trace of the engine's graph capture)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CONC=${CONC:-448}
ISL=${ISL:-173000}
OSL=${OSL:-1024}
ACCEPT=${ACCEPT:-3.51}
MEMUTIL=${MEMUTIL:-0.95}
WARMUP=${WARMUP:-10}
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-2}
EP_SIZE=${EP_SIZE:-1}
MAX_STEPS=${MAX_STEPS:-40}
PROFILE_START=${PROFILE_START:-20}
PROFILE_NUM=${PROFILE_NUM:-5}
PROFILE_RANKS=${PROFILE_RANKS:-0}
GRAPH=${GRAPH:-on}

DRY=()
if [[ "${1:-}" == --dry-run ]]; then DRY=(--dry-run); shift; fi
[[ $# -ge 1 ]] || { printf 'Usage: [CONC=...] [GRAPH=on|off] bash %s [--dry-run] ITERATION [benchmark arguments...]\n' "$0" >&2; exit 2; }
ITERATION=$1; shift
(( CONC % DP_SIZE == 0 )) || { printf 'CONC must be divisible by dp_size=%s: %s\n' "$DP_SIZE" "$CONC" >&2; exit 2; }
# MAX_STEPS=0 means "run the full OSL" (profile_decode.py's own semantics), so it always
# covers the window; only a positive bound has to reach past it (GLM-5.2 handoff trap #6).
(( MAX_STEPS == 0 || MAX_STEPS >= PROFILE_START + PROFILE_NUM )) || {
    printf 'MAX_STEPS=%s ends before the profile window closes at %s\n' "$MAX_STEPS" "$((PROFILE_START + PROFILE_NUM))" >&2; exit 2; }

ARGS=(--tp-size "$TP_SIZE" --dp-size "$DP_SIZE" --ep-size "$EP_SIZE"
      --batch-size "$CONC" --input-len "$ISL" --output-len "$OSL" --accept-length "$ACCEPT"
      --warmup-steps "$WARMUP" --max-steps "$MAX_STEPS"
      --profile --profile-start-step "$PROFILE_START" --profile-num-steps "$PROFILE_NUM"
      --profile-ranks "$PROFILE_RANKS")
[[ "$GRAPH" == off ]] && ARGS+=(--disable-cuda-graph)
[[ "${CAPTURE_PROFILE:-0}" == 1 ]] && ARGS+=(--profile-graph-capture-trace)

PREFIX=""
if [[ -n "${GRAPH_PACKET_CAPTURE:-}" ]]; then
    printf -v PREFIX 'export DEBUG_CLR_GRAPH_PACKET_CAPTURE=%q; ' "$GRAPH_PACKET_CAPTURE"
fi
KNOBS=$(printf '{"graph":"%s","conc":%s,"dp_size":%s,"ep_size":%s,"max_steps":%s,"profile_start":%s,"profile_num":%s,"graph_packet_capture":"%s","capture_profile":"%s"}' \
    "$GRAPH" "$CONC" "$DP_SIZE" "$EP_SIZE" "$MAX_STEPS" "$PROFILE_START" "$PROFILE_NUM" "${GRAPH_PACKET_CAPTURE:-unset}" "${CAPTURE_PROFILE:-0}")
ARGS_FILES=${ARGS_FILES:-presets/dsv41_llying_decode.args} PREFIX="$PREFIX" KNOBS_JSON="$KNOBS" \
STATUS_EXTRA=',"is_performance_measurement":false' \
    bash "$HERE/run_decode.sh" "${DRY[@]}" "$ITERATION" "${ARGS[@]}" --gpu-memory-utilization "$MEMUTIL" "$@"
