#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: replay a Mooncake-format production trace against the router with AIPerf, at the
#       timestamps the trace recorded.
# why : bench.sh sweeps `--dataset-name random`, which builds every prompt independently — no
#       shared prefix by construction, so it cannot measure prefix reuse at all. A Mooncake
#       trace carries hash_ids and AIPerf expands each one into a real token block, so the
#       radix cache, kvd and kv-aware routing are all exercised. Read "Trace replay" in the
#       README before quoting any number from here against a bench.sh number.
# how : DO NOT run this directly — run it through cluster/<your-cluster>.sh:
#         bash cluster/<your-cluster>.sh trace_replay prepare   # read-only checks + shape report
#         bash cluster/<your-cluster>.sh trace_replay run       # send the load
#         bash cluster/<your-cluster>.sh trace_replay           # both, in order
#
# AIPerf runs in its OWN container, never in $CTR: the engine image ships Python 3.10 and AIPerf
# needs >= 3.11. There is deliberately no row-count knob either — the loader applies the
# timestamp window before prompt synthesis, so START_MS/END_MS saves everything a pre-sliced
# file would, without cutting a session in half on a trace that carries session_id.
#
# Knobs (all optional):
#   START_MS/END_MS   replay only a window of the trace, in trace milliseconds
#   MAX_CONC=256      in-flight ceiling; 0 omits the flag. Not neutral either way — see README
#   WORKERS=16        AIPerf worker processes. Matters when $AIPERF_NODE is a serving node
#   BLOCK_SIZE=512    tokens per hash id, i.e. the block size the TRACE was recorded at
#   REQ_TIMEOUT=900   per-request timeout in seconds
#   SPEEDUP=<float>   scale every timestamp (2.0 = twice as fast). Also reshapes hash_ids
#   IGNORE_EOS=1      send ignore_eos so output length equals the trace's output_length
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

require_env PREFILL_NODE; require_env PREFILL_IP
require_env MODEL       "model directory — AIPerf loads its tokenizer from here"
require_env MODEL_MOUNT "host dir holding the weights, bind-mounted into the AIPerf container"
require_env AIPERF_TRACE "absolute path to the Mooncake-format trace JSONL, readable on \$AIPERF_NODE"

SSH_CMD="${SSH_CMD:-ssh -o StrictHostKeyChecking=no}"
on(){ local h="$1"; shift; $SSH_CMD "$h" "$*" </dev/null; }

AIPERF_IMAGE="${AIPERF_IMAGE:-nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0}"
AIPERF_NODE="${AIPERF_NODE:-$PREFILL_NODE}"
AIPERF_OUT="${AIPERF_OUT:-$(cd "$DIR/.." && pwd)/aiperf}"
URL="http://$PREFILL_IP:$ROUTER_PORT"

BLOCK_SIZE="${BLOCK_SIZE:-512}"
MAX_CONC="${MAX_CONC:-256}"
WORKERS="${WORKERS:-16}"
REQ_TIMEOUT="${REQ_TIMEOUT:-900}"

TRACE_BASE="$(basename "$AIPERF_TRACE")"; TRACE_BASE="${TRACE_BASE%.jsonl}"
SHAPE_DIR="$AIPERF_OUT/shapes"
SHAPE="$SHAPE_DIR/${TRACE_BASE}.shape.json"

# ---- container plumbing ----------------------------------------------------------------------
# Source and destination are always the same path, so a path means the same thing inside and
# outside. Deduped because docker rejects two mounts on one destination, and AIPERF_TRACE
# living under AIPERF_OUT is a natural layout.
MOUNT_ARGS=""
MOUNT_SEEN=""
add_mount(){
  local src="$1" ro="${2:-}"
  case "$MOUNT_SEEN" in *"|$src|"*) return 0 ;; esac
  MOUNT_SEEN="$MOUNT_SEEN|$src|"
  MOUNT_ARGS="$MOUNT_ARGS -v '$src:$src${ro:+:ro}'"
}

# what: run one command in the NGC image on $AIPERF_NODE.
# why : four of these settings are load-bearing and each fails silently if dropped.
#   MMAP_CACHE_DIR / MMAP_BASE_PATH  the image sets HOME=/app, so the dataset cache would live
#       inside a --rm container (re-tokenizing the trace every run) and the mmap data files
#       would land on the node's root fs through the overlay — README note 7's failure, reached
#       from a different direction.
#   --user + HOME=/tmp  the image runs as uid 1000 but writes into a host-owned directory, so
#       it needs this host's ids, resolved ON $AIPERF_NODE.
#   HF_HUB_OFFLINE  turns a mistyped --tokenizer into an instant failure, not a network timeout.
# note: $1 is single-quoted on the way through, so it may contain no single quote and no path
#       with a space. Adding quotes does not help — the entrypoint consumes them.
aiperf_in_image(){
  on "$AIPERF_NODE" "docker run --rm --network=host --user \$(id -u):\$(id -g) \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
    -e AIPERF_DATASET_MMAP_CACHE_DIR='$AIPERF_OUT/.mmap_cache' \
    -e AIPERF_DATASET_MMAP_BASE_PATH='$AIPERF_OUT/.mmap' \
    $MOUNT_ARGS \
    $AIPERF_IMAGE '$1'"
}

add_mount "$AIPERF_OUT"
add_mount "$MODEL_MOUNT" ro
add_mount "$(dirname "$AIPERF_TRACE")" ro

# ---- prepare ---------------------------------------------------------------------------------
# Read-only throughout, so it is safe to run against a stack that is still warming up. It does
# NOT pre-warm AIPerf's mmap dataset cache: that is populated from a completed run, so the first
# `run` over a (trace, window) pair synthesizes every prompt before its first request goes out.
prepare(){
  local workers n
  [ -f "$AIPERF_TRACE" ] || die "AIPERF_TRACE does not exist here: $AIPERF_TRACE"
  mkdir -p "$SHAPE_DIR" || die "could not create $SHAPE_DIR"

  log "=== prepare 1/4 image on $AIPERF_NODE ==="
  # Pulled now rather than at the head of the measured run. The image is public, so a 401 here
  # means a stale `docker login nvcr.io` on this host, not a wrong tag.
  on "$AIPERF_NODE" "docker image inspect $AIPERF_IMAGE >/dev/null 2>&1 || docker pull $AIPERF_IMAGE" \
    || die "could not obtain $AIPERF_IMAGE on $AIPERF_NODE"
  log "  $AIPERF_IMAGE present"

  log "=== prepare 2/4 paths visible on $AIPERF_NODE ==="
  mkdir -p "$AIPERF_OUT" || die "could not create $AIPERF_OUT"
  on "$AIPERF_NODE" "test -d '$AIPERF_OUT'" \
    || die "AIPERF_OUT is not visible on $AIPERF_NODE: $AIPERF_OUT
It must resolve to the same path on both hosts, like \$KIT_DIR already does."
  on "$AIPERF_NODE" "test -r '$AIPERF_TRACE'" \
    || die "the trace is not readable on $AIPERF_NODE: $AIPERF_TRACE"
  # AIPerf treats --tokenizer as a local path only when it exists, and as a HuggingFace repo id
  # otherwise — which HF_HUB_OFFLINE then turns into a hard failure at startup.
  on "$AIPERF_NODE" "test -r '$MODEL/tokenizer_config.json' || test -r '$MODEL/tokenizer.json'" \
    || die "no tokenizer files under $MODEL on $AIPERF_NODE"
  log "  trace, tokenizer and output dir all readable on $AIPERF_NODE"
  # Auto-promotion to fixed-schedule hinges on the first record carrying a timestamp. Without
  # one the replay runs but silently ignores the trace's pacing, which is the point of a replay.
  head -1 "$AIPERF_TRACE" | grep -q '"timestamp"' \
    || warn "  first record has no \"timestamp\" — AIPerf will NOT auto-promote to fixed-schedule"

  log "=== prepare 3/4 router reachable from $AIPERF_NODE ==="
  # From the HOST: the AIPerf container is --network=host, so the host's view is the container's.
  on "$AIPERF_NODE" "curl -sf -m10 '$URL/health' >/dev/null" \
    || die "no answer from the router at $URL from $AIPERF_NODE — is the stack up, and is that IP routable from there?"
  workers="$(on "$AIPERF_NODE" "curl -s -m10 '$URL/v1/workers'" | tr -d ' \r')"
  for role in prefill decode; do
    n="$(printf '%s' "$workers" | grep -c "\"disagg_mode\":\"$role\"")"
    [ "${n:-0}" -ge 1 ] || die "no $role worker registered with the router — replaying now would measure half a deployment"
    log "  $role workers registered: $n"
  done

  log "=== prepare 4/4 trace shape ==="
  # ISL/OSL distributions, prefix groups and the theoretical hit rate at this block size. A hit
  # rate near zero means this trace exercises no prefix reuse, which is cheaper to learn here
  # than from results. Reads the WHOLE file — it does not know about START_MS/END_MS.
  aiperf_in_image "aiperf analyze-trace --input-file $AIPERF_TRACE --block-size $BLOCK_SIZE --output-file $SHAPE" \
    || warn "  analyze-trace failed — continuing, the replay does not depend on it"
  log "  shape report -> $SHAPE"
}

# ---- run -------------------------------------------------------------------------------------
run_replay(){
  local ts run_dir args extra
  ts="$(date +%Y%m%d_%H%M%S)"
  run_dir="$AIPERF_OUT/$ts"

  [ -r "$AIPERF_TRACE" ] || die "trace not readable: $AIPERF_TRACE"
  mkdir -p "$run_dir" || die "could not create $run_dir"

  # Optional flags accumulate into $args; --extra-inputs takes a LIST, so $extra is separate and
  # goes LAST or the flags after it are swallowed as more of its values.
  #
  # auto-offset shifts the first timestamp to 0 so the replay starts immediately, and is mutually
  # exclusive with an explicit start offset. Both offsets are applied by the loader.
  args="--fixed-schedule"
  if [ -n "${START_MS:-}" ]; then
    args="$args --fixed-schedule-start-offset $START_MS"
  else
    args="$args --fixed-schedule-auto-offset"
  fi
  [ -n "${END_MS:-}" ] && args="$args --fixed-schedule-end-offset $END_MS"

  # Not only a time-axis knob: any value routes the trace through AIPerf's Synthesizer, which
  # rewrites hash_ids — shared prefix blocks keep their ids, each request is truncated at its
  # first non-shared block and the tail gets fresh unique ids. Near a no-op on a prefix-chained
  # trace, but prepare's hit rate then describes the raw file, not what replays.
  [ -n "${SPEEDUP:-}" ] && args="$args --synthesis-speedup-ratio $SPEEDUP"

  # A CEILING, not a target — fixed-schedule sends at the recorded timestamps regardless. Too
  # low degenerates into the closed-loop behaviour bench.sh already measures; too high lets a
  # deployment that cannot keep up grow unbounded queue depth, after which every percentile
  # describes the queue. 0 omits the flag if a future AIPerf rejects the pairing.
  [ "$MAX_CONC" != "0" ] && args="$args --concurrency $MAX_CONC"

  # temperature/top_p are the checkpoint's own generation_config defaults and deliberately NOT
  # greedy, for the reason bench.sh passes them — and it matters more here, where prompts are
  # longer. ignore_eos is the one thing bench.sh does not send: the mooncake_trace loader sets
  # max_tokens from output_length but injects no min_tokens (--force-min-tokens is honored only
  # by baseten_trace), so without it a thinking model that stops early replays short. That also
  # makes decode-side numbers here incomparable to bench.sh — see the README.
  extra="temperature:1.0 top_p:0.95"
  [ "${IGNORE_EOS:-1}" = "1" ] && extra="$extra ignore_eos:true"

  log "=== replay ==="
  log "  trace     : $AIPERF_TRACE"
  [ -n "${START_MS:-}${END_MS:-}" ] && log "  window    : ${START_MS:-start}..${END_MS:-end} ms"
  log "  endpoint  : $URL  (model $SERVED)"
  log "  client    : $AIPERF_NODE  ($AIPERF_IMAGE)"
  log "  artifacts : $run_dir"
  log "  options   : $args --extra-inputs $extra"
  # A cold cache spends minutes synthesizing prompts with nothing on the wire; say so, or it
  # reads as a hang.
  log "  a cold cache synthesizes and tokenizes every prompt BEFORE the first request goes out"

  aiperf_in_image "aiperf profile \
    --model $SERVED --tokenizer $MODEL \
    --endpoint-type chat --streaming \
    --url $URL \
    --input-file $AIPERF_TRACE --custom-dataset-type mooncake_trace \
    --isl-block-size $BLOCK_SIZE \
    --workers-max $WORKERS \
    --request-timeout-seconds $REQ_TIMEOUT \
    --no-gpu-telemetry --ui none \
    --artifact-dir $run_dir \
    $args --extra-inputs $extra" \
    || die "replay failed — read the output above, then $run_dir for whatever AIPerf wrote"

  log "replay done -> $run_dir"
  # Where a thinking model that stopped short of the requested output length shows up, as a
  # percentage rather than a failure.
  log "read the OSL-mismatch block in the summary before quoting any decode-side number"
}

case "${1:-all}" in
  prepare) prepare ;;
  run)     run_replay ;;
  all)     prepare && run_replay ;;
  *)       die "unknown subcommand '$1'. Use: prepare | run | (nothing, for both)" ;;
esac
exit 0
