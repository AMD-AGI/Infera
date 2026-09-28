#!/usr/bin/env bash
# Purpose: Arm B (decode radix cache ON) of the decode-radix correctness gate.
#   Arm A plus decode KV events, which Infera turns into
#   --disaggregation-decode-enable-radix-cache, and the SGLang opt-in that
#   accepts it under MTP. engine.sh appends DECODE_EXTRA_ARGS after its own
#   --no-enable-kv-events --kv-events off, so these later flags take effect.
# Usage: CONFIG=<this file> TOPOLOGY=<this dir>/topology.yihou.tsv ./launch.sh
# Artifacts: none; this file only defines shell variables.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.decrad.a.sh"

# Decode is topology row 1.
DECODE_EXTRA_ARGS="--disable-custom-all-reduce --enable-kv-events --kv-events on"
DECODE_EXTRA_ARGS+=" --kv-events-bind tcp://0.0.0.0:$((KV_EVENT_PORT_BASE + 1))"
DECODE_EXTRA_ARGS+=" --kv-snapshot-port $((SNAPSHOT_PORT_BASE + 1))"
DECODE_EXTRA_ENV="SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1"
