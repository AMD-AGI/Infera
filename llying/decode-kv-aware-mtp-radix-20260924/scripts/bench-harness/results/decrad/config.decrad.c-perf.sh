#!/usr/bin/env bash
# Purpose: Performance arm C: B-perf plus decode HiCache at full capacity, with
#   the same HiCache knobs as the prefill leg (no test-only device cap).
# Usage: CONFIG=<this file> TOPOLOGY=<this dir>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.decrad.b-perf.sh"

# engine.sh rejects DECODE_HICACHE=1 with DECODE_MTP=1 (see config.decrad.c.sh).
# AgentX's preflight (tools/agentx_env.py) compares the live flag with
# DECODE_HICACHE, so run_perf.sh passes this override to the AgentX step only.
AGENTX_OVERRIDES="DECODE_HICACHE=1"
DECODE_EXTRA_ARGS+=" --enable-hierarchical-cache --hicache-ratio $DECODE_HICACHE_RATIO"
DECODE_EXTRA_ARGS+=" --hicache-write-policy $DECODE_HICACHE_WRITE_POLICY"
DECODE_EXTRA_ARGS+=" --hicache-io-backend $DECODE_HICACHE_IO_BACKEND"
DECODE_EXTRA_ARGS+=" --hicache-mem-layout $DECODE_HICACHE_MEM_LAYOUT"
