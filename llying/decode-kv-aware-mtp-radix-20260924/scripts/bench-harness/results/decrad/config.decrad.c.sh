#!/usr/bin/env bash
# Purpose: Arm C: arm B (decode radix cache under MTP) plus decode HiCache, so
#   both legs run HiCache. The decode HiCache knobs match the prefill leg's.
#   Two test-only knobs make decode host write-back and load-back happen within
#   minutes: a capped decode device pool and an explicit, larger host pool.
# Usage: CONFIG=<this file> TOPOLOGY=<this dir>/topology.yihou.tsv ./launch.sh
# Artifacts: none; this file only defines shell variables.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.decrad.b.sh"

: "${DECODE_MAX_TOTAL_TOKENS:=200000}"  # per DP rank; the profiled pool is ~2.2M
: "${DECODE_HICACHE_SIZE_GB:=60}"       # per DP rank; overrides the ratio (~1.3M tokens)

# engine.sh rejects DECODE_HICACHE=1 together with DECODE_MTP=1; that guard
# predates the decode radix opt-in. Pass the flags through the escape hatch and
# leave DECODE_HICACHE=0, so engine.sh prints HiCache=0 for this leg.
DECODE_EXTRA_ARGS+=" --enable-hierarchical-cache --hicache-ratio $DECODE_HICACHE_RATIO"
DECODE_EXTRA_ARGS+=" --hicache-write-policy $DECODE_HICACHE_WRITE_POLICY"
DECODE_EXTRA_ARGS+=" --hicache-io-backend $DECODE_HICACHE_IO_BACKEND"
DECODE_EXTRA_ARGS+=" --hicache-mem-layout $DECODE_HICACHE_MEM_LAYOUT"
DECODE_EXTRA_ARGS+=" --hicache-size $DECODE_HICACHE_SIZE_GB"
DECODE_EXTRA_ARGS+=" --max-total-tokens $DECODE_MAX_TOTAL_TOKENS"
