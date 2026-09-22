#!/usr/bin/env bash
# AUS 1P1D P8D8: prefill HiCache, decode MTP, C80.
HICACHE_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HICACHE_CONFIG_DIR/config.phase-a.sh"

PREFILL_HICACHE=1
PREFILL_HICACHE_RATIO=1.5
PREFILL_HICACHE_WRITE_POLICY=write_through
PREFILL_HICACHE_IO_BACKEND=kernel
PREFILL_HICACHE_MEM_LAYOUT=page_first
DECODE_HICACHE=0

# The current image has no router affinity patch. Match phase A's actual
# independent P/D selection and avoid claiming that affinity is enforced.
PD_DP_RANK_AFFINITY=0
