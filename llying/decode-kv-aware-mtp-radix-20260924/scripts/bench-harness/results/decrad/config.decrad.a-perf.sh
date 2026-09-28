#!/usr/bin/env bash
# Purpose: Performance arm A: the same-image baseline (decode radix cache off,
#   decode ChunkCache) with the t2f simulated MTP acceptance (3.61).
# Usage: CONFIG=<this file> TOPOLOGY=<this dir>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.decrad.a.sh"
DECODE_SIMULATE_ACC_LEN=3.61
