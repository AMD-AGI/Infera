#!/usr/bin/env bash
# Purpose: Performance arm B: arm B (decode radix cache under MTP) with the t2f
#   simulated MTP acceptance (3.61), so timings compare with t2f and arm A-perf.
# Usage: CONFIG=<this file> TOPOLOGY=<this dir>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.decrad.b.sh"
DECODE_SIMULATE_ACC_LEN=3.61
