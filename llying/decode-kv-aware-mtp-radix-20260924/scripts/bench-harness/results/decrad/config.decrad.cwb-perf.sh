#!/usr/bin/env bash
# Purpose: Performance arm C-wb: C-perf with the decode leg on
#   --hicache-write-policy write_back, so decode writes to host only when it
#   evicts instead of on every insert. The prefill leg keeps write_through.
# Usage: CONFIG=<this file> TOPOLOGY=<this dir>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
DECODE_HICACHE_WRITE_POLICY=write_back
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.decrad.c-perf.sh"
