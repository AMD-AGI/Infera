#!/usr/bin/env bash
# v2 kit 1P2D at C96 and C128 (only after a clean 30-v2-1p2d-gate): against 4+4 at the same interactivity (4+4 C64 41,557 @ p90 76.2,
# C104 53,209 @ 61.0) -- one PP4 prefill near 3.5 req/s feeding two decodes at half the per-rank load.
o=/perf_apps/liyingli/vfix-logs/v2/v2.out
g="$(grep -a 'v2-1p2d-gate-[0-9T]* smoke:' "$o" | tail -1)" pr="$(grep -a 'v2-1p2d-gate-[0-9T]* probe:' "$o" | tail -1)"
if [[ "$g" != *6/6* || "$pr" != *" garbled=0 "* ]]; then echo "$(date -u +%T) v2 v2-1p2d-sweep skipped: gate not clean ($g; $pr)" >> "$o"; exit 0; fi
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/v2-aus/v2_item.sh v2-1p2d sweep 1p2d 96 128
