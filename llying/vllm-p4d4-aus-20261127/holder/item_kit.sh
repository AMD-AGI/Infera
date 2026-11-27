#!/usr/bin/env bash
# Holder item: one kit point.  item_kit.sh VARIANT TAG   (env passed through: CONC, STEPS, FAST, ...)
set -uo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
variant=${1:?variant} tag=${2:?tag}
out=$W/results/$tag
mkdir -p "$out"
echo "$(ts) $NODE: $variant -> $out (CONC=${CONC:-152} STEPS=${STEPS:-eval score} FAST=${FAST:-0})" | tee -a "$out/run.log"
bash "$(cd "$(dirname "$0")/.." && pwd)/kit/run.sh" "$variant" "$out" 2>&1 | tee -a "$out/run.log"
