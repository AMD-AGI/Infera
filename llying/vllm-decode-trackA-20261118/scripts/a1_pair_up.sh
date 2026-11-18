#!/usr/bin/env bash
# A1 step 1: aggK (GPUs 0-3, #47897 overlay on, LPT group 4) and aggR (GPUs 4-7, stock) side by side, then check that
# the gluon kernel is really taken on aggK. Output: $R/results/a1/<tag>/{aggK,aggR}-up.log, check.txt.
#   a1_pair_up.sh TAG
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackA-20261118}
tag=${1:-pair1}
out=$R/results/a1/$tag; mkdir -p "$out"
cd "$R"
TRACE_DIR=$out/trace-aggK bash scripts/agg_up.sh vta-aggK 0,1,2,3 8100 29610 gluon > "$out/aggK-up.log" 2>&1 &
sleep 20
TRACE_DIR=$out/trace-aggR bash scripts/agg_up.sh vta-aggR 4,5,6,7 8200 29710 stock > "$out/aggR-up.log" 2>&1 &
wait
{
  echo "== $(date -u +%T) up logs"; tail -2 "$out/aggK-up.log" "$out/aggR-up.log"
  for c in aggK aggR; do
    f=logs/engines/vta-$c.log
    echo "== $c"
    echo "gluon lines: $(grep -a -c 'gfx950 gluon paged-MQA logits' $f)"
    echo "aiter paged-MQA lines: $(grep -a -c 'Using AITER paged-MQA logits' $f)"
    grep -a -m 4 -E 'DSA indexer decode path|GPU KV cache size|Maximum concurrency|cudagraph_capture_sizes|Capturing CUDA graphs' $f | cut -c1-300
  done
} > "$out/check.txt" 2>&1
cat "$out/check.txt"
exit 0
