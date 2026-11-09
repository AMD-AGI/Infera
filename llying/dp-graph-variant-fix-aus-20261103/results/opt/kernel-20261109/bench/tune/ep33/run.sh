#!/usr/bin/env bash
# For each candidate row (33-expert EP shape, token key 1024 = M 513..1024): ep_moe_bench at M 704, rank 0, own process.
D=/perf_apps/liyingli/bench_agentx/kopt-20261109/tune/ep33
C=/sgl-workspace/aiter/aiter/configs
L=$C/tuned_fmoe.csv; for f in $C/model_configs/*tuned_fmoe*.csv; do case $f in *untuned*) ;; *) L=$L:$f;; esac; done
cd /perf_apps/liyingli/bench_agentx/kopt-20261109/bench
for c in $D/cand_from*.csv; do
  k=$(tail -1 $c | cut -d, -f18,21 | tr , ' ')
  r=$(AITER_CONFIG_FMOE=$L:$c timeout 600 python3 ep_moe_bench.py --ms 704 --ranks 0 --nsets 3 2>&1 | grep -E "^M " | sed 's/.*| ep/ep/')
  echo "$(basename $c): $k -> $r"
done
