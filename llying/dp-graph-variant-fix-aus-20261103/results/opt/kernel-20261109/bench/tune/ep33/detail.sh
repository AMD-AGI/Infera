#!/usr/bin/env bash
# Full kernel breakdown (top 10) of the sweep CSVs given as arguments (e.g. bm32_0 bm64_0), M 704.
D=/perf_apps/liyingli/bench_agentx/kopt-20261109/tune/ep33/sweep
C=/sgl-workspace/aiter/aiter/configs
L=$C/tuned_fmoe.csv; for f in $C/model_configs/*tuned_fmoe*.csv; do case $f in *untuned*) ;; *) L=$L:$f;; esac; done
cd /perf_apps/liyingli/bench_agentx/kopt-20261109/bench
for c in "$@"; do
  echo "== $c: $(tail -1 $D/$c.csv | cut -d, -f15,18,21)"
  AITER_CONFIG_FMOE=$L:$D/$c.csv TOPN=10 timeout 300 python3 ep_sweep_one.py 704 2>&1 | grep -A10 "^RESULT" | sed 's/ | .*//'
done
