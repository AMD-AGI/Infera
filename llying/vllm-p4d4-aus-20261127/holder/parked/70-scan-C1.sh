. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/holder/snap.sh
bash docker/build.sh series-C1.txt infera-vllm:nightly-de7cf7f8-C1 > /dev/null 2>&1 || exit 1
# Concurrency points around C152 for C1 (score only): where the best point is on the nightly
bash holder/item_seq.sh "C1|scan-C1-c136|CONC=136" "C1|scan-C1-c168|CONC=168"
