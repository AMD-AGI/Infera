. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/holder/snap.sh
# C1 score again with the decode metrics sampler: repeatability and the engine-LB balance baseline for stage F
bash docker/build.sh series-C1.txt infera-vllm:nightly-de7cf7f8-C1 > /dev/null 2>&1 || exit 1
STEPS=score  bash holder/item_kit.sh C1 C1-score-rep-$(date -u +%m%d%H%M)
