. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/holder/snap.sh
# F-B1 with request weight 125000 (about one mean input length per held request)
bash docker/build.sh series-C1.txt infera-vllm:nightly-de7cf7f8-C1 > /dev/null 2>&1 || exit 1
STEPS=score DECODE_REQUEST_WEIGHT=125000 bash holder/item_kit.sh F-B1 F-B1-W125k-$(date -u +%m%d%H%M)
