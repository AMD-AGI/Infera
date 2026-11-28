. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/holder/snap.sh
# Stage F B1: B0 + streamed decode output charged to the rank's demand; paired with C1 on this node
bash docker/build.sh series-C1.txt infera-vllm:nightly-de7cf7f8-C1 > /dev/null 2>&1 || exit 1
STEPS=score DECODE_REQUEST_WEIGHT=0 bash holder/item_kit.sh F-B1 F-B1-full-$(date -u +%m%d%H%M)
