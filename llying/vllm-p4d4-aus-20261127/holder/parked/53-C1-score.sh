. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/holder/snap.sh
# C1 score only on this node: pairs C1 with D and E (same node), and with C1 on n04-25 gives the node-to-node gap
bash docker/build.sh series-C1.txt infera-vllm:nightly-de7cf7f8-C1 > /dev/null 2>&1 || exit 1
STEPS=score bash holder/item_kit.sh C1 C1-score-$(date -u +%m%d%H%M)
