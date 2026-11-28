. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/holder/snap.sh
bash docker/build.sh series-C1.txt infera-vllm:nightly-de7cf7f8-C1 > /dev/null 2>&1 || exit 1
# Node A: C1 (engine LB) vs F-B1 (router LB + output growth, W=0) vs E (C1 + THP off), same node, with samplers
bash holder/item_seq.sh "C1|seqA-C1|" "F-B1|seqA-FB1-W0|DECODE_REQUEST_WEIGHT=0" "E|seqA-E|"
