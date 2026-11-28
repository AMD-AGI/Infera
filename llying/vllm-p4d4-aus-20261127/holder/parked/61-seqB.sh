. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/holder/snap.sh
bash docker/build.sh series-C1.txt infera-vllm:nightly-de7cf7f8-C1 > /dev/null 2>&1 || exit 1
# Node B: C1 vs F-B1 W=125000 vs F-B0 W=125000, same node, with samplers (C1 repeat + node gap with node A)
bash holder/item_seq.sh "C1|seqB-C1|" "F-B1|seqB-FB1-W125k|DECODE_REQUEST_WEIGHT=125000" "F-B0|seqB-FB0-W125k|DECODE_REQUEST_WEIGHT=125000"
