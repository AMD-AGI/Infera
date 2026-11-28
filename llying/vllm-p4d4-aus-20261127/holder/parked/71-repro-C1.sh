. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/holder/snap.sh
bash docker/build.sh series-C1.txt infera-vllm:nightly-de7cf7f8-C1 > /dev/null 2>&1 || exit 1
# Kit reproduction from scratch per kit/README (C1, eval + score): also C1 repeatability
bash holder/item_seq.sh "C1|repro-C1-full|STEPS=eval,score"
