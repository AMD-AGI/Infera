#!/usr/bin/env bash
# Kernel-level runs on free GPUs, one job per GPU in parallel, into $R/results/<tag>/:
#   topk-randn / topk-ties / topk-relu / topk-flush   kf/bench_topk.py (aiter vs vLLM _C vs #47872 port)
#   overlay-v3-test                                    kf/test_overlay_logits.py with overlay-v3 mounted
#   a3_kernels.sh TAG
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackA-20261118}
out=$R/results/${1:-a3}; mkdir -p "$out"
K=$R/kf/run_kf.sh
bash $K 0 "$out/topk-randn.txt" bench_topk.py &
EXTRA_ENV="DIST=ties" bash $K 1 "$out/topk-ties.txt" bench_topk.py &
EXTRA_ENV="DIST=relu" bash $K 2 "$out/topk-relu.txt" bench_topk.py &
EXTRA_ENV="FLUSH=1" bash $K 3 "$out/topk-flush.txt" bench_topk.py &
OVL=$R/overlay-v3 bash $K 4 "$out/overlay-v3-test.txt" test_overlay_logits.py &
wait
for f in "$out"/*.txt; do echo "== $(basename "$f")"; grep -v -E '^\[aiter\]|^\s*$|Warning|warn' "$f" | tail -12; done
exit 0
