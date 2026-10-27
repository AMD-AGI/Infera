#!/usr/bin/env bash
# Side jobs on 140's idle GPUs 0-3 while the night queue's prefill trace (step 2) holds 138 GPUs 4-7 and 140 GPUs 4-7,
# so they never overlap a perf measurement: the HiCache perf step (3) waits for all of 140's VRAM, i.e. for these jobs.
#   GPUs 0-2: aiter BF16 GEMM tuning of the untuned prefill shapes (gemm-tune/tune_prefill_bf16.sh)
#   GPU 3:    sparse-MLA prefill microbench (scripts/bench_sparse_mla.py), overlap and uniform index modes
# Usage (login node): gpu_side_jobs_1027.sh QUEUE_OUT     QUEUE_OUT = the night queue's .out on 138
set -uo pipefail
Q="$1"
D=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027
H138=crsuse2-m2m-138 H140=crsuse2-m2m-140
until timeout 60 ssh -o BatchMode=yes $H138 "grep -q 'step 2: prefill trace' $Q"; do sleep 120; done
echo "$(date -u +%T) step 2 started; waiting for the trace deployment on 140 GPUs 4-7"
until [[ "$(timeout 60 ssh -o BatchMode=yes $H140 "rocm-smi --showmemuse | awk -F: '/GPU\\[4\\].*VRAM%/{print int(\$NF)}'")" -ge 20 ]]; do
    sleep 60
done
busy=$(timeout 60 ssh -o BatchMode=yes $H140 "rocm-smi --showmemuse | awk -F: '/GPU\\[[0-3]\\].*VRAM%/ && int(\$NF)>0'")
[[ -z "$busy" ]] || { echo "$(date -u +%T) 140 GPUs 0-3 not idle, abort: $busy"; exit 1; }
echo "$(date -u +%T) trace deployment up; starting side jobs"
timeout 60 ssh -o BatchMode=yes $H140 "cd $D/gemm-tune && nohup bash tune_prefill_bf16.sh 0,1,2 > tune-$(date -u +%m%dT%H%M).log 2>&1 &"
timeout 60 ssh -o BatchMode=yes $H140 "nohup bash -c 'for m in overlap uniform; do docker run --rm --name lyl-smla-bench-\$m \
  --device /dev/kfd --device /dev/dri --group-add video -e HIP_VISIBLE_DEVICES=3 -v $D/scripts:/w \
  --entrypoint python3 infera-sglang:v0522-fdbafb60 /w/bench_sparse_mla.py --mode \$m; done' \
  > $D/results/bench_sparse_mla-$(date -u +%m%dT%H%M).log 2>&1 &"
echo "$(date -u +%T) launched; logs in $D/gemm-tune/tune-*.log and $D/results/bench_sparse_mla-*.log"
