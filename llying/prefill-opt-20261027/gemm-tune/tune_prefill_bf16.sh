#!/bin/bash
# Tune the prefill BF16 GEMM shapes that have no gfx950 entry at M = 8192 / 4096 in the image's merged aiter configs
# (o_proj 6144x16384, dense gate_up 24576x6144, q_a+kv_a 2624x6144, indexer wq_b 4096x2048, indexer weights_proj 32x6144).
# Same tuner invocation as yihou's decode retune (aiter_retune/ptpc/tune_ptpc.yihou.sh). Run on the node that owns the GPUs:
#   tune_prefill_bf16.sh GPUS [UNTUNED_CSV] [OUT_CSV]      e.g. tune_prefill_bf16.sh 0,1,2,3
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd)
gpus=$1
untuned=${2:-prefill_bf16_untuned_gemm.csv}
out=${3:-glm52_prefill_bf16_tuned_gemm.csv}
n=$(awk -F, '{print NF}' <<<"$gpus")
docker run --rm --name lyl-gemmtune --device /dev/kfd --device /dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES="$gpus" -v "$W":/w -w /sgl-workspace/aiter --entrypoint bash infera-sglang:v0522-fdbafb60 \
  -c "python3 csrc/gemm_a16w16/gemm_a16w16_tune.py --input_file /w/$untuned --tuned_file /w/$out -o2 /w/prof_$out \
      --mp $n --libtype all --with-hipblaslt"
