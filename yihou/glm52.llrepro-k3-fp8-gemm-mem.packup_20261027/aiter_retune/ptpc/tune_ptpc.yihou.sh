#!/usr/bin/env bash
# tune_ptpc (task llrepro 2026-10-26, ON pool node 190 as root): aiter a16w16 tune of decode GEMM shapes still untuned with draft FP8 (llr32: N=1024 K=6144, N=6144 K=512).
#
# Usage: tune_k3.yihou.sh <HIP_VISIBLE_DEVICES, e.g. 4,5,6,7>
W=/shared_nfs/yihou_glm52/llrepro_20261023/prep_next.yihou/aiter_retune/ptpc
n=$(echo "$1" | tr ',' '\n' | wc -l)
docker run --rm --name yihou-aitune-ptpc --device /dev/kfd --device /dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=$1 -v $W:/w -w /sgl-workspace/aiter --entrypoint bash infera-sglang:v0522-fdbafb60 -c \
  "python3 csrc/gemm_a16w16/gemm_a16w16_tune.py --input_file /w/decode_bf16_untuned_gemm.csv \
   --tuned_file /w/yihou_glm52_ptpc_bf16_tuned_gemm.csv -o2 /w/prof_gemm_ptpc.csv --mp $n --libtype all --with-hipblaslt" \
  > $W/tune_ptpc.log 2>&1
echo "TUNE_PTPC_DONE rc=$?" >> $W/tune_ptpc.log
