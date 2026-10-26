#!/usr/bin/env bash
# tune_k3 (task llrepro 2026-10-26, ON pool node 190 as root): aiter a16w16 tune of the 10 decode GEMM shapes that K3
# (EAGLE 3/1/4) hits but the C96/K5 tune did not cover (N=6144 K=16384, M=80..256). Output next to this script.
# Usage: tune_k3.yihou.sh <HIP_VISIBLE_DEVICES, e.g. 4,5,6,7>
W=/shared_nfs/yihou_glm52/llrepro_20261023/prep_next.yihou/aiter_retune/k3
n=$(echo "$1" | tr ',' '\n' | wc -l)
docker run --rm --name yihou-aitune-k3 --device /dev/kfd --device /dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=$1 -v $W:/w -w /sgl-workspace/aiter --entrypoint bash infera-sglang:v0522-fdbafb60 -c \
  "python3 csrc/gemm_a16w16/gemm_a16w16_tune.py --input_file /w/decode_bf16_untuned_gemm.new.csv \
   --tuned_file /w/yihou_glm52_k3_bf16_tuned_gemm.csv -o2 /w/prof_gemm_k3.csv --mp $n --libtype all --with-hipblaslt" \
  > $W/tune_k3.log 2>&1
echo "TUNE_K3_DONE rc=$?" >> $W/tune_k3.log
