set -e
K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages
# S2i = S2e + torch allocator GC threshold (single variable vs S2e)
sed -e 's#^\# gmu S2e.*#\# gmu S2i: S2e (0.88, memlog) + PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9 on decode#' \
    -e 's#work/S2e-\$NK#work/S2i-$NK#' -e 's#-s2e$#-s2i#' \
    -e 's#SGLANG_LOG_DECODE_SEQ_LENS=1"#SGLANG_LOG_DECODE_SEQ_LENS=1 PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9"#' $K/S2e.env > $K/S2i.env
# S2j = S2e at 0.87 (single variable vs S2e)
sed -e 's#^\# gmu S2e.*#\# gmu S2j: S2e (memlog) at decode mem 0.87#' -e 's#work/S2e-\$NK#work/S2j-$NK#' -e 's#-s2e$#-s2j#' \
    -e 's#^export DECODE_EXTRA_ARGS=.*#export DECODE_EXTRA_ARGS="--mem-fraction-static 0.87"#' $K/S2e.env > $K/S2j.env
grep -nE 'gmu|CTR|PYTORCH|DECODE_EXTRA' $K/S2i.env $K/S2j.env
