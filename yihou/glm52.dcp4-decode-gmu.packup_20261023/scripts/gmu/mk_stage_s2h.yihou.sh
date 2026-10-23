set -e
K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages
sed -e 's#^\# gmu S2e.*#\# gmu S2h: S2e (0.88, memlog) + PYTORCH_HIP_ALLOC_CONF=expandable_segments:True on decode (single variable vs S2e)#' \
    -e 's#work/S2e-\$NK#work/S2h-$NK#' -e 's#-s2e$#-s2h#' \
    -e 's#SGLANG_LOG_DECODE_SEQ_LENS=1"#SGLANG_LOG_DECODE_SEQ_LENS=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True"#' $K/S2e.env > $K/S2h.env
grep -nE 'gmu|WORK_DIR|CTR|DECODE_ENV|DECODE_EXTRA' $K/S2h.env
