set -e
K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages
sed -e 's#^\# gmu S2e.*#\# gmu S2f: S2e (memlog overlay) at decode mem 0.92 + PYTORCH_HIP_ALLOC_CONF=expandable_segments:True on decode; burst v3: SDMA=0 both legs#' \
    -e 's#work/S2e-\$NK#work/S2f-$NK#' -e 's#-s2e$#-s2f#' \
    -e 's#SGLANG_LOG_DECODE_SEQ_LENS=1"#SGLANG_LOG_DECODE_SEQ_LENS=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True HSA_ENABLE_SDMA=0"#' \
    -e 's#^export DECODE_EXTRA_ARGS=.*#export DECODE_EXTRA_ARGS="--mem-fraction-static 0.92"#' $K/S2e.env > $K/S2f.env
echo 'export PREFILL_ENV="HSA_ENABLE_SDMA=0"' >> $K/S2f.env
grep -nE 'gmu|WORK_DIR|CTR|OVERLAY|DECODE_ENV|DECODE_EXTRA|PREFILL_ENV' $K/S2f.env
