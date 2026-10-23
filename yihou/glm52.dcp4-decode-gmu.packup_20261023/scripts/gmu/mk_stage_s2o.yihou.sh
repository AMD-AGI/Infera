set -e
K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages
sed -e 's#^\# gmu S2m.*#\# gmu S2o: S2m (0.92, memlog-ec2 all-rank empty_cache) on a burst v3 node: HSA_ENABLE_SDMA=0 both legs#' \
    -e 's#work/S2m-\$NK#work/S2o-$NK#' -e 's#-s2m$#-s2o#' \
    -e 's#SGLANG_LOG_DECODE_SEQ_LENS=1"#SGLANG_LOG_DECODE_SEQ_LENS=1 HSA_ENABLE_SDMA=0"#' $K/S2m.env > $K/S2o.env
echo 'export PREFILL_ENV="HSA_ENABLE_SDMA=0"' >> $K/S2o.env
grep -nE 'gmu|CTR|OVERLAY|DECODE_ENV|DECODE_EXTRA|PREFILL_ENV' $K/S2o.env
