set -e
K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages
sed -e 's#^\# dcp_pd llying stage S2 .*#\# gmu S2e: S2 + decode mem 0.88 + memlog overlay (Decode batch line prints device/torch memory)#' \
    -e 's#work/S2-\$NK#work/S2e-$NK#' -e 's#-s2$#-s2e#' \
    -e 's#src/sglang-ab7a898b77/python/sglang#src/sglang-ab7a898b77-memlog.yihou/python/sglang#' $K/S2.env > $K/S2e.env
echo 'export DECODE_EXTRA_ARGS="--mem-fraction-static 0.88"' >> $K/S2e.env
grep -nE 'gmu|WORK_DIR|CTR|OVERLAY|DECODE_EXTRA' $K/S2e.env
