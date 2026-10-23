set -e
K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages
sed -e 's#^\# gmu S2k.*#\# gmu S2l: S2k (memlog-ec, empty_cache when free < 6 GiB) at decode mem 0.92#' \
    -e 's#work/S2k-\$NK#work/S2l-$NK#' -e 's#-s2k$#-s2l#' \
    -e 's#^export DECODE_EXTRA_ARGS=.*#export DECODE_EXTRA_ARGS="--mem-fraction-static 0.92"#' $K/S2k.env > $K/S2l.env
grep -nE 'gmu|CTR|OVERLAY|DECODE_EXTRA' $K/S2l.env
