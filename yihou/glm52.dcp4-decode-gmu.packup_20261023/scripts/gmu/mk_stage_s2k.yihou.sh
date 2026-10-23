set -e
K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages
sed -e 's#^\# gmu S2e.*#\# gmu S2k: S2e (0.88) with memlog-ec overlay: empty_cache() at the Decode-batch log point when device free < 6 GiB#' \
    -e 's#work/S2e-\$NK#work/S2k-$NK#' -e 's#-s2e$#-s2k#' \
    -e 's#sglang-ab7a898b77-memlog.yihou#sglang-ab7a898b77-memlog-ec.yihou#' $K/S2e.env > $K/S2k.env
grep -nE 'gmu|CTR|OVERLAY|DECODE_EXTRA' $K/S2k.env
