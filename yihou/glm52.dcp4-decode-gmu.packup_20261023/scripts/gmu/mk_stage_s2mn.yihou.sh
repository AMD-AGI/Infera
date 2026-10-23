set -e
K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages
for x in "m 0.92" "n 0.88"; do set -- $x
sed -e "s#^\# gmu S2k.*#\# gmu S2$1: memlog-ec2 overlay (empty_cache on every rank when free < 6 GiB) at decode mem $2#" \
    -e "s#work/S2k-\\\$NK#work/S2$1-\$NK#" -e "s#-s2k\$#-s2$1#" -e 's#memlog-ec.yihou#memlog-ec2.yihou#' \
    -e "s#^export DECODE_EXTRA_ARGS=.*#export DECODE_EXTRA_ARGS=\"--mem-fraction-static $2\"#" $K/S2k.env > $K/S2$1.env
grep -nE 'gmu|CTR|OVERLAY|DECODE_EXTRA|WORK_DIR' $K/S2$1.env; done
