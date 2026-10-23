set -e
K=/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages
sed -e 's#^\# dcp_pd llying stage S2 .*#\# gmu S2r: S2 unchanged (decode mem 0.85), reference rerun with VRAM sampling; new CTR name (stopped s2 containers keep theirs)#' \
    -e 's#work/S2-\$NK#work/S2r-$NK#' -e 's#-s2$#-s2r#' $K/S2.env > $K/S2r.env
grep -nE 'gmu|WORK_DIR|CTR' $K/S2r.env
