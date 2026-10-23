#!/usr/bin/env bash
# sumrounds (task gap 2026-10-23, ON 135): summary rows for the rounds given as args (uses packup/summary.dcpx.yihou.sh).
D=/shared_nfs/yihou_glm52/dcpx_20261022; S=$D/tmp_sum.yihou.$$
sed "s/^ROUNDS=.*/ROUNDS=\"$*\"/" $D/packup/summary.dcpx.yihou.sh > $S.sh; mkdir -p $S/results; bash $S.sh $S; cat $S/results/summary.tsv
