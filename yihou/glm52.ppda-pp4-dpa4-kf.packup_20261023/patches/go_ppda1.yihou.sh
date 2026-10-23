#!/usr/bin/env bash
# go_ppda1 (task ppdc 2026-10-23, ON 135): llying 4+4 C80 frontier is PP4 prefill + DPA4 decode (37,310). ppda1 = ppdc4
# (PP4 prefill, C80) with the decode leg = dpax1 (TP4/DP4 DPA, mem 0.95, max-running 96, graph 64, no EC patch).
# PP prefill has one DP rank, so router PD_DP_RANK_AFFINITY must be 0 (config: DCPX_PP sets it). A1 only.
set -uo pipefail
W=/shared_nfs/yihou_glm52/g4tune_20261019; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; D=/shared_nfs/yihou_glm52/dcpx_20261022/ppdc
C=$H/config.yihou.dcpx.sh
grep -q 'PD_DP_RANK_AFFINITY=0   # DCPX_PP' $C || { cp -p $C $D/config.yihou.dcpx.sh.before-ppaff; python3 -I - $C <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = '    export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-pp-prefill.tsv"\n'
assert s.count(a) == 1
s = s.replace(a, a + '    PD_DP_RANK_AFFINITY=0   # DCPX_PP: one prefill DP rank -> affinity would pin every request to decode DP rank 0\n')
open(p, "w").write(s)
PY
}
bash -n $C && echo CONFIG_OK
E=$W/env.ppda1-135x136.yihou.sh
[[ -e $E ]] || { echo "# env.ppda1-135x136.yihou.sh (2026-10-23) = env.ppdc4 (PP4 prefill, C80) with decode = dpax1 DPA4 (DCPX_DCP=0, mem 0.95, max-running 96, graph 64, no decode EC patch, no DCP graph list); router affinity 0 (PP prefill DP1). llying 4+4 C80 PP4+DPA4 = 37,310."; grep -v "^# env.ppdc" $W/env.ppdc4-135x136.yihou.sh | sed -e "s/YIHOU_ROUND=ppdc4/YIHOU_ROUND=ppda1/" -e "s/DCPX_DCP=4 /DCPX_DCP=0 /" -e "s/ DCPX_DECODE_EC=1 / DCPX_DECODE_EC=0 /" -e "s/DECODE_MEM_FRACTION=0.88 /DECODE_MEM_FRACTION=0.95 /" -e "s/DECODE_GRAPH_MAX_BS=96 DCPX_GRAPH_BS=1,2,4,8,12,16,24,32,40,48,64,80,96 //"; } > $E
grep -o "DCPX_DCP=[0-9]\|DCPX_DECODE_EC=[0-9]\|DECODE_MEM_FRACTION=[0-9.]*\|DECODE_MAX_RUNNING=[0-9]*\|DECODE_GRAPH_MAX_BS=[0-9]*\| CONC=[0-9]*" $E | tr '\n' ' '; echo
( source $E; cd $H; bash $H/run.v3x.yihou.sh dryrun ) > $W/dryrun.ppda1-135x136.yihou.txt 2>&1 || { echo DRYRUN_FAIL; tail -4 $W/dryrun.ppda1-135x136.yihou.txt; }
echo "== tokdiff ppdc4 -> ppda1"; diff <(tr ' ' '\n' < $W/dryrun.ppdc4-135x136.yihou.txt | sed 's/ppdc4/ROUND/g') <(tr ' ' '\n' < $W/dryrun.ppda1-135x136.yihou.txt | sed 's/ppda1/ROUND/g') | grep '^[<>]' | grep -v '^[<>] +$' | tr '\n' ' '; echo
echo "== check ppdc4 dryrun affinity (it ran with affinity false already?)"; grep -o "INFERA_PD_DP_RANK_AFFINITY=[a-z]*" $W/dryrun.ppdc4-135x136.yihou.txt $W/dryrun.ppda1-135x136.yihou.txt | sort | uniq -c
cd /; setsid nohup bash $D/orch16.yihou.sh ppda1 80 80 > $D/orch16.ppda1.yihou.log 2>&1 < /dev/null &
sleep 3; cat $D/orch16.ppda1.yihou.log
