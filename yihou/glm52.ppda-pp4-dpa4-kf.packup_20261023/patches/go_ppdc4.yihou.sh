#!/usr/bin/env bash
# go_ppdc4 (task ppdc 2026-10-23, ON 135): continue the concurrency sweep: ppdc4 = ppdc2 (= ppdc1 config) at C80, A1 only.
set -uo pipefail
W=/shared_nfs/yihou_glm52/g4tune_20261019; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; D=/shared_nfs/yihou_glm52/dcpx_20261022/ppdc
E=$W/env.ppdc4-135x136.yihou.sh
[[ -e $E ]] || { echo "# env.ppdc4-135x136.yihou.sh (2026-10-23) = env.ppdc2 with CONC 64 -> 80 (sweep: C56 27426, C64 29817, C72 31128); A1 only."; grep -v "^# env.ppdc2" $W/env.ppdc2-135x136.yihou.sh | sed -e "s/YIHOU_ROUND=ppdc2/YIHOU_ROUND=ppdc4/" -e "s/ CONC=64 / CONC=80 /"; } > $E
( source $E; cd $H; bash $H/run.v3x.yihou.sh dryrun ) > $W/dryrun.ppdc4-135x136.yihou.txt 2>&1 || echo DRYRUN_FAIL
echo "== tokdiff ppdc2 -> ppdc4"; diff <(tr ' ' '\n' < $W/dryrun.ppdc2-135x136.yihou.txt | sed 's/ppdc2/ROUND/g') <(tr ' ' '\n' < $W/dryrun.ppdc4-135x136.yihou.txt | sed 's/ppdc4/ROUND/g') | grep '^[<>]' | tr '\n' ' '; echo
sed -e "s/orch15/orch16/g" -e "s/bench.\*ok.\*A2/bench.*ok.*A1/" -e "s/wait for A2 ok/wait for A1 ok/" -e "s/did not finish A2 ok/did not finish A1 ok/" $D/orch15.yihou.sh > $D/orch16.yihou.sh; bash -n $D/orch16.yihou.sh && grep -c "A1" $D/orch16.yihou.sh
cd /; setsid nohup bash $D/orch16.yihou.sh ppdc4 80 80 > $D/orch16.ppdc4.yihou.log 2>&1 < /dev/null &
sleep 3; cat $D/orch16.ppdc4.yihou.log
