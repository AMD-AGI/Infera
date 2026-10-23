#!/usr/bin/env bash
# go_ppdc2 (task ppdc 2026-10-23, ON 135): ppdc2 = ppdc1 config, concurrency sweep on one stack: A1 @C64, A2 @C72
# (different concurrency, not a replicate); stop after A2. Prefill queue p90 at C56 was 0.14 s -> capacity left.
set -uo pipefail
W=/shared_nfs/yihou_glm52/g4tune_20261019; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
D=/shared_nfs/yihou_glm52/dcpx_20261022/ppdc; O=/shared_nfs/yihou_glm52/dcpx_20261022/dcpx11/orch14.yihou.sh
E=$W/env.ppdc2-135x136.yihou.sh
[[ -e $E ]] || { echo "# env.ppdc2-135x136.yihou.sh (2026-10-23) = env.ppdc1 with CONC 56 -> 64 (chain A1 @C64, A2 @C72 on the same stack)."; grep -v "^# env.ppdc1" $W/env.ppdc1-135x136.yihou.sh | sed -e "s/YIHOU_ROUND=ppdc1/YIHOU_ROUND=ppdc2/" -e "s/ CONC=56 / CONC=64 /"; } > $E
grep -o " CONC=[0-9]*" $E
( source $E; cd $H; bash $H/run.v3x.yihou.sh dryrun ) > $W/dryrun.ppdc2-135x136.yihou.txt 2>&1 || echo DRYRUN_FAIL
echo "== tokdiff ppdc1 -> ppdc2"; diff <(tr ' ' '\n' < $W/dryrun.ppdc1-135x136.yihou.txt | sed 's/ppdc1/ROUND/g') <(tr ' ' '\n' < $W/dryrun.ppdc2-135x136.yihou.txt | sed 's/ppdc2/ROUND/g') | grep '^[<>]' | tr '\n' ' '; echo
# orch: same helpers as orch14, but wait for A2 before stopping
sed -n '1,/^echo "\$(ts) orch14 start: \$\*"/p' $O | sed '$d' | sed 's/^# orch14 (task dcpx11.*/# orch15 (task ppdc 2026-10-23, ON 135): launch <round> with chain C1 C2, wait for A2 ok, stop./' > $D/orch15.yihou.sh
cat >> $D/orch15.yihou.sh <<'EOS'
r=$1; c1=$2; c2=$3
echo "$(ts) orch15 start: $r C$c1 C$c2"
df -h / | tail -1; ssh -n -o BatchMode=yes crsuse2-m2m-136 "df -h / | tail -1"
cd /; setsid nohup bash $H/chain_x.yihou.sh $W/env.$r-135x136.yihou.sh $c1 $c2 > $R/$r.chain.log 2>&1 < /dev/null &
echo "$(ts) launched $r"
if wait_status $r "bench.*ok.*A2" 14400; then grep -a "bench.*ok" $R/$r/status.tsv | cut -c1-200
else echo "$(ts) $r did not finish A2 ok:"; tail -3 $R/$r/status.tsv | cut -c1-200; fi
stop_round $r
echo "$(ts) orch15 done"
EOS
bash -n $D/orch15.yihou.sh && echo ORCH15_OK
cd /; setsid nohup bash $D/orch15.yihou.sh ppdc2 64 72 > $D/orch15.ppdc2.yihou.log 2>&1 < /dev/null &
sleep 5; cat $D/orch15.ppdc2.yihou.log
