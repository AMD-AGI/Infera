#!/usr/bin/env bash
# go_ppdc3 (task ppdc 2026-10-23, ON 135): user: drop HiCache write_back. ppdc3 = ppdc1 (C56) with prefill
# write_through ONLY; dryrun + tokdiff; queued after ppdc2 (waits for orch15 done), then orch14 ppdc3 (A1 only).
set -uo pipefail
W=/shared_nfs/yihou_glm52/g4tune_20261019; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; D=/shared_nfs/yihou_glm52/dcpx_20261022/ppdc
E=$W/env.ppdc3-135x136.yihou.sh
[[ -e $E ]] || { echo "# env.ppdc3-135x136.yihou.sh (2026-10-23, user) = env.ppdc1 with prefill HiCache write_back -> write_through ONLY (ablation; C56)."; grep -v "^# env.ppdc1" $W/env.ppdc1-135x136.yihou.sh | sed -e "s/YIHOU_ROUND=ppdc1/YIHOU_ROUND=ppdc3/" -e "s/ PREFILL_HICACHE_WRITE_POLICY=write_back / PREFILL_HICACHE_WRITE_POLICY=write_through /"; } > $E
grep -o "PREFILL_HICACHE_WRITE_POLICY=[a-z_]*\| CONC=[0-9]*" $E | tr '\n' ' '; echo
( source $E; cd $H; bash $H/run.v3x.yihou.sh dryrun ) > $W/dryrun.ppdc3-135x136.yihou.txt 2>&1 || echo DRYRUN_FAIL
echo "== tokdiff ppdc1 -> ppdc3"; diff <(tr ' ' '\n' < $W/dryrun.ppdc1-135x136.yihou.txt | sed 's/ppdc1/ROUND/g') <(tr ' ' '\n' < $W/dryrun.ppdc3-135x136.yihou.txt | sed 's/ppdc3/ROUND/g') | grep '^[<>]' | tr '\n' ' '; echo
cd /; setsid nohup bash -c "for i in \$(seq 1 600); do grep -q 'orch15 done' $D/orch15.ppdc2.yihou.log && break; sleep 30; done; grep -q 'orch15 done' $D/orch15.ppdc2.yihou.log && bash /shared_nfs/yihou_glm52/dcpx_20261022/dcpx11/orch14.yihou.sh ppdc3" > $D/orch14.ppdc3.yihou.log 2>&1 < /dev/null &
sleep 2; echo "queued: $(pgrep -af 'orch15 done' | head -1 | cut -c1-80)"
