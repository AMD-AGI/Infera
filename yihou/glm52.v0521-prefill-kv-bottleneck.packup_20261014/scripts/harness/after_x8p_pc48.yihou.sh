#!/usr/bin/env bash
# Task v0521cap2 (user 09:3x "好的"): on 274, let x8p A2 finish (its chain driver was killed so no A3/C52 starts), then stop
# the x8p stack and run pc48 (prefill_complete, HiCache 370 write_back, decode 0.96, C48: A1 fresh -> A2 scored).
set -uo pipefail
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export PATH=$N/shimbin:$PATH
ts(){ date -u +%FT%TZ; }
echo "$(ts) waiting for x8p A2 to end"
until grep -qE "bench	(ok|FAIL).*agentx-c48-A2" $R/x8p/status.tsv 2>/dev/null; do sleep 60; done
grep -a "agentx-c48-A2" $R/x8p/status.tsv | tail -1
echo "$(ts) stop x8p stack"
( cd $H && source $H/env.x8p-274.yihou.sh && bash $H/run.v3.yihou.sh stop ) || echo "$(ts) stop rc!=0 (checking containers)"
left=$(docker ps --format '{{.Names}}' | grep -c '^glm52-v0521-yihou-274-x8p-')
[[ "$left" == 0 ]] || { echo "$(ts) STOP: $left x8p containers still running"; exit 1; }
for i in $(seq 1 60); do   # engines gone; wait until no KFD process remains (chain_cap2 refuses otherwise)
  k=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ '); [[ "$k" == 0 ]] && break; sleep 30; done
echo "$(ts) x8p stopped (KFD procs=$k) -> pc48"
exec bash $H/chain_cap2.yihou.sh $H/env.pc48-274.yihou.sh 48 48 > $R/pc48.chain.log 2>&1 < /dev/null
